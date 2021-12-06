# tf2-onnx.py -> onnx-trt.py -> trt-tst.py

import argparse
import json
import os
import sys
import copy
import cv2
import platform
from tqdm import tqdm
import numpy as np
import torch
#import tensorrt as trt
#import pycuda.driver as cuda
import pycuda.autoinit
from datetime import datetime

sys.path.insert(0, os.path.abspath(''))
from utils.jsonutil import ReadDictJson
from utils.s3 import s3store, Connect
from datasets.cocostore import CocoDataset
from utils.metrics import jaccard, similarity, confusionmatrix, DatasetResults
from datasets.cocostore import CocoDataset
from target.trtinference import TrtInference

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-fast', action='store_true', help='Fast run with a few iterations')
    parser.add_argument('-fast_steps', type=int, default=5, help='Number of min steps.')

    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
    parser.add_argument('-model_type', type=str,  default='segmentation')
    parser.add_argument('-model_class', type=str,  default='segmin')
    parser.add_argument('-model', type=str,  default='segment_nas_512x442_20211126_00fp32.trt', help=' Model to test')
    parser.add_argument('-height', type=int, default=480, help='Batch image height')
    parser.add_argument('-width', type=int, default=512, help='Batch image width')
    parser.add_argument('-imflags', type=int, default=cv2.IMREAD_COLOR, help='cv2.imdecode flags')
    parser.add_argument('-batch_size', type=int, default=1, help='Number of examples per batch.')

    parser.add_argument('-validationset', type=str, default='data/coco/annotations/instances_val2017.json', help='Coco dataset instance json file.')
    parser.add_argument('-val_image_path', type=str, default='data/coco/val2017', help='Coco image path for dataset.')
    parser.add_argument('-class_dict', type=str, default='model/deeplabv3/coco.json', help='Model class definition file.')

    parser.add_argument('-test_results', type=str, default='test_results.json')
    parser.add_argument('-test_dir', type=str, default=None,help='Directory to store training model')
    parser.add_argument('-imStatistics', type=bool, default=False, help='Record individual image statistics')
            
    parser.add_argument('-num_workers', type=int, default=0, help='Number of workers.')    
    parser.add_argument('--description', type=str, default='Test TensorRT model', help='Test TensorRT model inference')

    args = parser.parse_args()
    return args



def main(args):

    system = {
        'platform':platform.platform(),
        'python':platform.python_version(),
        'tensorrt version': sys.modules['tensorrt'].__version__,
        'numpy version': sys.modules['numpy'].__version__,
    }

    print('system={}'.format(system))

    s3, creds, s3def = Connect(args.credentails)
    class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.class_dict)

    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    test_summary = {'date':date_time}

    trtobjname = '{}/{}/{}'.format(s3def['sets']['model']['prefix'], args.model_class, args.model)
    print('Load TensorRT model {}/{}'.format(s3def['sets']['model']['bucket'], trtobjname))
    trtserializedmodel = s3.GetObject(s3def['sets']['model']['bucket'], trtobjname)
    if not trtserializedmodel:
        print('Failed to download onnx model bucket:{} object:{}.  Exiting'.format(s3def['sets']['model']['bucket'], trtobjname))
        return -1

    # Load dataset
    testset = CocoDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], dataset_desc=args.validationset, 
        image_paths=args.val_image_path,
        class_dictionary=class_dictionary, 
        height=args.height, 
        width=args.width, 
        imflags=args.imflags,
        astype='float32',
        enable_transform=False)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.test_dir is not None:
        outputdir = '{}/{}'.format(args.test_dir,args.model_class)
        os.makedirs(outputdir, exist_ok=True)
    else:
        outputdir = None

    dsResults = DatasetResults(class_dictionary, args.batch_size, imStatistics=args.imStatistics, imgSave=outputdir)

    inf = TrtInference(trtserializedmodel, args.batch_size, args.height, args.width, class_dictionary)

    #dummy_input_batch = np.zeros((args.batch_size, args.height, args.width, 3), dtype = np.float32) 

    #runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
    #engine = runtime.deserialize_cuda_engine(trtserializedmodel)
    #context = engine.create_execution_context()
    #output = np.empty([args.batch_size, args.height, args.width, len(class_dictionary)], dtype = np.float32) # Need to set output dtype to FP16 to enable FP16

    # Allocate device memory
    #d_input = cuda.mem_alloc(dummy_input_batch.nbytes)
    #d_output = cuda.mem_alloc(output.nbytes)

    #bindings = [int(d_input), int(d_output)]

    #stream = cuda.Stream()

    #def predict(batch): # result gets copied into output
    #    # Transfer input data to device
    #    cuda.memcpy_htod_async(d_input, batch, stream)
    #    # Execute model
    #    context.execute_async_v2(bindings, stream.handle, None)
    #    # Transfer predictions back
    #    cuda.memcpy_dtoh_async(output, d_output, stream)
    #    # Syncronize threads
    #    stream.synchronize()
    #    
    #    return output

    #print("Warming up...")
    #pred = predict(dummy_input_batch)
    #print("Done warming up!")

    try:
        for i, data in tqdm(enumerate(testloader), total=int(testset.__len__()/args.batch_size), desc="Inference steps"):
            images, labels, mean, stdev = data

            initial = datetime.now()
            segmentations = inf.predict(images.cpu().numpy().astype(np.float32))
            #predtrt = predict(images.cpu().numpy().astype(np.float32))
            #segmentations = np.argmax(predtrt, axis=-1).astype('uint8')
            dt = (datetime.now()-initial).total_seconds()
            imageTime = dt/args.batch_size

            images = images.cpu().permute(0, 2, 3, 1).numpy()
            labels = np.around(labels.cpu().numpy()).astype('uint8')

            totalConfusion = dsResults.infer_results(i, images, labels, segmentations, mean.numpy(), stdev.numpy(), dt)

            iPrint = 100
            if i % iPrint == iPrint-1:    # print every iPrint iterations
                tqdm.write('Total confusion:\n {}'.format(totalConfusion))

            if args.fast and i+1 >= args.fast_steps:
                break
    except Exception as e:
        print("Error: test exception {} step {}".format(e, i))
        return -1

    test_summary['objects'] = dsResults.objTypes
    test_summary['object store'] =s3def
    test_summary['results'] = dsResults.Results()
    test_summary['config'] = args.__dict__
    test_summary['system'] = system

    # If there is a way to lock this object between read and write, it would prevent the possability of loosing data
    test_path = '{}/{}/{}'.format(s3def['sets']['test']['prefix'], args.model_type, args.test_results)
    training_data = s3.GetDict(s3def['sets']['test']['bucket'], test_path)
    if training_data is None or type(training_data) is not list:
        training_data = []
    training_data.append(test_summary)
    s3.PutDict(s3def['sets']['test']['bucket'], test_path, training_data)

    test_url = s3.GetUrl(s3def['sets']['test']['bucket'], test_path)
    print("Test results {}".format(test_url))

    return 0


if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy

        debugpy.listen(address=('0.0.0.0', args.debug_port))

        # Pause the program until a remote debugger is attached
        debugpy.wait_for_client()
        print("Debugger attached")

    result = main(args)
    sys.exit(result)
