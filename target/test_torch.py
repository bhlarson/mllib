# Based on https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb

import argparse
import json
import os
import sys
import copy
import cv2
from tqdm import tqdm
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from datetime import datetime
from sklearn.metrics import confusion_matrix

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('')), '..')))
sys.path.insert(0, os.path.abspath(''))
#from segment.display import DrawFeatures, WritePredictions
from utils.s3 import s3store, Connect
from utils.jsonutil import WriteDictJson, ReadDictJson
from utils.similarity import jaccard, similarity
from datasets.cocostore import CocoDataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-min', action='store_true', help='If set, minimum training to generate output.')
    parser.add_argument('-min_steps', type=int, default=5, help='Number of min steps.')


    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
    parser.add_argument('-model_class', type=str,  default='segmin')

    parser.add_argument('-validationset', type=str, default='data/coco/annotations/instances_val2017.json', help='Coco dataset instance json file.')
    parser.add_argument('-train_image_path', type=str, default='data/coco/train2017', help='Coco image path for dataset.')
    parser.add_argument('-val_image_path', type=str, default='data/coco/val2017', help='Coco image path for dataset.')
    parser.add_argument('-class_dict', type=str, default='model/segmin/coco.json', help='Model class definition file.')

    parser.add_argument('-trtmodel', type=str, default='segment_nas_prune_640x640_20211015-fp16.trt', help='TRT file name')
    parser.add_argument('-tests_json', type=str, default='tests.json', help='Test Archive')

    parser.add_argument('-trainingset_dir', type=str, default='/store/test/coco', help='Path training set tfrecord')
    parser.add_argument('-test_dir', type=str, default='./test/segnas',help='Directory to store training model')

    parser.add_argument('--trainingset', type=str, default='2021-02-22-14-17-19-cocoseg', help='training set')
    parser.add_argument('-batch_size', type=int, default=1, help='Number of examples per batch.')
    parser.add_argument('-height', type=int, default=640, help='Batch image height')
    parser.add_argument('-width', type=int, default=640, help='Batch image width')
    parser.add_argument('-imflags', type=int, default=cv2.IMREAD_COLOR, help='cv2.imdecode flags')       

    parser.add_argument("-strategy", type=str, default='onedevice', help="Replication strategy. 'mirrored', 'onedevice' now supported ")
    parser.add_argument("-devices", type=json.loads, default=["/gpu:0"],  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')

    parser.add_argument('-training_crop', type=json.loads, default='[640, 640]', help='Training crop size [height, width]')
    parser.add_argument('-train_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB')
    parser.add_argument('-channel_order', type=str, default='channels_first', choices=['channels_first', 'channels_last'], help='Channels_last = NHWC, Tensorflow default, channels_first=NCHW')
    parser.add_argument('-fp16', action='store_true', help='If set, Generate FP16 model.')

    parser.add_argument('-savedmodel', type=str, default='./saved_model', help='Path to fcn savedmodel.')

    args = parser.parse_args()
    return args


def main(args):
    print('Start test')

    s3, creds, s3def = Connect(args.credentails)

    class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.class_dict)
    trainingset = '{}/{}/'.format(s3def['sets']['trainingset']['prefix'] , args.trainingset)
    print('Load training set {}/{} to {}'.format(s3def['sets']['trainingset']['bucket'],trainingset,args.trainingset_dir ))

    # Load dataset
    config = copy.deepcopy(args.__dict__)
    config['class_dictionary']= class_dictionary
    config['test_archive'] = trainingset

    valset = CocoDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], dataset_desc=args.validationset, 
        image_paths=args.val_image_path,
        class_dictionary=class_dictionary, 
        height=args.height, 
        width=args.width, 
        imflags=args.imflags, 
        astype='float32',
        enable_transform=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Prepare datasets for similarity computation
    objTypes = {}
    for objType in class_dictionary['objects']:
        if objType['trainId'] not in objTypes:
            objTypes[objType['trainId']] = copy.deepcopy(objType)
            # set name to category for objTypes and id to trainId
            objTypes[objType['trainId']]['name'] = objType['category']
            objTypes[objType['trainId']]['id'] = objType['trainId']

    results = {'class similarity':{}, 'config':config, 'image':[]}

    for objType in objTypes:
        results['class similarity'][objType] = {'union':0, 'intersection':0}

    modelobjname = '{}/{}/{}'.format(s3def['sets']['model']['prefix'], args.model_class, args.trtmodel)
    #modelfilename = '{}/{}/{}/{}'.format(args.test_dir, s3def['sets']['model']['prefix'], config['initialmodel'], config['trtmodel'])
    #print('Load trt model {}/{} to {}'.format(s3def['sets']['model']['bucket'], modelobjname, modelfilename))
    engine_obj = s3.GetObject(s3def['sets']['model']['bucket'], modelobjname)

    #accuracy = tf.keras.metrics.Accuracy()

    print("Begin inferences")
    dtSum = 0.0
    accuracySum = 0.0
    total_confusion = None
    numsteps = valloader.__len__()

    if(args.min):
        numsteps=min(args.min_steps, numsteps)

    step = 0
    color_chanels=3
    try:

        #f = open(modelfilename, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

        USE_FP16 = args.fp16
        target_dtype = np.float16 if USE_FP16 else np.float32
        dummy_input_batch = np.zeros((args.batch_size, color_chanels, args.width, args.height), dtype = np.float32) 

        engine = runtime.deserialize_cuda_engine(engine_obj)
        context = engine.create_execution_context()

        output = np.empty([args.batch_size, args.width, args.height, class_dictionary['classes']], dtype = target_dtype)
        # Allocate device memory
        d_input = cuda.mem_alloc(args.batch_size * dummy_input_batch.nbytes)
        d_output = cuda.mem_alloc(args.batch_size * output.nbytes)

        bindings = [int(d_input), int(d_output)]

        stream = cuda.Stream()

        def predict(batch): # result gets copied into output
            # Transfer input data to device
            cuda.memcpy_htod_async(d_input, batch, stream)
            # Execute model
            context.execute_async_v2(bindings, stream.handle, None)
            # Transfer predictions back
            cuda.memcpy_dtoh_async(output, d_output, stream)
            # Syncronize threads
            stream.synchronize()
                
            return output

        predict(dummy_input_batch) # Run to load dependencies
        predict(dummy_input_batch) # Run to load dependencies

        for step, data in tqdm(enumerate(valloader), total=valloader.__len__(), desc="Inference steps"):
            image, labels, mean, stdev = data

            initial = datetime.now()
            logitstft = predict(image.numpy())
            segmentationtrt = np.argmax(logitstft, axis=3)
            dt = (datetime.now()-initial).total_seconds()
            dtSum += dt
            imageTime = dt/config['batch_size']

            image = image.cpu().permute(0, 2, 3, 1).numpy() # Convert image to [batch, chanel, height, width]
            labels = labels.numpy()
            for j in range(config['batch_size']):

                img = np.squeeze(image[j])
                ann = np.squeeze(labels[j])
                seg = np.squeeze(segmentationtrt[j])

                font = cv2.FONT_HERSHEY_SIMPLEX

                iman = trainingset.coco.MergeIman(img, ann, mean[j].item(), stdev[j].item())
                imseg = trainingset.coco.MergeIman(img, seg, mean[j].item(), stdev[j].item())

                iman = cv2.putText(iman, 'Segmentation',(10,25), font, 1,(255,255,255),1,cv2.LINE_AA)
                imtrtseg = cv2.putText(imtrtseg, 'TensorRT',(10,25), font, 1,(255,255,255),1,cv2.LINE_AA)

                im = cv2.hconcat([iman, imtrtseg])
                im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                cv2.imwrite('{}/{}{:03d}{:03d}.png'.format(args.test_dir, 'seg', i, j), im_bgr)


                #accuracy.update_state(ann,seg)
                #seg_accuracy = accuracy.result().numpy()

                #accuracySum += seg_accuracy
                imagesimilarity, results['class similarity'], unique = jaccard(ann, seg, objTypes, results['class similarity'])

                confusion = confusion_matrix(ann.flatten(),seg.flatten(), range(class_dictionary['classes']))
                if total_confusion is None:
                    total_confusion = confusion
                else:
                    total_confusion += confusion
                        

                results['image'].append({'dt':imageTime,'similarity':imagesimilarity, 'accuracy':0.0, 'confusion':confusion.tolist()})
    except Exception as e:
        print("Error: test exception {} step {}".format(e, step))
        numsteps = step

    num_images = numsteps*config['batch_size']
    
    if numsteps > 0: 
        num_images = numsteps*config['batch_size']
        average_time = dtSum/num_images
        average_accuracy = accuracySum/num_images
    else:
        num_images = 0
        average_time = 0.0
        average_accuracy = 0.0

    sumIntersection = 0
    sumUnion = 0
    sumAccuracy = 0.0
    dataset_similarity = {}
    for key in results['class similarity']:
        intersection = results['class similarity'][key]['intersection']
        sumIntersection += intersection
        union = results['class similarity'][key]['union']
        sumUnion += union
        class_similarity = similarity(intersection, union)

        # convert to int from int64 for json.dumps
        dataset_similarity[key] = {'intersection':int(intersection) ,'union':int(union) , 'similarity':class_similarity}

    results['class similarity'] = dataset_similarity
    total_similarity = similarity(sumIntersection, sumUnion)

    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    test_summary = {'date':date_time, 'model':modelobjname}
    test_summary['accuracy']=average_accuracy
    test_summary['class_similarity']=dataset_similarity
    test_summary['similarity']=total_similarity
    if total_confusion is not None:
        test_summary['confusion']=total_confusion.tolist()
    else:
        test_summary['confusion']=None
    test_summary['images']=num_images
    test_summary['image time']=average_time
    test_summary['batch size']=config['batch_size']
    test_summary['test store'] =s3def['address']
    test_summary['test bucket'] = s3def['sets']['trainingset']['bucket']
    test_summary['results'] = results
    
    print ("Average time {}".format(average_time))
    print ('Similarity: {}'.format(dataset_similarity))

    # If there is a way to lock this object between read and write, it would prevent the possability of loosing data
    training_data = s3.GetDict(s3def['sets']['trainingset']['bucket'], config['test_archive']+args.tests_json)
    if training_data is None:
        training_data = []
    training_data.append(test_summary)
    s3.PutDict(s3def['sets']['trainingset']['bucket'], config['test_archive']+args.tests_json, training_data)

    test_url = s3.GetUrl(s3def['sets']['trainingset']['bucket'], config['test_archive']+args.tests_json)

    print("Test results {}".format(test_url))



if __name__ == '__main__':
    import argparse

    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy

        debugpy.listen(address=('0.0.0.0', args.debug_port))

        # Pause the program until a remote debugger is attached
        debugpy.wait_for_client()
        print("Debugger attached")

    main(args)
