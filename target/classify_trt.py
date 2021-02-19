# Based on ONNXExample-2-2.ipynb

import argparse
import json
import os
import sys
import copy
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from datetime import datetime
#import matplotlib.pyplot as plt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('')), '..')))
sys.path.insert(0, os.path.abspath(''))
from classify.data import input_fn

sys.path.append('utils')
from s3 import s3store
from jsonutil import WriteDictJson, ReadDictJson
from similarity import jaccard, similarity

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
parser.add_argument('-dataset_dir', type=str, default='./dataset',help='Directory to store training model')
parser.add_argument('-saveonly', action='store_true', help='Do not train.  Only produce saved model')
parser.add_argument('-min', action='store_true', help='If set, minimum training to generate output.')
parser.add_argument('-min_steps', type=int, default=5, help='Number of min steps.')

parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')

parser.add_argument('-tests_json', type=str, default='tests.json', help='Test Archive')
parser.add_argument('-run_json', type=str, default='run.json', help='Test Archive')

parser.add_argument('-trainingset', type=str, default='2021-01-12-19-36-49-cocoseg', help='training set')
parser.add_argument('-trainingset_dir', type=str, default='/store/training/2021-01-12-19-36-49-cocoseg', help='Path training set tfrecord')

parser.add_argument('-batch_size', type=int, default=1, help='Number of examples per batch.')              

parser.add_argument("-strategy", type=str, default='onedevice', help="Replication strategy. 'mirrored', 'onedevice' now supported ")
parser.add_argument("-devices", type=json.loads, default=["/gpu:0"],  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')

parser.add_argument('-training_crop', type=json.loads, default='[480, 512]', help='Training crop size [height, width]')
parser.add_argument('-train_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB')
parser.add_argument('-channel_order', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Channels_last = NHWC, Tensorflow default, channels_first=NCHW')

parser.add_argument('-savedmodel', type=str, default='2021-01-13-18-16-49-cfy', help='Saved model to load if no checkpoint')


def main(args):
    print('Start test')

    creds = ReadDictJson(args.credentails)
    if not creds:
        print('Failed to load credentials file {}. Exiting'.format(args.credentails))
        return False

    s3def = creds['s3'][0]
    s3 = s3store(s3def['address'], 
                 s3def['access key'], 
                 s3def['secret key'], 
                 tls=s3def['tls'], 
                 cert_verify=s3def['cert_verify'], 
                 cert_path=s3def['cert_path']
                 )

    trainingset = '{}/{}/'.format(s3def['sets']['trainingset']['prefix'] , args.trainingset)
    print('Load training set {}/{} to {}'.format(s3def['sets']['trainingset']['bucket'],trainingset,args.trainingset_dir ))
    s3.Mirror(s3def['sets']['trainingset']['bucket'], trainingset, args.trainingset_dir)

    trainingsetDescriptionFile = '{}/description.json'.format(args.trainingset_dir)
    trainingsetDescription = json.load(open(trainingsetDescriptionFile))
    
    config = {
        's3_address':s3def['address'],
        's3_sets':s3def['sets'],
        'savedmodel': args.savedmodel,
        'training_dir': None,
        'shape':[],

        'batch_size': args.batch_size,
        'trainingset': trainingsetDescription,
        'input_shape': [args.training_crop[0], args.training_crop[1], args.train_depth],
        'classScale': 0.001, # scale value for each product class
        'augment_rotation' : 5., # Rotation in degrees
        'augment_flip_x': False,
        'augment_flip_y': True,
        'augment_brightness':0.,
        'augment_contrast': 0.,
        'augment_shift_x': 0.0, # in fraction of image
        'augment_shift_y': 0.0, # in fraction of image
        'scale_min': 0.75, # in fraction of image
        'scale_max': 1.25, # in fraction of image
        'ignore_label': trainingsetDescription['classes']['ignore'],
        'classes': trainingsetDescription['classes']['classes'],
        'epochs': 1,
        'area_filter_min': 25,
        'channel_order': args.channel_order,
        's3_address':s3def['address'],
        's3_sets':s3def['sets'],
        'training_dir': None, # used by LoadModel
        'learning_rate': 1e-3, # used by LoadModel
        'clean' : True,
        'test_archive': trainingset,
        'run_archive': '{}{}/'.format(trainingset, args.savedmodel),
        'min':args.min,
    }

    desc_obj = '{}/{}/description.json'.format(s3def['sets']['model']['prefix'], config['savedmodel'], config['savedmodel'])
    desc = s3.GetObject(s3def['sets']['model']['bucket'], desc_obj)
    trainingsetDescription = json.loads(desc.decode('utf-8'))

    config['shape'] = trainingsetDescription['config']['input_shape']

    # Prepare datasets for similarity computation
    objTypes = {}
    for objType in trainingsetDescription['config']['trainingset description']['classes']['objects']:
        if objType['trainId'] not in objTypes:
            objTypes[objType['trainId']] = copy.deepcopy(objType)
            # set name to category for objTypes and id to trainId
            objTypes[objType['trainId']]['name'] = objType['category']
            objTypes[objType['trainId']]['id'] = objType['trainId']

    modelname = '{}.trt'.format(config['savedmodel'])
    modelpath = '{}/{}/{}'.format(s3def['sets']['model']['prefix'], config['savedmodel'], modelname)
    #plan = s3.GetObject(s3def['sets']['model']['bucket'], modelpath)
    if not s3.GetFile(s3def['sets']['model']['bucket'], modelpath, modelname):
        print('Failed to load {}/{}'.format(s3def['sets']['model']['bucket'], modelpath))

    f = open(modelname, "rb")

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
    engine = runtime.deserialize_cuda_engine(f.read())
    #engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()

    output = np.empty(trainingsetDescription['config']['classes'], dtype = np.float32)

    # Allocate device memory
    dummy_input_batch = np.zeros((1, config['shape'][0], config['shape'][1], config['shape'][2]))
    d_input = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    def predict_batch(batch, output): # result gets copied into output
        # Transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # Execute model
        context.execute_async(1, bindings, stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # Syncronize threads
        stream.synchronize()
        return output

    predict_batch(dummy_input_batch, output)

    val_dataset = input_fn('val', args.trainingset_dir, config)

    train_images = config['batch_size'] # Guess training set if not provided
    val_images = config['batch_size']

    for dataset in trainingsetDescription['config']['trainingset description']['sets']:
        if(dataset['name']=="train"):
            train_images = dataset["length"]
        if(dataset['name']=="val"):
            val_images = dataset["length"]
    validation_steps=int(val_images/config['batch_size'])
    if(args.min):
        validation_steps=min(args.min_steps, validation_steps)

    print("Begin inferences")
    dtSum = 0.0
    errSum = 0.0
    total_confusion = None
    iterator = iter(val_dataset)

    results = {'class error':{}, 'config':config, 'image':[]}

    for objType in objTypes:
        results['class error'][objType] = {'err':0} 

    for i in tqdm(range(validation_steps)):
        image, annotation  = iterator.get_next()

        initial = datetime.now()
        predict_batch(image.numpy(), output) # Check TRT performance
        dt = (datetime.now()-initial).total_seconds()
        dtSum += dt
        imageTime = dt/config['batch_size']

        for j in range(config['batch_size']):
            annotation = annotation.numpy()
            err = ((annotation - output)**2).mean(axis=None) # axis=None returns scalar error value
            errSum += err                   
            results['image'].append({'dt':imageTime,'err':err, 'predict':output, 'annotation':annotation}) 

    num_images = numsteps*config['batch_size']
    average_time = dtSum/num_images
    average_error = errSum/num_images

    results['error'] = average_error
    s3.PutDict(s3def['sets']['trainingset']['bucket'], config['run_archive']+args.run_json, results)

    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    test_summary = {'date':date_time, 'model':config['savedmodel']}
    test_summary['accuracy']=average_accuracy
    test_summary['class_similarity']=dataset_similarity
    test_summary['similarity']=total_similarity
    test_summary['confusion']=total_confusion.tolist()
    test_summary['images']=num_images
    test_summary['image time']=average_time
    test_summary['batch size']=config['batch_size']
    test_summary['test store'] =s3def['address']
    test_summary['test bucket'] = s3def['sets']['trainingset']['bucket']
    test_summary['test object'] = config['run_archive']+args.run_json
        
    print ("Average time {}".format(average_time))
    print ('Similarity: {}'.format(dataset_similarity))

    # If there is a way to lock this object between read and write, it would prevent the possability of loosing data
    training_data = s3.GetDict(s3def['sets']['trainingset']['bucket'], config['test_archive']+args.tests_json)
    if training_data is None:
        training_data = []
    training_data.append(test_summary)
    s3.PutDict(s3def['sets']['trainingset']['bucket'], config['test_archive']+args.tests_json, training_data)

    test_url = s3.GetUrl(s3def['sets']['trainingset']['bucket'], config['test_archive']+args.tests_json)
    run_url = s3.GetUrl(s3def['sets']['trainingset']['bucket'], config['run_archive']+args.run_json)

    print("Test complete {}".format(test_summary))
    print(test_url)


    '''    #train_dataset = input_fn('train', args.trainingset_dir, config)
        val_dataset = input_fn('val', args.trainingset_dir, config)

        train_images = config['batch_size'] # Guess training set if not provided
        val_images = config['batch_size']

        for dataset in trainingsetDescription['sets']:
            if(dataset['name']=="train"):
                train_images = dataset["length"]
            if(dataset['name']=="val"):
                val_images = dataset["length"]
        validation_steps=int(val_images/config['batch_size'])

        if(args.min):
            validation_steps=min(args.min_steps, validation_steps)

        print("Begin inferences")
        dtSum = 0.0
        errSum = 0.0
        total_confusion = None
        iterator = iter(val_dataset)

        results = {'class error':{}, 'config':config, 'image':[]}

        for objType in objTypes:
            results['class error'][objType] = {'err':0} 

        m = tf.keras.metrics.MeanSquaredError()
        for i in tqdm(range(validation_steps)):
            image, annotation  = iterator.get_next()
            initial = datetime.now()
            logits = model.predict(image, batch_size=config['batch_size'], steps=1)
            dt = (datetime.now()-initial).total_seconds()
            dtSum += dt
            imageTime = dt/config['batch_size']
            m.update_state(annotation, logits)
            for j in range(config['batch_size']):
                m.update_state(annotation[j], logits[j])
                err = m.result().numpy()

                errSum += err
                    

                results['image'].append({'dt':imageTime,'err':err}) 

    num_images = numsteps*config['batch_size']
    average_time = dtSum/num_images
    average_error = errSum/num_images

    results['error'] = average_error
    s3.PutDict(s3def['sets']['trainingset']['bucket'], config['run_archive']+args.run_json, results)

    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    test_summary = {'date':date_time, 'model':config['initialmodel']}
    test_summary['accuracy']=average_accuracy
    test_summary['class_similarity']=dataset_similarity
    test_summary['similarity']=total_similarity
    test_summary['confusion']=total_confusion.tolist()
    test_summary['images']=num_images
    test_summary['image time']=average_time
    test_summary['batch size']=config['batch_size']
    test_summary['test store'] =s3def['address']
    test_summary['test bucket'] = s3def['sets']['trainingset']['bucket']
    test_summary['test object'] = config['run_archive']+args.run_json
    
    print ("Average time {}".format(average_time))
    print ('Similarity: {}'.format(dataset_similarity))

    # If there is a way to lock this object between read and write, it would prevent the possability of loosing data
    training_data = s3.GetDict(s3def['sets']['trainingset']['bucket'], config['test_archive']+args.tests_json)
    if training_data is None:
        training_data = []
    training_data.append(test_summary)
    s3.PutDict(s3def['sets']['trainingset']['bucket'], config['test_archive']+args.tests_json, training_data)

    test_url = s3.GetUrl(s3def['sets']['trainingset']['bucket'], config['test_archive']+args.tests_json)
    run_url = s3.GetUrl(s3def['sets']['trainingset']['bucket'], config['run_archive']+args.run_json)

    print("Test complete {}".format(test_summary))
    print(test_url)

    #---------------------------------------------------------------------------------

    f = open("resnet_engine.trt", "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    output = np.empty(1000, dtype = np.float32)

    # Allocate device memory
    d_input = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    def predict_batch(batch): # result gets copied into output
        # Transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # Execute model
        context.execute_async(1, bindings, stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # Syncronize threads
        stream.synchronize()

    print("Warming up...")

    predict_batch(dummy_input_batch)

    print("Done warming up!")

    %%time

    predict_batch(dummy_input_batch) # Check TRT performance

    print ("Prediction: " + str(np.argmax(output))) '''



if __name__ == '__main__':
  args, unparsed = parser.parse_known_args()
  
  if args.debug:
      print("Wait for debugger attach")
      import ptvsd
      # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
      # Launch applicaiton on remote computer: 
      # > python3 -m ptvsd -host 10.150.41.30 -port 3000 -wait fcn/train.py
      # Allow other computers to attach to ptvsd at this IP address and port.
      ptvsd.enable_attach(address=('0.0.0.0', args.debug_port), redirect_output=True)
      # Pause the program until a remote debugger is attached

      ptvsd.wait_for_attach()

      print("Debugger attach")

  main(args)
