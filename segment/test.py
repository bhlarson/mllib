# Based on https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb

import argparse
import json
import os
import sys
import copy
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from datetime import datetime

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('')), '..')))
sys.path.insert(0, os.path.abspath(''))
from segment.data import input_fn
from segment.display import WritePredictions
from networks.unet import unet_model

sys.path.append('utils')
from s3 import s3store
from jsonutil import WriteDictJson, ReadDictJson
from loadmodel import LoadModel
from similarity import jaccard, similarity


parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
parser.add_argument('-min', action='store_true', help='If set, minimum training to generate output.')
parser.add_argument('-min_steps', type=int, default=5, help='Number of min steps.')

parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')

parser.add_argument('-initialmodel', type=str, default='2021-02-24-10-28-35-cocoseg', help='Initial model.  Empty string if no initial model')
parser.add_argument('-tests_json', type=str, default='tests.json', help='Test Archive')

parser.add_argument('-trainingset_dir', type=str, default='/store/training/coco', help='Path training set tfrecord')
parser.add_argument('-trainingset', type=str, default='coco', help='training set')

parser.add_argument('-batch_size', type=int, default=1, help='Number of examples per batch.')              

parser.add_argument("-strategy", type=str, default='onedevice', help="Replication strategy. 'mirrored', 'onedevice' now supported ")
parser.add_argument("-devices", type=json.loads, default=["/gpu:0"],  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')

parser.add_argument('-training_crop', type=json.loads, default='[480, 512]', help='Training crop size [height, width]')
parser.add_argument('-train_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB')
parser.add_argument('-channel_order', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Channels_last = NHWC, Tensorflow default, channels_first=NCHW')

parser.add_argument('-savedmodel', type=str, default='./saved_model', help='Path to fcn savedmodel.')

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
        'weights': None,
        'channel_order': args.channel_order,
        's3_address':s3def['address'],
        's3_sets':s3def['sets'],
        'initialmodel':args.initialmodel,
        'training_dir': None, # used by LoadModel
        'learning_rate': 1e-3, # used by LoadModel
        'clean' : True,
        'test_archive': trainingset,
        'run_archive': '{}{}/'.format(trainingset, args.initialmodel),
        'min':args.min,
    }

    trainingsetDescriptionFile = '{}/description.json'.format(args.trainingset_dir)
    trainingsetDescription = json.load(open(trainingsetDescriptionFile))

    strategy = None
    if(args.strategy == 'mirrored'):
        strategy = tf.distribute.MirroredStrategy(devices=args.devices)

    else:
        device = "/gpu:0"
        if args.devices is not None and len(args.devices) > 0:
            device = args.devices[0]

        strategy = tf.distribute.OneDeviceStrategy(device=device)

    # Prepare datasets for similarity computation
    objTypes = {}
    for objType in trainingsetDescription['classes']['objects']:
        if objType['trainId'] not in objTypes:
            objTypes[objType['trainId']] = copy.deepcopy(objType)
            # set name to category for objTypes and id to trainId
            objTypes[objType['trainId']]['name'] = objType['category']
            objTypes[objType['trainId']]['id'] = objType['trainId']

    results = {'class similarity':{}, 'config':config, 'image':[]}

    for objType in objTypes:
        results['class similarity'][objType] = {'union':0, 'intersection':0} 

    with strategy.scope(): # Apply training strategy 
        model =  LoadModel(config, s3)
        accuracy = tf.keras.metrics.Accuracy()

        # Display model
        model.summary()

        #train_dataset = input_fn('train', args.trainingset_dir, config)
        val_dataset = input_fn('val', args.trainingset_dir, config)

        trainingsetdesc = {}
        validationsetdec = {}
        for dataset in config['trainingset']['sets']:
            if dataset['name'] == 'val':
                validationsetdec = dataset
            if dataset['name'] == 'train':
                trainingsetdesc = dataset

        print("Begin inferences")
        dtSum = 0.0
        accuracySum = 0.0
        total_confusion = None
        iterator = iter(val_dataset)
        numsteps = int(validationsetdec['length']/config['batch_size'])

        if(config['min']):
            numsteps=min(args.min_steps, numsteps)

        try:
            for i in tqdm(range(numsteps)):
                image, annotation  = iterator.get_next()
                initial = datetime.now()
                logits = model.predict(image, batch_size=config['batch_size'], steps=1)
                segmentation = tf.argmax(logits, axis=-1)
                dt = (datetime.now()-initial).total_seconds()
                dtSum += dt
                imageTime = dt/config['batch_size']
                for j in range(config['batch_size']):
                    img = tf.squeeze(image[j]).numpy().astype(np.uint8)
                    ann = tf.squeeze(annotation[j]).numpy().astype(np.uint8)
                    seg = tf.squeeze(segmentation[j]).numpy().astype(np.uint8)

                    accuracy.update_state(ann,seg)
                    seg_accuracy = accuracy.result().numpy()

                    accuracySum += seg_accuracy
                    imagesimilarity, results['class similarity'], unique = jaccard(ann, seg, objTypes, results['class similarity'])

                    confusion = tf.math.confusion_matrix(ann.flatten(),seg.flatten(), config['classes']).numpy().astype(np.int64)
                    if total_confusion is None:
                        total_confusion = confusion
                    else:
                        total_confusion += confusion
                        

                    results['image'].append({'dt':imageTime,'similarity':imagesimilarity, 'accuracy':seg_accuracy.astype(float), 'confusion':confusion.tolist()})
        except Exception as e:
            print("Error: test exception {} step {}".format(e, i))
            numsteps = i
        except:
            print("Error: test exception step {}".format(i))
            numsteps = i

    num_images = numsteps*config['batch_size']
    average_time = dtSum/num_images
    average_accuracy = accuracySum/num_images
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
