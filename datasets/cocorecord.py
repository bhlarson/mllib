#%%
import os
import sys
import argparse
import json
import random
import math
import glob
import io
import shutil
import numpy as np
import tensorflow as tf
import cv2
from cocoio import CocoIO
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(''))
from utils.s3 import s3store
from utils.jsonutil import WriteDictJson

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Path to dataset direcotry')
    parser.add_argument('-record_dir', type=str, default='./record', help='Path record work directory')

    defaultname = '{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-cocoseg'))
    parser.add_argument('-trainingset_name', type=str, default=defaultname, help='Training set name')

    parser.add_argument('-datasets', type=json.loads,
        default='[ \
                  { "jsonpath":"annotations/instances_train2017.json", \
                    "imagepath":"train2017", \
                    "name_decoration":"", \
                    "set":"train" \
                  }, \
                  { "jsonpath":"annotations/instances_val2017.json", \
                    "imagepath":"val2017", \
                    "name_decoration":"", \
                    "set":"val" \
                  } \
                ]',
        help='Json string containing an array of [{"jsonpath":"<>", "imagepath":"<>", "name_decoration":"<>","set":"<>",}]')
   
    parser.add_argument('-shard_images', 
        type=int,
        default= 500,
        help='Number images per shard')

    parser.add_argument('-classes', type=json.loads, default='{}', help='Class dictionary JSON.  Leave empty if classes_file points to a JSON file.')
    parser.add_argument('-classes_file', type=str, default='datasets/coco.json', help='Class dictionary JSON file')

    parser.add_argument('-author', type=str, default='Brad Larson')
    parser.add_argument('-description', type=str, default='coco 2017 set for training image segmentation')

    parser.add_argument('-debug', action='store_true',help='Wait for debugge attach')
    parser.add_argument('-min', action='store_true', help='If set, minimum training to generate output.')
    parser.add_argument('-min_steps', type=int, default=5, help='Number of min steps.')
    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
    parser.add_argument('-dataset', type=str, default='coco', help='Dataset.')

    args = parser.parse_args()
    return args

def feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _str_feature(value):
    return _bytes_feature(str.encode(value))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _str_feature(value):
    return _bytes_feature(str.encode(value))

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def Example(args, image, label, classes):

    if image is None:
        raise ValueError('{} annotation failed to load '.format(iman['im']))

    if label is None:
        raise ValueError('{} annotation failed to load '.format(iman['an']))

    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    retval, buffer = cv2.imencode('.png', image, compression_params)
    encoded_image = io.BytesIO(buffer).read()

    retval, buffer = cv2.imencode('.png', label, compression_params)
    encoded_label =io.BytesIO(buffer).read()

    height, width, depth = image.shape
    heightL, widthL = label.shape
    if width!= widthL or height != heightL:
        raise ValueError('image {} size != annotation {} size '.format(iman['im'], iman['an']))

    feature = {
        'image/height': _int_feature([height]),
        'image/width': _int_feature([width]),
        'image/depth': _int_feature([depth]),
        'image/encoded': _bytes_feature(encoded_image),
        'image/format': _str_feature('png'),
        'label/encoded': _bytes_feature(encoded_label),
        'label/format': _str_feature('png'),
        'label/classes': _float_feature(classes),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def Shuffle(sets):
    seed = random.random()
    for key in sets:
        random.Random(seed).shuffle(sets[key])

def WriteRecords(s3def, s3, args):
    setDescriptions = []
    for dataset in args.datasets:
        jsonpath = '{}/{}/{}'.format(s3def['sets']['dataset']['prefix'],args.dataset, dataset["jsonpath"])
        annotations = s3.GetDict(s3def['sets']['dataset']['bucket'], jsonpath)
        imagepath = '{}/{}/{}'.format(s3def['sets']['dataset']['prefix'], args.dataset, dataset["imagepath"])
        coco = CocoIO(args.classes, annotations, imagepath, name_deccoration = dataset["name_decoration"])
        setDescriptions.append({'name':dataset['set'], 'length': coco.len()})
        shard_id = 0
        shardImages = 0
        shards =  math.ceil(float(coco.len())/float(args.shard_images))
        print('Processing {} dataset with {} images from {}'.format(dataset["set"], coco.len(), jsonpath))
        for i, iman in enumerate(tqdm(coco, total=coco.len())):

            if shardImages == 0:
                output_filename = os.path.join(args.record_dir, '{}-{:05d}-of-{:05d}.tfrecord'.format(dataset["set"], shard_id, shards))
                tfrecord_writer = tf.io.TFRecordWriter(output_filename)
            imgbuff = s3.GetObject(s3def['sets']['dataset']['bucket'], iman['img'])
            imgbuff = np.fromstring(imgbuff, dtype='uint8')
            img = cv2.imdecode(imgbuff, cv2.IMREAD_COLOR)        
            example = Example(args, img, iman['ann'], iman['classes'])
            tfrecord_writer.write(example.SerializeToString())

            shardImages += 1
            if shardImages >= args.shard_images:
                print('{} {}'.format(output_filename, shardImages))
                shardImages = 0
                shard_id += 1

            if args.min and i >= args.min_steps:
                break


    description = {'creation date':datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'author':args.author,'description':args.description, 'sets': setDescriptions, 'classes':args.classes}
    with open(args.record_dir+'/description.json', 'w') as fp:
        json.dump(description, fp, indent=4, separators=(',', ': '))

    saved_name = '{}/{}'.format(s3def['sets']['trainingset']['prefix'] , args.trainingset_name)
    print('Save model to {}/{}'.format(s3def['sets']['trainingset']['bucket'],saved_name))
    if s3.PutDir(s3def['sets']['trainingset']['bucket'], args.record_dir, saved_name):
        shutil.rmtree(args.record_dir, ignore_errors=True)

def main(args):

    creds = {}
    with open(args.credentails) as json_file:
        creds = json.load(json_file)
    if not creds:
        print('Failed to load credentials file {}. Exiting'.format(args.credentails))

    s3def = creds['s3'][0]
    s3 = s3store(s3def['address'], 
                 s3def['access key'], 
                 s3def['secret key'], 
                 tls=s3def['tls'], 
                 cert_verify=s3def['cert_verify'], 
                 cert_path=s3def['cert_path']
                 )

    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    WriteRecords(s3def, s3, args)
    
if __name__ == '__main__':
    args = parse_arguments()
    if not args.classes and args.classes_file is not None :
        if '.json' in args.classes_file:
            args.classes = json.load(open(args.classes_file))

    if args.debug:
        print("Wait for debugger attach")
        import ptvsd
        # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        # Launch applicaiton on remote computer: 
        # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait fcn/train.py
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=('0.0.0.0', 3000), redirect_output=True)
        # Pause the program until a remote debugger is attached
        ptvsd.wait_for_attach()
        print("Debugger attached")

    main(args)
    print('{} exit'.format(sys.argv[0]))