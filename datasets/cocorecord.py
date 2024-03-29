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
from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import WriteDictJson

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-dataset', type=str, default='coco', help='Dataset.')
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
        default= 1000,
        help='Number images per shard')

    parser.add_argument('-classes', type=json.loads, default='{}', help='Class dictionary JSON.  Leave empty if classes_file points to a JSON file.')
    parser.add_argument('-classes_file', type=str, default='datasets/coco.json', help='Class dictionary JSON file')

    parser.add_argument('-author', type=str, default='Brad Larson')
    parser.add_argument('-description', type=str, default='coco 2017 set for training image segmentation')

    parser.add_argument('-debug', action='store_true',help='Wait for debugge attach')
    parser.add_argument('-min', action='store_true', help='If set, minimum training to generate output.')
    parser.add_argument('-min_steps', type=int, default=20, help='Number of min steps.')
    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')


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
        shard_id = 0
        shardImages = 0
        set_len = 0
        set_images = coco.len()
        if args.min and set_images >= args.min_steps:
            set_images = args.min_steps

        shards =  math.ceil(float(set_images)/float(args.shard_images))
        print('Processing {} dataset with {} images from {}'.format(dataset["set"], set_images, jsonpath))
        tfrecord_writer = None
        for i, iman in enumerate(tqdm(coco, total=set_images)):
            imgbuff = s3.GetObject(s3def['sets']['dataset']['bucket'], iman['img'])
            if imgbuff:
                imgbuff = np.fromstring(imgbuff, dtype='uint8')
                img = cv2.imdecode(imgbuff, cv2.IMREAD_COLOR)      
                example = Example(args, img, iman['ann'], iman['classes'])
                
                if tfrecord_writer is None:
                    output_filename = os.path.join(args.record_dir, '{}-{:05d}-of-{:05d}.tfrecord'.format(dataset["set"], shard_id+1, shards))
                    tfrecord_writer = tf.io.TFRecordWriter(output_filename)

                tfrecord_writer.write(example.SerializeToString())

                shardImages += 1
                set_len += 1

                if shardImages >= args.shard_images:
                    shardImages = 0
                    shard_id += 1

                    tfrecord_writer.flush()
                    tfrecord_writer.close()
                    tfrecord_writer = None

                if set_len >= set_images: # if set_images < len(coco), we need to exit the loop
                    break

        if tfrecord_writer is not None:
            tfrecord_writer.flush()
            tfrecord_writer.close()
        setDescriptions.append({'name':dataset['set'], 'length': set_len})


    description = {
        'creation date':datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        'author':args.author,
        'description':args.description, 
        'sets': setDescriptions, 
        'classes':args.classes
        }
    with open(args.record_dir+'/description.json', 'w') as fp:
        json.dump(description, fp, indent=4, separators=(',', ': '))

    saved_name = '{}/{}'.format(s3def['sets']['trainingset']['prefix'] , args.trainingset_name)
    print('Save trainingset to the object store {}/{}'.format(s3def['sets']['trainingset']['bucket'],saved_name))
    if s3.PutDir(s3def['sets']['trainingset']['bucket'], args.record_dir, saved_name):
        shutil.rmtree(args.record_dir, ignore_errors=True)

def main(args):

    s3, creds, s3def = Connect(args.credentails)

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