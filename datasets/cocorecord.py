#%%
import os
import sys
import argparse
import json
import random
import math
import glob
import io
import tensorflow as tf
import cv2
from cocoio import CocoIO
from datetime import datetime
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-ann_dir', type=json.loads, default='[]')
    parser.add_argument('-record_dir', type=str, default='cocorecord', help='Path record work directory')

    parser.add_argument('-datasets', type=json.loads,
        default='[ \
                  { "jsonpath":"/store/Datasets/coco/instances_val2017.json", \
                    "imagepath":"/store/Datasets/coco/val2017", \
                    "name_decoration":"", \
                    "set":"val"}, \
                  { "jsonpath":"/store/Datasets/coco/instances_train2017.json", \
                    "imagepath":"/store/Datasets/coco/train2017", \
                    "name_decoration":"", \
                    "set":"train"} \
                ]',
        help='Json string containing an array of [{"jsonpath":"<>", "imagepath":"<>", "name_decoration":"<>","set":"<>",}]')

    defaultsetname = '{}-coco'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('-setname', type=str, default=defaultsetname, help='Path to training set directory')
    
    parser.add_argument('-shard_images', 
        type=int,
        default= 500,
        help='Number images per shard')

    parser.add_argument('-classes', type=json.loads, default='{}', help='Class dictionary JSON.  Leave empty if classes_file points to a JSON file.')
    parser.add_argument('-classes_file', type=str, default='datasets/coco.json', help='Class dictionary JSON file')

    parser.add_argument('-author', type=str, default='Brad Larson')
    parser.add_argument('-description', type=str, default='coco training set')

    parser.add_argument('-debug', action='store_true',help='Wait for debugge attach')

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

def Example(args, image, label):

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
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def Shuffle(sets):
    seed = random.random()
    for key in sets:
        random.Random(seed).shuffle(sets[key])

def WriteRecords(args):

    setDescriptions = []
    for dataset in args.datasets:
        coco = CocoIO(args.classes, dataset["jsonpath"], dataset["imagepath"], name_deccoration = dataset["name_decoration"])
        setDescriptions.append({'name':dataset['set'], 'length': coco.len()})
        shard_id = 0
        shardImages = 0
        shards =  math.ceil(float(coco.len())/float(args.shard_images))
        print('Processing {} dataset with {} images from {}'.format(dataset["set"], coco.len(), dataset["jsonpath"]))
        for i, iman in enumerate(tqdm(coco, total=coco.len())):

            if shardImages == 0:
                output_filename = os.path.join(args.record_dir, '{}-{:05d}-of-{:05d}.tfrecord'.format(dataset["set"], shard_id, shards))
                tfrecord_writer = tf.io.TFRecordWriter(output_filename)
            img = cv2.imread(iman['img'])        
            example = Example(args, img, iman['ann'])
            tfrecord_writer.write(example.SerializeToString())

            shardImages += 1
            if shardImages > args.shard_images:
                shardImages = 0
                shard_id += 1


    description = {'creation date':datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'author':args.author,'description':args.description, 'sets': setDescriptions, 'classes':args.classes}
    with open(args.record_dir+'/description.json', 'w') as fp:
        json.dump(description, fp, indent=4, separators=(',', ': '))

def main(args):  

    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    WriteRecords(args)
    
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