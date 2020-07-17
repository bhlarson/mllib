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
from datetime import datetime
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('--ann_dir', type=json.loads, default='[]')
    parser.add_argument('--record_dir', type=str, default='cocorecord', help='Path record work directory')

    parser.add_argument('--sets', type=json.loads,
        default='[{"name":"train"}, {"name":"val"}, {"name":"test"}]',
        help='Json string containing an array of [{"name":"<>", "ratio":<probability>}]')

    defaultsetname = '{}-coco'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--setname', type=str, default=defaultsetname, help='Path to training set directory')

    parser.add_argument('--image_extension', type=str, default='_leftImg8bit.png', help='Expect tiff image type with microscope data tags')
    parser.add_argument('--annotation_extension', type=str, default='_gtFine_labelIds.png', help='Annotation decoration e.g.: _cls.png')
    
    parser.add_argument('--shards', 
        type=int,
        default= 10,
        help='Number of tfrecord shards')

    parser.add_argument('--image_format', 
        type=str,
        default='tif',
        help='Image format.')

    parser.add_argument('--classes', type=json.loads, default='{}', help='Class dictionary JSON.  Leave empty if classes_file points to a JSON file.')
    parser.add_argument('--classes_file', type=str, default='datasets/coco.json', help='Class dictionary JSON file')

    parser.add_argument('--img', type=json.loads, default='["/store/Datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit"]', help='Cityscape image path')
    parser.add_argument('--ann', type=json.loads, default='["/store/Datasets/cityscapes/gtFine_trainvaltest/gtFine"]', help='Cityscape annotations path')
    parser.add_argument('--dest', type=str, default='/store/Datasets/cityscapes/trainingset', help='Trainingset location')

    parser.add_argument('--author', type=str, default='Brad Larson')
    parser.add_argument('--description', type=str, default='coco to record conversion')

    parser.add_argument('--debug', action='store_true',help='Wait for debugge attach')

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

def Example(args, iman):
    image = cv2.imread(iman['im'],cv2.IMREAD_COLOR )
    label = cv2.imread(iman['an'],cv2.IMREAD_GRAYSCALE )

    if image is None:
        raise ValueError('{} annotation failed to load '.format(iman['im']))

    if label is None:
        raise ValueError('{} annotation failed to load '.format(iman['an']))

    # convert label lable id to trainId pixel value
    for objecttype in args.classes['objects']:
        label[label == objecttype['id']] = objecttype['trainId']


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

def WriteRecords(args, iman):

    Shuffle(iman)

    for dataset in iman:
        start = 0
        numEntries = len(iman[dataset])
        setdata = next((aset for aset in args.sets if aset['name'] == dataset), None) 
        setdata['length']=numEntries
        for shard_id in range(args.shards):
            output_filename = os.path.join(args.record_dir, '{}-{:05d}-of-{:05d}.tfrecord'.format(dataset, shard_id, args.shards))
            if(shard_id == args.shards-1):
                stop = numEntries
            else:
                groupSize = int(numEntries/args.shards)
                stop = start+groupSize

            print('{} start {} stop {}'.format(output_filename, start, stop))
            with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i in tqdm(range(start,stop)):
                    example = Example(args, iman[dataset][i])
                    tfrecord_writer.write(example.SerializeToString())

            start = stop

            sys.stdout.write('\n')
            sys.stdout.flush()

    description = {'creation date':datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'author':args.author,'description':args.description, 'sets': args.sets, 'classes':args.classes}
    with open(args.record_dir+'/description.json', 'w') as fp:
        json.dump(description, fp)

def PutDir(s3, src, bucket, setname):
    success = True
    files = glob.glob(glob.escape(src)+'/*.*')
    try:
        s3.make_bucket(bucket)
    except BucketAlreadyOwnedByYou as err:
        pass
    except BucketAlreadyExists as err:
        pass
    except ResponseError as err:
        print(err)
        raise

    try:
        for file in files:
            filename = setname+'/'+os.path.basename(file)
            s3.fput_object(bucket, filename, file)
    except ResponseError as err:
       print(err)

    return success

def list_dirs(top, maxdepth=1):
    dirs, nondirs = [], []

    for iTop in top:
        entries = os.scandir(iTop)
        for entry in entries:
            (dirs if entry.is_dir() else nondirs).append(entry.path)

    if maxdepth > 1:
        outDirs = list_dirs(dirs, maxdepth-1)
    elif maxdepth == 1:
        outDirs = dirs
    else:
        outDirs = top

    return outDirs

def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]
def remove_suffix(text, suffix):
    if not text.endswith(suffix):
        return text
    return text[:len(text)-len(suffix)]

def list_iman(sets, imgPaths, annPaths):
    iman = {}
    imgs = []
    # Extract list of images and annotations
    for ids, dataset in enumerate(sets):
        iman[dataset['name']] = []
        for iPath, imgPath in enumerate(imgPaths):
            dirs = ['{}/{}'.format(imgPath, dataset['name'])]
            srcImgDirs = list_dirs(dirs)
            for imgdir in srcImgDirs:
                imgs.extend(glob.glob(glob.escape(imgdir)+'/*'+args.image_extension))

            for img in imgs:
                interDir = remove_prefix(img, imgPath)
                interDir = remove_suffix(interDir, args.image_extension)

                ann = '{}{}{}'.format(annPaths[iPath],interDir,args.annotation_extension)
                if(os.path.exists(img) and os.path.exists(ann)):
                    iman[dataset['name']].append({'im':img, 'an':ann})

    return iman   

def main(args):  
    iman = list_iman(args.sets, args.img, args.ann)

    if(len(iman)<1):
        print('No files found {}  Exiting'.format(args.annotations))
        return

    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    WriteRecords(args, iman)
    
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