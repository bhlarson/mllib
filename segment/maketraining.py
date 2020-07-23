import os
import sys
import argparse
import json
import random
import math
import glob
import io
import shutil
import tensorflow as tf
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import tifffile
import xmltodict
from minio import Minio
sys.path.insert(0, os.path.abspath(''))
from utils.s3 import s3store
from minio.error import (ResponseError, BucketAlreadyOwnedByYou,
                         BucketAlreadyExists)
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-annotations', type=json.loads, help='Array of annotation paths to include: ["<path1>", "<path2>", "<path3>"]',
        default=['Annotations'
                 ])

    parser.add_argument('-ann_dir', type=str, default='ann')
    parser.add_argument('-record_dir', type=str, default='record', help='Path record work directory')

    defaultsetname = '{}-lit'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('-setname', type=str, default=defaultsetname, help='Path to training set directory')

    parser.add_argument('-image_type', type=str, default='tif', help='Expect tiff image type with microscope data tags')

    parser.add_argument('-annotation_decoration', type=str, default='_cls.png', help='Annotation decoration e.g.: _cls.png')

    parser.add_argument('-sets', type=json.loads,
        default='[{"name":"training", "ratio":0.7}, {"name":"validation", "ratio":0.3}]',
        help='Json string containing an array of [{"name":"<>", "ratio":<probability>}]')

    parser.add_argument('-seed', 
        type=float, 
        default=None, 
        help='Random float seed')
    
    parser.add_argument('-shards', 
        type=int,
        default= 1,
        help='Number of tfrecord shards')

    parser.add_argument('-size', type=int,
        default= 200,
        help='Image pizel size')

    parser.add_argument('-show', 
        type=bool,
        default=False,
        help='Display incremental results')

    parser.add_argument('-image_format', 
        type=str,
        default='tif',
        help='Image format.')

    parser.add_argument('-minio_address', type=str, default='192.168.1.66:19002', help='Minio archive IP address')
    parser.add_argument('-minio_access_key', type=str, default='access', help='Minio access key')
    parser.add_argument('-minio_secret_key', type=str, default='secretkey', help='Minio secret key')

    parser.add_argument('-srcbucket', type=str, default='annotations', help='Annotations bucket')
    parser.add_argument('-destbucket', type=str, default='trainingset', help='Trainingset bucket')

    parser.add_argument('-description', type=str, default='Added exceptions')

    parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')

    args = parser.parse_args()
    return args

def feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpeg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        args.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            args.image_folder, filenames[i] + '.' + args.image_format)
        image_data = tf.compat.v1.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            args.semantic_segmentation_folder,
            filenames[i] + '.' + args.label_format)
        seg_data = tf.compat.v1.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def shuffle(seed, *lists):

    if seed is None:
        seed = random.random()
    for ls in lists:
        random.Random(seed).shuffle(ls)

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

    '''with tf.gfile.GFile(label_path, 'rb') as fid:
        encoded_label = fid.read()
    encoded_label_io = io.BytesIO(encoded_label)
    label = PIL.Image.open(encoded_label_io)
    if label.format != 'PNG':
        raise ValueError('Label format not PNG')'''
    
    try:
        image = tifffile.TiffFile(iman['im'])
        id = image.pages.pages[0].tags['FEI_HELIOS']
        mpp = [id.value['Scan']['PixelWidth'], id.value['Scan']['PixelHeight']]


    except:
        raise ValueError('Failed to read pixel size in {}.  Continuing with default {}'.format(iman['im'], mpp))

    label = cv2.imread(iman['an'],cv2.IMREAD_GRAYSCALE )

    if label is None:
        raise ValueError('{} annotation failed to load '.format(iman['an']))

    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    retval, buffer = cv2.imencode('.png', image.asarray(), compression_params)
    encoded_image = io.BytesIO(buffer).read()

    retval, buffer = cv2.imencode('.png', label, compression_params)
    encoded_label =io.BytesIO(buffer).read()
    
    if args.show:
        fig = plt.figure(figsize=(20,8))
        a = fig.add_subplot(1, 2, 1)
        plt.imshow(image.asarray(), cmap='gray', vmin=0, vmax=255)
        a.set_title(iman['im'])

        a = fig.add_subplot(1, 2, 2)
        plt.imshow(label)
        a.set_title(iman['an'])

        plt.show()

    height, width = image.asarray().shape
    heightL, widthL = label.shape
    if width!= widthL or height != heightL:
        raise ValueError('image {} size != annotation {} size '.format(iman['im'], iman['an']))

    feature = {
        'image/height': _int_feature([height]),
        'image/width': _int_feature([width]),
        'image/encoded': _bytes_feature(encoded_image),
        'image/format': _str_feature('png'),
        'label/encoded': _bytes_feature(encoded_label),
        'label/format': _str_feature('png'),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def Shuffle(seed, *lists):

    if seed is None:
        seed = random.random()
    for ls in lists:
        random.Random(seed).shuffle(ls)

def WriteRecords(args, iman):

    Shuffle(args.seed, iman)

    start = 0
    numEntries = len(iman)
    for ids, dataset in enumerate(args.sets):
        for shard_id in range(args.shards):
            output_filename = os.path.join(args.record_dir, '%s-%05d-of-%05d.tfrecord' % (dataset['name'], shard_id, args.shards))
            if(ids == len(args.sets)-1 and shard_id == args.shards-1):
                stop = numEntries
            else:
                groupSize = int(numEntries*dataset['ratio']/args.shards)
                stop = start+groupSize

            print('{} start {} stop {}'.format(output_filename, start, stop))
            with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i in tqdm(range(start,stop)):
                    example = Example(args, iman[i])
                    tfrecord_writer.write(example.SerializeToString())
            start = stop

            sys.stdout.write('\n')
            sys.stdout.flush()

    description = {'description':args.description, 'annotations':args.annotations, 'seed':args.seed}
    with open(args.record_dir+'/description.json', 'w') as fp:
        json.dump(description, fp, indent=4, separators=(',', ': '))

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

def main(args):
    
    iman = []
    s3 = s3store(args.minio_address, args.minio_access_key, args.minio_secret_key)
    shutil.rmtree(args.ann_dir, ignore_errors=True)
    for src in args.annotations:
        destDir = args.ann_dir+'/'+src
        s3.GetDir(args.srcbucket,src,destDir)
        file_list = glob.glob(glob.escape(destDir)+'/*.'+args.image_type)

        for imfile in file_list:
            annFile = '{}{}'.format(os.path.splitext(imfile)[0],args.annotation_decoration)
            if(os.path.exists(imfile) and os.path.exists(annFile)):
                iman.append({'im':imfile, 'an':annFile})
    if(len(iman)<1):
        print('No files found {}  Exiting'.format(args.annotations))
        return

    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    WriteRecords(args, iman)
    
    print('Write to s3  {}/{}'.format(args.destbucket,args.setname))

    s3.PutDir(args.destbucket, args.record_dir, args.setname)
    shutil.rmtree(args.ann_dir, ignore_errors=True)
    shutil.rmtree(args.record_dir, ignore_errors=True)
    print('{} complete'.format(os.path.basename(__file__)))

if __name__ == '__main__':
    args = parse_arguments()

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