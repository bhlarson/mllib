"""Trein segmentation model"""

import argparse
import os
import sys
import shutil
import glob
import random
import json
from datetime import datetime
import natsort as ns
import tensorflow as tf
from tensorboard import program
import networks.deeplab
from segment.data import (input_fn, serving_input_receiver_fn)
from tensorflow.python import debug as tf_debug
#from networks.deeplab import DeepLabV3Plus
from networks.deeplabv3 import Deeplabv3
from tensorflow.keras.optimizers import Adam
from networks.fcn import FCN

print('Python Version {}'.format(sys.version))
print('Tensorflow version {}'.format(tf.__version__))
print('GPU Available: {}'.format(tf.test.is_gpu_available()))
if(tf.test.is_gpu_available()):
  print('GPU Devices: {}'.format(tf.test.gpu_device_name()))

parser = argparse.ArgumentParser()

parser.add_argument('--record_dir', type=str, default='record', help='Path training set tfrecord')
parser.add_argument('--model_dir', type=str, default='./trainings/fcn',help='Directory to store training model')

parser.add_argument('--clean_model_dir', type=bool, default=True,
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--train_epochs', type=int, default=1,
                    help='Number of training epochs')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=2,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--batch_size', type=int, default=2, help='Number of examples per batch.')
parser.add_argument('--crops', type=int, default=1, help='Crops/image/step')                

parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='Adam optimizer learning rate.')

parser.add_argument("--strategy", type=str, default='onedevice', help="Replication strategy. 'mirrored', 'onedevice' now supported ")
parser.add_argument("--devices", type=json.loads, default=["/gpu:0"],  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')

parser.add_argument('--classes', type=json.loads, default='{}', help='Class dictionary JSON.  Leave empty if classes_file points to a JSON file.')
parser.add_argument('--classes_file', type=int, default=16, help='Class dictionary JSON file')
parser.add_argument('--training_crop', type=json.loads, default='[256, 512]', help='Training crop size [height, width]')
parser.add_argument('--train_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB') 

defaultfinalmodelname = '{}-dl3'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
parser.add_argument('--finalmodel', type=str, default=defaultfinalmodelname, help='Final model')

parser.add_argument('--savedmodel', type=str, default='./saved_model', help='Path to fcn savedmodel.')
defaultsavemodelname = '{}-dl3'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
parser.add_argument('--savedmodelname', type=str, default=defaultsavemodelname, help='Final model')
parser.add_argument('--tbport', type=int, default=6006, help='Tensorboard network port.')
parser.add_argument('--saveonly', type=bool, default=False, help='True, enable debug and stop at breakpoint')
parser.add_argument('--debug', action='store_true',help='Wait for debuger attach')


def get_filenames(data_dir, dataset='train'):
  """Return a list of filenames.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.

  Returns:
    A list of file names.
  """
  if dataset=='train':
    return glob.glob(os.path.join(data_dir, 'train-?????-of-?????.tfrecord'))
  elif dataset=='validation':
    return glob.glob(os.path.join(data_dir, 'val-?????-of-?????.tfrecord'))
  elif dataset=='test':
    return glob.glob(os.path.join(data_dir, 'test-?????-of-?????.tfrecord'))
  return []


def parse_record(raw_record, config):
  """Parse PASCAL image and label from a tf record."""
  keys_to_features = {
      'image/height': tf.io.FixedLenFeature((), tf.int64),
      'image/width': tf.io.FixedLenFeature((), tf.int64),
      'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.io.FixedLenFeature((), tf.string, default_value='png'),
      'label/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
      'label/format': tf.io.FixedLenFeature((), tf.string, default_value='png'),
  }

  parsed = tf.io.parse_single_example(serialized=raw_record, features=keys_to_features)

  image = tf.image.decode_image(tf.reshape(parsed['image/encoded'], shape=[]), config['depth'])
  image = tf.cast(tf.image.convert_image_dtype(image, dtype=tf.uint8), dtype=tf.float32)
  image.set_shape([None, None, config['depth']])

  label = tf.image.decode_image(tf.reshape(parsed['label/encoded'], shape=[]), 1)
  label = tf.cast(tf.image.convert_image_dtype(label, dtype=tf.uint8), dtype=tf.float32)
  label.set_shape([None, None, 1])

  return image, label



def random_crop_or_pad_image_and_label(image, label, crop_width, crop_height):
  """Crops and/or pads an image to a target width and height.

  Resizes an image to a target width and height by rondomly
  cropping the image or padding it evenly with zeros.

  Args:
    image: 3-D Tensor of shape `[height, width, channels]`.
    label: 3-D Tensor of shape `[height, width, 1]`.
    crop_height: The new height.
    crop_width: The new width.

  Returns:
    Cropped and/or padded image.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """

  image_height = tf.shape(input=image)[0]
  image_width = tf.shape(input=image)[1]
  image_depth = tf.shape(input=image)[2]
  label_depth = tf.shape(input=label)[2]

  image_and_label = tf.concat([image, label], axis=2)
  image_and_label_pad = tf.image.pad_to_bounding_box(
      image_and_label, 0, 0,
      tf.maximum(crop_height, image_height),
      tf.maximum(crop_width, image_width))
  image_and_label_crop = tf.image.random_crop(image_and_label_pad, [crop_height, crop_width, image_depth+label_depth])

  image_crop = image_and_label_crop[:, :, :image_depth]
  label_crop = image_and_label_crop[:, :, image_depth:]

  return image_crop, label_crop

def augment_image(image, label, config):

    # Augment transformations
    '''rotate = tf.random.uniform(shape=[], minval=-config["augment_rotation"],maxval=config["augment_rotation"],name='rotation_augmentation')
    rotate =  tf.math.multiply(rotate, 0.0174532925199) # Multiply by pi/180 to convert degrees to radians

    scale = tf.random.uniform(shape=[], minval=config["scale_min"],maxval=config["scale_max"],
      dtype=tf.dtypes.float32,name='scale_augmentation')

    shift_x = tf.random.uniform(shape=[], minval=-config['size_X']*config['augment_shift_x'],maxval=config['size_X']*config['augment_shift_x'],
      dtype=tf.dtypes.float32,name='scale_augmentation')
  
    shift_y = tf.random.uniform(shape=[], minval=-config['size_Y']*config['augment_shift_y'],maxval=config['size_Y']*config['augment_shift_y'],
      dtype=tf.dtypes.float32,name='scale_augmentation')

    # Modeled on OpenCV 2D Transformations: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    alpha = tf.math.multiply(scale, tf.math.cos(rotate))
    beta = tf.math.multiply(scale, tf.math.sin(rotate))

    m02 = tf.multiply(tf.math.subtract(1.0, alpha), config['size_X']/2.0)
    m02 = tf.math.subtract(m02, tf.multiply(beta,config['size_Y']/2.0))
    m02 = tf.math.add(m02,shift_x)

    m12 = tf.multiply(tf.math.subtract(1.0, alpha),config['size_Y']/2.0)
    m12 = tf.math.add( tf.multiply(beta, config['size_X']/2.0),m12)
    m12 = tf.math.add(m12,shift_y)

    M = [alpha, beta, m02 ,tf.math.negative(beta), alpha, m12,0,0]

    # Transform before cropping so full data is available
    image = tf.cast(tf.contrib.image.transform(image,M,interpolation='BILINEAR'), tf.uint8)
    label = tf.cast(tf.contrib.image.transform(label,M,interpolation='NEAREST'), tf.uint8)
    #label = tf.contrib.image.transform(label,M,interpolation='NEAREST')'''

    '''scalex = tf.random.uniform(shape=[], minval=config["scale_min"],maxval=config["scale_max"])
    scaley = tf.random.uniform(shape=[], minval=config["scale_min"],maxval=config["scale_max"])
    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)

    sizex = tf.cast(scalex*image_width, tf.int32)
    sizey = tf.cast(scaley*image_height, tf.int32)
    
    image = tf.image.resize(image, size=(sizey, sizex), method='area')
    label = tf.image.resize(label, size=(sizey, sizex), method='nearest')'''

    #image = tf.cast(image, tf.uint8)
    #label = tf.cast(label, tf.uint8)

    image, label = random_crop_or_pad_image_and_label(image, label, config['size_X'], config['size_Y']) 

    # Flip cropped images to reduce computation
    if(config['augment_flip_x']):
      flip = tf.math.greater_equal(tf.random.uniform(shape=[], maxval=1.0),0.5)
      image = tf.cond(pred=flip, true_fn=lambda: tf.image.flip_up_down(image), false_fn=lambda: image)
      label = tf.cond(pred=flip, true_fn=lambda: tf.image.flip_up_down(label), false_fn=lambda: label)

    if(config['augment_flip_y']):
      flip = tf.math.greater_equal(tf.random.uniform(shape=[], maxval=1.0),0.5)
      image = tf.cond(pred=flip, true_fn=lambda: tf.image.flip_left_right(image), false_fn=lambda: image)
      label = tf.cond(pred=flip, true_fn=lambda: tf.image.flip_left_right(label), false_fn=lambda: label)
    
    return image, label

def augment_image_crops(image, label, config):

    for i in range(config['image_crops']):
        image, label = augment_image(image, label, config)

    return image, label

def prepare_image(image, label, config):
    image = tf.cast(tf.image.resize_with_crop_or_pad(image, config['size_Y'], config['size_X']), dtype=tf.float32)
    label = tf.cast(tf.image.resize_with_crop_or_pad(label, config['size_Y'], config['size_X']), dtype=tf.float32)

    return image, label

def input_fn(is_training, data_dir, config, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """

  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(data_dir))
  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  if is_training:

    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=250)
    dataset = dataset.repeat(num_epochs)

  dataset = dataset.map(lambda raw_record: parse_record(raw_record, config), num_parallel_calls = 10)
  dataset = dataset.map(lambda image, label: prepare_image(image, label, config) , num_parallel_calls = 10)
  
  #if is_training:
  #  for i in range(config['image_crops']):
  #    dataset = dataset.map(lambda image, label: augment_image(image, label, config))
    #dataset = dataset.batch(config['image_crops'])
  #else:
  #  dataset = dataset.map(lambda image, label: prepare_image(image, label, config) , num_parallel_calls = 10)


  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.batch(config['batch_size'])
  dataset = dataset.prefetch(config['batch_size'])

  return dataset

def serving_input_receiver_fn(config):
    shape = [None, None, config['depth']]
    image = tf.placeholder(dtype=tf.uint8, shape=shape, name='image')
    images = tf.expand_dims(image, 0)
    return tf.estimator.export.TensorServingInputReceiver(images, image)

def main(FLAGS):
  #tf.compat.v1.disable_eager_execution()
  tf.config.experimental_run_functions_eagerly(False)
  trainingsetDescriptionFile = '{}/description.json'.format(FLAGS.record_dir)
  trainingsetDescription = json.load(open(trainingsetDescriptionFile))

  config = {
      'batch_size': FLAGS.batch_size,
      'trainingset': trainingsetDescription,
      'size_X': FLAGS.training_crop[1], 'size_Y': FLAGS.training_crop[0], 'depth':FLAGS.train_depth,
      'classScale': 0.001, # scale value for each product class
      'augment_rotation' : 5., 
      'augment_flip_x': False,
      'augment_flip_y': True,
      'augment_brightness':0.,
      'augment_contrast': 0.,
      'augment_shift_x': 0.1,
      'augment_shift_y': 0.1,
      'scale_min': 0.75,
      'scale_max': 1.25,
      'seed':None,
      'ignore_label': trainingsetDescription['classes']['ignore'],
      'image_crops': FLAGS.crops,
      'images_max_outputs': FLAGS.tensorboard_images_max_outputs,
      }

  if(FLAGS.strategy == 'mirrored'):
    strategy = tf.distribute.MirroredStrategy(devices=FLAGS.devices)
    num_batches = strategy.num_replicas_in_sync
    device = FLAGS.devices
  else:
    num_batches = 1
    FLAGS.strategy = 'onedevice'
    if(FLAGS.devices is not None and len(FLAGS.devices) > 0):
      device = FLAGS.devices[0]
    else:
      device = None
    strategy = tf.distribute.OneDeviceStrategy(device=device)

  print('{} strategy with {} GPU(s) {}'.format(FLAGS.strategy,strategy.num_replicas_in_sync, device))
  config['batch_size'] = num_batches*config['batch_size']

  if FLAGS.clean_model_dir:
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

  with strategy.scope():
    #model = DeepLabV3Plus(img_width=config['size_X'], img_height=config['size_Y'],  nclasses=config['trainingset']['classes']['classes'])
    #model = Deeplabv3(weights=None, input_shape=(config['size_Y'], config['size_X'], config['depth']), classes=config['trainingset']['classes']['classes'])
    model = FCN(input_shape=(config['size_Y'], config['size_X'], config['depth']), classes=config['trainingset']['classes']['classes'], weights=None)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        optimizer=Adam(lr=FLAGS.learning_rate), 
        metrics=['accuracy'])
    model.summary()

  #model.load_weights(tf.train.latest_checkpoint(FLAGS.model_dir))

  if(FLAGS.saveonly != True):
    # Launch tensorboard for training
    # Remove http messages
    #tb = program.TensorBoard()
    #tb.configure(argv=[None, '--logdir', FLAGS.model_dir, '--port', str(FLAGS.tbport)])
    #url = tb.launch()
    #print('TensorBoard at {}'.format(url))

    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.model_dir, update_freq='batch', write_graph=True, write_images=True, histogram_freq=1)

    callbacks = [
        #tf.keras.callbacks.TensorBoard(log_dir='FLAGS.model_dir'),
        #tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.model_dir, save_weights_only=True)
    ]

    #for _ in range(FLAGS.train_epochs):
    #dataset = tf.data.Dataset.from_tensor_slices(get_filenames(FLAGS.record_dir))
    #dataset = dataset.flat_map(tf.data.TFRecordDataset)
    #dataset = dataset.shuffle(buffer_size=5*config['batch_size'])
    #dataset = dataset.repeat(FLAGS.train_epochs)

    #dataset = dataset.map(lambda raw_record: parse_record(raw_record, config), num_parallel_calls = 10)
    #for i in range(config['image_crops']):
    #  dataset = dataset.map(lambda image, label: augment_image(image, label, config))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    #dataset = dataset.batch(config['batch_size'])
    #dataset = dataset.prefetch(config['batch_size'])
    dataset = input_fn(True, FLAGS.record_dir, config)

    model.fit(dataset, epochs=FLAGS.train_epochs, callbacks=callbacks)

  model.save(FLAGS.savedmodel)
  #savedmodel = model.export_saved_model(FLAGS.savedmodel, serving_input_receiver_fn=lambda:serving_input_receiver_fn(config), experimental_mode=tf.estimator.ModeKeys.PREDICT, as_text=True)

  print('{} saved'.format(savedmodel.decode('utf-8')))

if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  
  if FLAGS.debug:
      print("Wait for debugger attach")
      import ptvsd
      # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
      # Launch applicaiton on remote computer: 
      # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait fcn/train.py
      # Allow other computers to attach to ptvsd at this IP address and port.
      ptvsd.enable_attach(address=('0.0.0.0', 3000), redirect_output=True)
      # Pause the program until a remote debugger is attached

      ptvsd.wait_for_attach()

      print("Debugger attach")

  main(FLAGS)
