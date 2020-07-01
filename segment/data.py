import os
import glob
import random
from datetime import datetime
import tensorflow as tf
import tensorflow_addons as tfa

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

def random_crop_or_pad_image_and_label(image, label, config):
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

    image_and_label = tf.concat([image, label], axis=2)
    image_and_label_pad = tf.image.pad_to_bounding_box(
        image_and_label, 0, 0,
        tf.maximum(config['size_Y'], image_height),
        tf.maximum(config['size_X'], image_width))
    image_and_label_crop = tf.image.random_crop(image_and_label, [config['size_Y'], config['size_X'], image_depth+1])

    image_crop = image_and_label_crop[:, :, :image_depth]
    label_crop = image_and_label_crop[:, :, image_depth:]

    return image_crop, label_crop

def augment_image(image, label, config):

    # Augment transformations
    rotate = tf.random.uniform(shape=[], minval=-config["augment_rotation"],maxval=config["augment_rotation"],name='rotation_augmentation')
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
    image = tfa.image.transform(image, M, interpolation='BILINEAR')
    label = tfa.image.transform(label, M, interpolation='NEAREST')    

    image, label = random_crop_or_pad_image_and_label(image, label, config) 

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

def input_fn(is_training, data_dir, config, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.

  Returns:
    A tuple of images and labels.
  """
  if is_training:
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(data_dir))
  else:
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(data_dir, 'validation'))

  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  if is_training:

    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=250)
    dataset = dataset.repeat(config['epochs'])
    dataset = dataset.map(lambda raw_record: parse_record(raw_record, config), num_parallel_calls = 10)
    dataset = dataset.map(lambda image, label: augment_image(image, label, config) , num_parallel_calls = 10)

  else:
    dataset = dataset.map(lambda raw_record: parse_record(raw_record, config), num_parallel_calls = 10)
    dataset = dataset.map(lambda image, label: random_crop_or_pad_image_and_label(image, label, config) , num_parallel_calls = 10)
  
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
