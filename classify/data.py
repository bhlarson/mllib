import os
import glob
import random
from datetime import datetime
import tensorflow as tf
import tensorflow_addons as tfa

DEBUG = False

def get_filenames(data_dir, dataset='train'):
  """Return a list of filenames.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.

  Returns:
    A list of file names.
  """
  files = []
  if dataset=='train':
    files = glob.glob(os.path.join(data_dir, 'train-?????-of-?????.tfrecord'))
  elif dataset=='validation' or dataset=='val':
    files =  glob.glob(os.path.join(data_dir, 'val-?????-of-?????.tfrecord'))
  elif dataset=='test':
    files =  glob.glob(os.path.join(data_dir, 'test-?????-of-?????.tfrecord'))
  else:
    files =  glob.glob(os.path.join(data_dir, '{}-?????-of-?????.tfrecord'.format(dataset)))
  return files

def parse_record(raw_record, input_shape, channel_order, classes):
  """Parse PASCAL image and label from a tf record."""
  keys_to_features = {
      'image/height': tf.io.FixedLenFeature((), tf.int64),
      'image/width': tf.io.FixedLenFeature((), tf.int64),
      'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.io.FixedLenFeature((), tf.string, default_value='png'),
      'label/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
      'label/format': tf.io.FixedLenFeature((), tf.string, default_value='png'),
      'label/classes': tf.io.FixedLenFeature([classes], tf.float32),
  }

  parsed = tf.io.parse_single_example(serialized=raw_record, features=keys_to_features)

  image = tf.image.decode_image(tf.reshape(parsed['image/encoded'], shape=[]), input_shape[2])
  image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
  image.set_shape([None, None, input_shape[2]])
  #label = tf.reshape(parsed['label/classes'], [-1, classes])
  label = parsed['label/classes']
  #label = tf.image.decode_image(tf.reshape(parsed['label/encoded'], shape=[]), 1)
  #label = tf.image.convert_image_dtype(label, dtype=tf.uint8)

  if DEBUG:
    tf.print("parse_record: image", tf.shape(image), image.dtype, 'label shape', tf.shape(label), 'label', label )

  return image, label


def prepare_image(image, label, input_shape):
    image = tf.image.resize_with_crop_or_pad(image, input_shape[0], input_shape[1])
    #label = tf.image.resize_with_crop_or_pad(label, input_shape[0], input_shape[1])

    return image, label

def crop_or_pad_image(image, label, input_shape):
    """resize_with_crop_or_pad image to a target width and height.

    Resizes an image to a target width and height by rondomly
    cropping the image or padding it evenly with zeros.

    Args:
      image: 3-D Tensor of shape `[height, width, channels]`.
      label: 1-D logits tensor.
      crop_height: The new height.
      crop_width: The new width.

    Returns:
      Cropped and/or padded image.
      If `images` was 3-D, a 3-D float Tensor of shape
      `[new_height, new_width, channels]`.
    """

    crop_height = input_shape[0]
    crop_width = input_shape[1]
   
    image = tf.image.resize_with_crop_or_pad(image, crop_height, crop_width )

    if DEBUG:
      tf.print("crop_or_pad_image final image", tf.shape(image), image.dtype, 'label', tf.shape(label), label)

    return image, label

def random_crop_or_pad_image_and_label(image, label, input_shape):
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

    #image_and_label = tf.concat([image, label], axis=2)
    crop_height = tf.maximum(input_shape[0], image_height)
    crop_width = tf.maximum(input_shape[1], image_width)

    if DEBUG:
      tf.print("random_crop_or_pad_image_and_label initial image", tf.shape(image), image.dtype, 'label', tf.shape(label), label.dtype)
    
    image = tf.image.resize_with_crop_or_pad(image, crop_height, crop_width )
    #label = tf.image.resize_with_crop_or_pad(label, crop_height, crop_width )

    xExtra = crop_width-input_shape[1]
    yExtra = crop_height-input_shape[0]
    if(xExtra > 0):
      xBegin = tf.random.uniform(shape=[], maxval=xExtra, dtype=tf.int32)
    else:
      xBegin = 0
    if(yExtra > 0):
      yBegin = tf.random.uniform(shape=[], maxval=yExtra, dtype=tf.int32)
    else:
      yBegin = 0

    beginImage = [yBegin, xBegin, 0]
    sizeImage = [input_shape[0], input_shape[1],image_depth]
    beginLabel = [yBegin, xBegin,0]
    sizeLabel = [input_shape[0], input_shape[1], 1]

    if DEBUG:
      tf.print("random_crop_or_pad_image_and_label after resize_with_crop_or_pad image", tf.shape(image), image.dtype, 'label', tf.shape(label), label.dtype)

    image = tf.slice(image, beginImage, sizeImage)
    label = tf.slice(label, beginLabel, sizeLabel)

    #image_and_label = tf.image.random_crop(image_and_label, [input_shape[0], input_shape[1], image_depth+1])

    if DEBUG:
      tf.print("random_crop_or_pad_image_and_label shape after slice image", tf.shape(image), image.dtype, 'label', tf.shape(label), label.dtype)
    #image = image_and_label[:, :, :image_depth]
    #label = image_and_label[:, :, image_depth:]

    return image, label

def print_data(image, label, msg):
    # print("{} image {} {} label {} {}".format(msg.numpy().decode('UTF-8'), image.shape, image.dtype, label.shape, label.dtype))
    return image, label

def augment_image_config(image, label, config):
    return augment_image(image = image, label = label, 
      input_shape = config["input_shape"], 
      augment_rotation = config["augment_rotation"], 
      scale_min = config["scale_min"], 
      scale_max = config["scale_max"], 
      augment_shift_x = config["augment_shift_x"], 
      augment_shift_y = config["augment_shift_y"],
      augment_flip_x = config["augment_flip_x"],
      augment_flip_y = config["augment_flip_y"])

def augment_image(image, label, input_shape, augment_rotation, scale_min, scale_max, augment_shift_x, augment_shift_y, augment_flip_x, augment_flip_y):

    if DEBUG:
      tf.print('augment_image image input', tf.shape(image), image.dtype, 'label', tf.shape(label), label.dtype)
    shape =  tf.cast(input_shape, tf.float32)
    # Augment transformations
    rotate = tf.random.uniform(shape=[], minval=-augment_rotation,maxval=augment_rotation,name='rotation_augmentation')
    rotate =  tf.math.multiply(rotate, 0.0174532925199) # Scale from degrees to radians - multiply by pi/180

    scale = tf.random.uniform(shape=[], minval=scale_min,maxval=scale_max,name='scale_augmentation')

    maxShiftX = shape[1]*augment_shift_x # Normalize by image width
    shift_x = tf.random.uniform(shape=[], minval=-maxShiftX,maxval=maxShiftX, name='shift_x_augmentation')
  
    maxShiftY = shape[0]*augment_shift_y # Normalize by image height
    shift_y = tf.random.uniform(shape=[], minval=-maxShiftY,maxval=maxShiftY, name='shift_y_augmentation')

    # Modeled on OpenCV 2D Transformations: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    alpha = tf.math.multiply(scale, tf.math.cos(rotate))
    beta = tf.math.multiply(scale, tf.math.sin(rotate))

    m02 = tf.multiply(tf.math.subtract(1.0, alpha), shape[1]/2.0)
    m02 = tf.math.subtract(m02, tf.multiply(beta,shape[0]/2.0))
    m02 = tf.math.add(m02,shift_x)

    m12 = tf.multiply(tf.math.subtract(1.0, alpha),shape[0]/2.0)
    m12 = tf.math.add( tf.multiply(beta, shape[1]/2.0),m12)
    m12 = tf.math.add(m12,shift_y)

    M = [alpha, beta, m02 ,tf.math.negative(beta), alpha, m12,0,0]

    # Transform before cropping so full data is available
    image = tfa.image.transform(image, M, interpolation='BILINEAR')
    #label = tfa.image.transform(label, M, interpolation='NEAREST')    

    # Flip cropped images to reduce computation
    if(augment_flip_x):
        flip = tf.math.greater_equal(tf.random.uniform(shape=[], maxval=1.0),0.5)
        image = tf.cond(pred=flip, true_fn=lambda: tf.image.flip_up_down(image), false_fn=lambda: image)
        #label = tf.cond(pred=flip, true_fn=lambda: tf.image.flip_up_down(label), false_fn=lambda: label)

    if(augment_flip_y):
        flip = tf.math.greater_equal(tf.random.uniform(shape=[], maxval=1.0),0.5)
        image = tf.cond(pred=flip, true_fn=lambda: tf.image.flip_left_right(image), false_fn=lambda: image)
        #label = tf.cond(pred=flip, true_fn=lambda: tf.image.flip_left_right(label), false_fn=lambda: label)

    if DEBUG:
      tf.print("augment_image final", tf.shape(image), image.dtype, 'label', tf.shape(label), label.dtype)
    
    return image, label

def input_fn(datasetname, data_dir, config, num_epochs=1, shuffle_buffer=1, num_parallel_calls=1):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.

    Returns:
      A tuple of images and labels.
    """
    dataset = tf.data.TFRecordDataset(get_filenames(data_dir, dataset=datasetname))

    #dataset = dataset.flat_map(tf.data.TFRecordDataset)

    if datasetname=='train':
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        # is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        dataset = dataset.repeat(config['epochs'])

    dataset = dataset.map(lambda raw_record: parse_record(raw_record, config['input_shape'], config['channel_order'], config['classes']), num_parallel_calls = num_parallel_calls)

    if datasetname=='train':
        dataset = dataset.map(lambda image, label: augment_image_config(image, label, config) , num_parallel_calls = num_parallel_calls)
    
    dataset = dataset.map(lambda image, label: crop_or_pad_image(image, label, config['input_shape']) , num_parallel_calls = num_parallel_calls)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.prefetch(config['batch_size'])

    return dataset
