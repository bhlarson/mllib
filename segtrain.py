# Based on https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import cv2
import copy
from datetime import datetime
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from segdataset import input_fn


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('--dataset_dir', type=str, default='./dataset',help='Directory to store training model')
parser.add_argument('--saveonly', action='store_true', help='Do not train.  Only produce saved model')

parser.add_argument('--record_dir', type=str, default='record', help='Path training set tfrecord')
parser.add_argument('--model_dir', type=str, default='./trainings/fcn',help='Directory to store training model')

parser.add_argument('--clean_model_dir', type=bool, default=True,
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--epochs', type=int, default=5,
                    help='Number of training epochs')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=2,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--batch_size', type=int, default=8, help='Number of examples per batch.')
parser.add_argument('--crops', type=int, default=1, help='Crops/image/step')                

parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='Adam optimizer learning rate.')

parser.add_argument("--strategy", type=str, default='onedevice', help="Replication strategy. 'mirrored', 'onedevice' now supported ")
parser.add_argument("--devices", type=json.loads, default=["/gpu:0"],  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')

parser.add_argument('--training_crop', type=json.loads, default='[224, 224]', help='Training crop size [height, width]')
parser.add_argument('--train_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB') 

defaultfinalmodelname = '{}-dl3'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
parser.add_argument('--finalmodel', type=str, default=defaultfinalmodelname, help='Final model')

parser.add_argument('--savedmodel', type=str, default='./saved_model', help='Path to fcn savedmodel.')
defaultsavemodelname = '{}-dl3'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
parser.add_argument('--savedmodelname', type=str, default=defaultsavemodelname, help='Final model')
parser.add_argument('--tbport', type=int, default=6006, help='Tensorboard network port.')


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


trainingsetDescriptionFile = '{}/description.json'.format(FLAGS.record_dir)
trainingsetDescription = json.load(open(trainingsetDescriptionFile))

config = {
      'batch_size': FLAGS.batch_size,
      'trainingset': trainingsetDescription,
      'input_shape': [FLAGS.training_crop[1], FLAGS.training_crop[0], FLAGS.train_depth],
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
      'ignore_label': trainingsetDescription['classes']['ignore'],
      'classes': trainingsetDescription['classes']['classes'],
      'image_crops': FLAGS.crops,
      'epochs': FLAGS.epochs,
      'area_filter_min': 25,
      }

dataset, info = tfds.load('oxford_iiit_pet:3.2.0', data_dir=FLAGS.dataset_dir, with_info=True)
VAL_SUBSPLITS = 5
VALIDATION_STEPS = 100

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

def WriteDictJson(outdict, path):

    jsonStr = json.dumps(outdict, sort_keys=False)

    f = open(path,"w")
    f.write(jsonStr)
    f.close()
       
    return True

def FindObjectType(index, objectTypes):
    for objectType in objectTypes:
        if objectType['trainId'] == index:
            return objectType
    return None

def DefineFeature(c, iClass, minArea = 0, maxArea = float("inf"), iContour=1):
    M = cv2.moments(c)

    area = M['m00']
    # Ignore contours that are too small or too large
    
    if area < 1.0 or area < minArea or area > maxArea:
        return {}

    center = (M['m10']/M['m00'], M['m01']/M['m00'])

    rect = cv2.boundingRect(c)   

    feature = {'class':iClass, 'iContour':iContour, 'center':center,  'rect': rect, 'area':area, 'contour':c}
    
    return feature

def FilterContours(seg, objectType, config):
    

    [height, width] = seg.shape

    # Area filter
    if 'area_filter_min' in objectType:
        minArea  = objectType['area_filter_min']
    else:
        minArea = config['area_filter_min']
    
    if 'area_filter_max' in objectType:
        maxArea  = objectType['area_filter_max']
    else:
        maxArea = height*width

    iSeg = copy.deepcopy(seg)
    iSeg[iSeg != objectType['trainId']] = 0 # Convert to binary mask        
    
    
    segFeatures = []
    contours, hierarchy = cv2.findContours(iSeg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):

        feature = DefineFeature(c, objectType['trainId'], minArea, maxArea, i)
        if feature:
            segFeatures.append(feature)

    return segFeatures

def ExtractFeatures(seg, objTypes, config):
    segFeatures = []
    for segobj in objTypes:
      if segobj['display']:
        feature_contours = FilterContours(seg, segobj, config)
        segFeatures.extend(feature_contours)


    return segFeatures

def ColorToBGR(color):
    return (color[2], color[1], color[0])

def DrawFeatures(img, seg, objTypes, config):
    features = ExtractFeatures(seg, objTypes, config)

    for feature in features:
        obj = FindObjectType(feature['class'], objTypes)
        if obj and obj['display']:
            cv2.drawContours(img, [feature['contour']], 0, ColorToBGR(obj['color']), thickness=3)

def DrawSeg(img, ann, pred, objTypes, config):
    ann = tf.squeeze(ann)
    pred = tf.squeeze(pred)

    img = img.numpy()
    ann = ann.numpy()
    pred = pred.numpy()

    img = img.astype(np.uint8)
    ann = ann.astype(np.uint8)
    pred = pred.astype(np.uint8)

    annImg = copy.deepcopy(img)
    DrawFeatures(annImg, ann, objTypes, config)

    predImg = copy.deepcopy(img)
    DrawFeatures(predImg, pred, objTypes, config)

    return annImg, predImg

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask



def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


# The dataset already contains the required splits of test and train and so let's continue to use the same split.



TRAIN_LENGTH = info.splits['train'].num_examples
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // config['batch_size']

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(config['batch_size']).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(config['batch_size'])


# Let's take a look at an image example and it's correponding mask from the dataset.



def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()



for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.

  Conv2DTranspose => Batchnorm => Dropout => Relu

  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer

  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

# ## Define the model
# The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features, and reduce the number of trainable parameters, a pretrained model can be used as the encoder. Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate outputs will be used, and the decoder will be the upsample block already implemented in TensorFlow Examples in the [Pix2pix tutorial](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py). 
# 
# The reason to output three channels is because there are three possible labels for each pixel. Think of this as multi-classification where each pixel is being classified into three classes.

# As mentioned, the encoder will be a pretrained MobileNetV2 model which is prepared and ready to use in [tf.keras.applications](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/applications). The encoder consists of specific outputs from intermediate layers in the model. Note that the encoder will not be trained during the training process.
def unet_model(classes, input_shape):
  base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

  # Use the activations of these layers
  layer_names = [
      'block_1_expand_relu',   # 64x64
      'block_3_expand_relu',   # 32x32
      'block_6_expand_relu',   # 16x16
      'block_13_expand_relu',  # 8x8
      'block_16_project',      # 4x4
  ]
  layers = [base_model.get_layer(name).output for name in layer_names]

  # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

  down_stack.trainable = False


  # The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples.

  up_stack = [
      upsample(512, 3),  # 4x4 -> 8x8
      upsample(256, 3),  # 8x8 -> 16x16
      upsample(128, 3),  # 16x16 -> 32x32
      upsample(64, 3),   # 32x32 -> 64x64
  ]

  inputs = tf.keras.layers.Input(shape=input_shape)
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      classes, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


# ## Train the model
# Now, all that is left to do is to compile and train the model. The loss being used here is `losses.SparseCategoricalCrossentropy(from_logits=True)`. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and `losses.SparseCategoricalCrossentropy(from_logits=True)` is the recommended loss for 
# such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.

classes = 3
input_shape=[128, 128, 3]

classes = config['classes']
input_shape = config['input_shape']

model = unet_model(classes, input_shape)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

try:
  model.load_weights(FLAGS.savedmodel)
except:
  print('Unable to load weghts from {}'.format(FLAGS.savedmodel))

# Display model
tf.keras.utils.plot_model(model, to_file='unet.png', show_shapes=True)
model.summary()


# Let's try out the model to see what it predicts before training.
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask

def WritePredictions(dataset, model, config, num=1, outpath=''):
    batch_size = config['batch_size']
    objTypes = config['trainingset']['classes']['objects']
    i = 0
    for image, mask in dataset.take(num):
      for j in range(batch_size):
        pred_mask = create_mask(model.predict(image))

        ann, pred = DrawSeg(image[j], mask[j], pred_mask[j], objTypes, config)
        resultImg = np.concatenate((ann, pred), axis=1)
        cv2.imwrite('{}ann-pred{}.png'.format(outpath, i*batch_size+j), resultImg)

      i=i+1

train_dataset = input_fn(True, FLAGS.record_dir, config)
test_dataset = input_fn(False, FLAGS.record_dir, config)

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

# Ssave callback the model's weights
save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.savedmodel,
                                                 save_weights_only=True,
                                                 verbose=1)
outpath = '{}/{}/'.format(FLAGS.savedmodel, FLAGS.savedmodelname)

if not os.path.exists(outpath):
    os.makedirs(outpath)

if not FLAGS.saveonly:
  model_history = model.fit(train_dataset, epochs=config['epochs'],
                            steps_per_epoch=int(trainingsetDescription['sets'][0]['length']/config['batch_size']),
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_dataset,
                            callbacks=[save_callback])

  history = model_history.history
  if 'loss' in history:
    loss = model_history.history['loss']
  else:
    loss = []
  if 'val_loss' in history:
    val_loss = model_history.history['val_loss']
  else:
    val_loss = []

  model_description = {'config':config,
                       'results': history
                      }
  epochs = range(config['epochs'])

  plt.figure()
  plt.plot(epochs, loss, 'r', label='Training loss')
  plt.plot(epochs, val_loss, 'bo', label='Validation loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss Value')
  plt.ylim([0, 1])
  plt.legend()
  plt.savefig('{}training.svg'.format(outpath))

else:
  model_description = {'config':config,
                      }

WriteDictJson(model_description, '{}descrption.json'.format(outpath))

model.save(outpath)

# Let's make some predictions. In the interest of saving time, the number of epochs was kept small, but you may set this higher to achieve more accurate results.
WritePredictions(test_dataset, model, config, outpath=outpath)
print("Segmentation training complete. Results saved to {}".format(outpath))
