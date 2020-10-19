# Model based on https://www.tensorflow.org/tutorials/images/segmentation

import tensorflow as tf
DEBUG = False

class ImageStandardization(tf.keras.layers.Layer):

  def __init__(self):
    super(ImageStandardization, self).__init__()

  #@tf.function
  @tf.function(autograph=not DEBUG)
  def call(self, image):
    if DEBUG:
      tf.print("ImageStandardization.call initial image", tf.shape(image), image.dtype)

    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)

    if DEBUG:
      tf.print("ImageStandardization final image", tf.shape(image), image.dtype)
    return image

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

  #@tf.function
  @tf.function(autograph=not DEBUG)
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

@tf.function(autograph=not DEBUG)
def unet_loss(labels,logits):
    if DEBUG:
      tf.print('unet_loss logits ', logits.shape, logits.dtype, 'labels', labels.shape, labels.dtype)

    labels = tf.squeeze(tf.cast(labels, tf.int32))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='Crossentropy')
    loss = tf.reduce_mean(input_tensor=(loss))
    if DEBUG:
      tf.print('unet_loss loss:', loss, loss.shape, loss.dtype)
    return loss


def unet_compile(model, learning_rate=0.0001):
  #loss = {'format2_channels_last':unet_loss}
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizers = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizers, loss=loss, loss_weights=[1.0], metrics=['accuracy'], run_eagerly=DEBUG)
  return model

# ## Define the model
# The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). 
# In-order to learn robust features, and reduce the number of trainable parameters, a pretrained model can be used as 
# the encoder. Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate outputs will 
# be used, and the decoder will be the upsample block already implemented in TensorFlow Examples in the 
# [Pix2pix tutorial](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py). 
#
# The encoder, a pretrained MobileNetV2 model, is prepared and ready to use in 
# [tf.keras.applications](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/applications). The encoder 
# consists of specific outputs from intermediate layers in the model. Note that the encoder will not be trained during 
# the training process.
def unet_model(classes, input_shape, learning_rate=0.0001, weights='imagenet', channel_order='channels_last', train_base_model = True):
  
  tf.keras.backend.set_image_data_format(channel_order)
  
  # The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples.
  feature_channel = 1 if channel_order == 'channels_first' else len(input_shape)

  #if channel_order == 'channels_first':
  #  feature_channel = 1
  #else:
  #  feature_channel = 3

  base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=weights)
  print('Display MobileNetV2 encoder')
  base_model.summary()
  model_input = base_model.get_layer('Conv1').input # Remove Conv1_pad to facilitate resizing

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

  down_stack.trainable = train_base_model

  up_stack = [
      upsample(512, feature_channel),  # 4x4 -> 8x8
      upsample(256, feature_channel),  # 8x8 -> 16x16
      upsample(128, feature_channel),  # 16x16 -> 32x32
      upsample(64, feature_channel),   # 32x32 -> 64x64
  ]

  inputs = tf.keras.layers.Input(shape=input_shape)

  x = inputs

  if channel_order == 'channels_first':
    NCHW = tf.keras.layers.Permute((3, 1, 2))
    x = NCHW(x)

  #x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
  standardization = ImageStandardization()
  x = standardization(x)

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate(axis=feature_channel)
    x = concat([x, skip])

  # This is the last layer of the model
  Conv2DTranspose = tf.keras.layers.Conv2DTranspose(classes, 3, strides=2,padding='same', name='Conv2DTranspose')
  x = Conv2DTranspose(x)

  if channel_order == 'channels_first':
    NHWC = tf.keras.layers.Permute((2, 3, 1))
    x = NHWC(x)

  model = tf.keras.Model(inputs=inputs, outputs=[x], name='UnetSegmentation')


  # How do I output both logits and argmax operation is computed by the GPU?
  #seg = tf.argmax(logits, axis=-1, name='segmentation')
  #model = tf.keras.Model(inputs=inputs, outputs=[x, seg], name='UnetSegmentation')


  return model