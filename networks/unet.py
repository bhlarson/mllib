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

    if tf.keras.backend.image_data_format() == 'channels_first':
      image = tf.transpose(a=image, perm=[0, 3, 1, 2])

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
    if tf.keras.backend.image_data_format() == 'channels_first':
      logits = tf.transpose(a=logits, perm=[0, 2, 3, 1])

    if DEBUG:
      tf.print('unet_loss logits ', logits.shape, logits.dtype, 'labels', labels.shape, labels.dtype)

    labels = tf.squeeze(tf.cast(labels, tf.int32))
    #tf.print('unet_loss 2 logits ', logits.shape, logits.dtype, 'labels', labels.shape, labels.dtype)
    #loss = tf.keras.losses.CategoricalCrossentropy(labels,logits, from_logits=True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='Crossentropy')
    loss = tf.reduce_mean(input_tensor=(loss))
    if DEBUG:
      tf.print('unet_loss loss:', loss, loss.shape, loss.dtype)
    return loss

@tf.function(autograph=not DEBUG)
def constant_loss(y_true,y_pred):
    loss = tf.constant(0.0, shape=[], dtype=tf.float32)
    #tf.print('constant_loss loss:', loss, loss.shape, loss.dtype)
    return loss

def unet_compile(model, learning_rate=0.0001):
  loss = {'logits':unet_loss}
  #loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(optimizer=adam, loss=loss, loss_weights=[1.0], metrics=[], run_eagerly=DEBUG)

  return model

# ## Define the model
# The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features, and reduce the number of trainable parameters, a pretrained model can be used as the encoder. Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate outputs will be used, and the decoder will be the upsample block already implemented in TensorFlow Examples in the [Pix2pix tutorial](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py). 
# 
# The reason to output three channels is because there are three possible labels for each pixel. Think of this as multi-classification where each pixel is being classified into three classes.

# As mentioned, the encoder will be a pretrained MobileNetV2 model which is prepared and ready to use in [tf.keras.applications](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/applications). The encoder consists of specific outputs from intermediate layers in the model. Note that the encoder will not be trained during the training process.
def unet_model(classes, input_shape, learning_rate=0.0001, weights='imagenet', chanel_order='channels_last'):
  
  tf.keras.backend.set_image_data_format(chanel_order)
  
  # The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples.
  if tf.keras.backend.image_data_format() == 'channels_first':
    feature_chanel = 1
    #input_shape = [input_shape[2], input_shape[0], input_shape[1]]
  else:
    feature_chanel = 3


  base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=weights)
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

  down_stack.trainable = False


  up_stack = [
      upsample(512, feature_chanel),  # 4x4 -> 8x8
      upsample(256, feature_chanel),  # 8x8 -> 16x16
      upsample(128, feature_chanel),  # 16x16 -> 32x32
      upsample(64, feature_chanel),   # 32x32 -> 64x64
      #upsample(layers[3].shape[1], feature_chanel),  # 4x4 -> 8x8
      #upsample(layers[2].shape[1], feature_chanel),  # 8x8 -> 16x16
      #upsample(layers[1].shape[1], feature_chanel),  # 16x16 -> 32x32
      #upsample(layers[0].shape[1], feature_chanel),   # 32x32 -> 64x64
  ]

  inputs = tf.keras.layers.Input(shape=input_shape)
  x = inputs
  standardization = ImageStandardization()
  x = standardization(inputs)

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate(axis=feature_chanel)
    x = concat([x, skip])

  # This is the last layer of the model
  Conv2DTranspose = tf.keras.layers.Conv2DTranspose(classes, 3, strides=2,padding='same', name='logits')
  x = Conv2DTranspose(x)

  #seg = tf.argmax(x, axis=-1, name='segmentation')

  #model = tf.keras.Model(inputs=inputs, outputs=[x, seg], name='UnetSegmentation')
  model = tf.keras.Model(inputs=inputs, outputs=[x], name='UnetSegmentation')

  return model