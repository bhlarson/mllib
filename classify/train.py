import argparse
import os
import sys

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

print('Python Version {}'.format(sys.version))
print('Tensorflow version {}'.format(tf.__version__))
print('GPU Available: {}'.format(tf.test.is_gpu_available()))
if(tf.test.is_gpu_available()):
  print('GPU Devices: {}'.format(tf.test.gpu_device_name()))

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',help='Wait for debugger attach')
parser.add_argument('--batch_size', type=int, default=32, help='Number of examples per batch.')
parser.add_argument('--classes', type=int, default=5, help='Number of examples per batch.')
parser.add_argument('--size_x', type=int, default=299, help='Training image size_x')
parser.add_argument('--size_y', type=int, default=299, help='Training image size_y')
parser.add_argument('--depth', type=int, default=3, help='Training image depth')
parser.add_argument('--epochs', type=int, default=3, help='Training epochs')

def prepare_image(image, label, config):
    image = tf.image.resize_with_crop_or_pad(image, config['shape'][0], config['shape'][1])
    return image, label

def input_fn(config):
    dataset, metadata = tfds.load('tf_flowers', with_info=True, split=config['split'], shuffle_files=True, as_supervised=True)
    dataset = dataset.map(lambda features, label: prepare_image(features, label, config) , num_parallel_calls = 10)
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.prefetch(config['batch_size'])
    return dataset, metadata

def main(args):
    tf.config.experimental_run_functions_eagerly(False)
    config = {
        'batch_size': args.batch_size,
        'shape': (args.size_y, args.size_x, args.depth),
        'split': tfds.Split.TRAIN,
    }

    model = keras.applications.Xception(
        include_top=True, weights='imagenet', input_tensor=None, input_shape=config['shape'],
        pooling=None, classifier_activation='softmax'
    )

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.summary()
    dataset, datasetdata = input_fn(config)

    print(datasetdata)

    model.fit(dataset, epochs=args.epochs)

if __name__ == '__main__':
  args, unparsed = parser.parse_known_args()
  
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