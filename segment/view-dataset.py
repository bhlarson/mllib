# Based on https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb

import argparse
import json
import os
import sys
import tensorflow as tf
from datetime import datetime
import numpy as np
import cv2

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('')), '..')))
sys.path.insert(0, os.path.abspath(''))
from segment.data import input_fn
from segment.display import DrawFeatures
from networks.unet import unet_model

#tf.config.experimental_run_functions_eagerly(True)


parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')

parser.add_argument('-record_dir', type=str, default='cocorecord', help='Path training set tfrecord')


parser.add_argument('-epochs', type=int, default=1,
                    help='Number of training epochs')

parser.add_argument('-batch_size', type=int, default=8, help='Number of examples per batch.')

parser.add_argument('-crops', type=int, default=1, help='Crops/image/step')                

parser.add_argument('-training_crop', type=json.loads, default='[512, 512]', help='Training crop size [height, width]')
parser.add_argument('-train_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB') 

def main(unparsed):
    trainingsetDescriptionFile = '{}/description.json'.format(FLAGS.record_dir)
    trainingsetDescription = json.load(open(trainingsetDescriptionFile))

    config = {
        'batch_size': FLAGS.batch_size,
        'trainingset': trainingsetDescription,
        'input_shape': [FLAGS.training_crop[0], FLAGS.training_crop[1], FLAGS.train_depth],
        'classScale': 0.001, # scale value for each product class
        'augment_rotation' : 5., # Rotation in degrees
        'augment_flip_x': False,
        'augment_flip_y': True,
        'augment_brightness':0.,
        'augment_contrast': 0.,
        'augment_shift_x': 0.0, # in fraction of image
        'augment_shift_y': 0.0, # in fraction of image
        'scale_min': 0.75, # in fraction of image
        'scale_max': 1.25, # in fraction of image
        'ignore_label': trainingsetDescription['classes']['ignore'],
        'classes': trainingsetDescription['classes']['classes'],
        'image_crops': FLAGS.crops,
        'epochs': FLAGS.epochs,
        'area_filter_min': 25,
        }

    # ## Train the model
    # Now, all that is left to do is to compile and train the model. The loss being used here is `losses.SparseCategoricalCrossentropy(from_logits=True)`. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and `losses.SparseCategoricalCrossentropy(from_logits=True)` is the recommended loss for 
    # such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.

    train_dataset = input_fn(True, FLAGS.record_dir, config)
    test_dataset = input_fn(False, FLAGS.record_dir, config)

    outpath = 'test'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    iterator = iter(train_dataset)
    for i in range(5):
        image, mask  = iterator.get_next()
        for j in range(config['batch_size']):
            img = tf.squeeze(image[j]).numpy().astype(np.uint8)
            ann = tf.squeeze(mask[j]).numpy().astype(np.uint8)

            iman = DrawFeatures(img, ann, config)
            iman = cv2.cvtColor(iman, cv2.COLOR_RGB2BGR)
            cv2.imwrite('{}/ann-pred{}.png'.format(outpath, j+i*config['batch_size']), iman)


    #WriteImgAn(train_dataset, config, outpath=outpath)
    #WriteImgAn(test_dataset, config, outpath=outpath)
    print("Write complete. Results saved to {}".format(outpath))


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

      print("Debugger attached")

  main(unparsed)
