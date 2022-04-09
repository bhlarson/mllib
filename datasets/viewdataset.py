# Based on https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb

import argparse
import json
import os
import sys
import tensorflow as tf
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('')), '..')))
sys.path.insert(0, os.path.abspath(''))
from segment.data import input_fn
from segment.display import DrawFeatures
from networks.unet import unet_model
from pymlutil.s3 import s3store

#tf.config.experimental_run_functions_eagerly(True)

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
parser.add_argument('-min', action='store_true', help='If set, minimum training to generate output.')
parser.add_argument('-min_steps', type=int, default=5, help='Number of min steps.')
parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
parser.add_argument('-trainingset_dir', type=str, default='/store/training/2021-01-12-08-26-56-cocoseg', help='Path training set tfrecord')
parser.add_argument('--trainingset', type=str, default='2021-01-12-08-26-56-cocoseg', help='training set')

parser.add_argument('-epochs', type=int, default=2, help='Number of training epochs')
parser.add_argument('-batch_size', type=int, default=1, help='Number of examples per batch.')
parser.add_argument('-crops', type=int, default=1, help='Crops/image/step')                
parser.add_argument('-training_crop', type=json.loads, default='[512, 512]', help='Training crop size [height, width]')
parser.add_argument('-train_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB') 

def main(args):

    creds = {}
    with open(args.credentails) as json_file:
        creds = json.load(json_file)
    if not creds:
        print('Failed to load credentials file {}. Exiting'.format(args.credentails))

    s3def = creds['s3'][0]
    s3 = s3store(s3def['address'], 
                 s3def['access key'], 
                 s3def['secret key'], 
                 tls=s3def['tls'], 
                 cert_verify=s3def['cert_verify'], 
                 cert_path=s3def['cert_path']
                 )

    trainingset = '{}/{}/'.format(s3def['sets']['trainingset']['prefix'] , args.trainingset)
    print('Load training set {}/{} to {}'.format(s3def['sets']['trainingset']['bucket'],trainingset,args.trainingset_dir ))
    s3.Mirror(s3def['sets']['trainingset']['bucket'], trainingset, args.trainingset_dir)

    trainingsetDescriptionFile = '{}/description.json'.format(args.trainingset_dir)
    trainingsetDescription = json.load(open(trainingsetDescriptionFile))

    config = {
        'batch_size': args.batch_size,
        'trainingset': args.trainingset,
        'trainingset description': trainingsetDescription,
        'input_shape': [args.training_crop[0], args.training_crop[1], args.train_depth],
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
        'image_crops': args.crops,
        'epochs': args.epochs,
        'area_filter_min': 25,
        'channel_order': 'channels_last',
        's3_address':s3def['address'],
        's3_sets':s3def['sets'],
        'min':args.min,
        }

    # ## Train the model
    # Now, all that is left to do is to compile and train the model. The loss being used here is `losses.SparseCategoricalCrossentropy(from_logits=True)`. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and `losses.SparseCategoricalCrossentropy(from_logits=True)` is the recommended loss for 
    # such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.

    train_dataset = input_fn('train', args.trainingset_dir, config)
    val_datast = input_fn('val', args.trainingset_dir, config)

    outpath = 'test'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    train_images = config['batch_size'] # Guess training set if not provided
    val_images = config['batch_size']

    for dataset in trainingsetDescription['sets']:
        if(dataset['name']=="train"):
            train_images = dataset["length"]
        if(dataset['name']=="val"):
            val_images = dataset["length"]
    steps_per_epoch=int(train_images/config['batch_size'])        
    validation_steps=int(val_images/config['batch_size'])
              
    if(args.min):
        steps_per_epoch= min(args.min_steps, steps_per_epoch)
        validation_steps=min(args.min_steps, validation_steps)
        config['epochs'] = 1


    try:
        i=0
        j=0
        k=0
        iterator = iter(train_dataset)
        for i in range(config['epochs']):
            for j in range(steps_per_epoch):
                image, mask  = iterator.get_next()
                for k in range(image.shape[0]):
                    img = tf.squeeze(image[k]).numpy().astype(np.uint8)
                    ann = tf.squeeze(mask[k]).numpy().astype(np.uint8)
                    img = cv.cvtColor(img, cv2.COLOR_RGB2BGR)
                    iman = DrawFeatures(img, ann, config)
                    inxstr = '{:02d}_{:04d}'.format(i, config['batch_size']*j+k)
                    cv2.imwrite('{}/train_iman{}.png'.format(outpath, inxstr), iman)
    except:
        print("Write train_dataset failed epoch {} step {} image {}".format(i, j, k))

    try:
        j=0
        k=0
        iterator = iter(val_datast)
        for j in range(validation_steps):
            image, mask  = iterator.get_next()
            for k in range(image.shape[0]):
                img = tf.squeeze(image[k]).numpy().astype(np.uint8)
                ann = tf.squeeze(mask[k]).numpy().astype(np.uint8)

                iman = DrawFeatures(img, ann, config)
                inxstr = '{:04d}'.format(config['batch_size']*j+k)
                cv2.imwrite('{}/val_iman{}.png'.format(outpath, inxstr), iman)

    except:
        print("Write val_datast failed step {} image {}".format(j, k))

    #WriteImgAn(train_dataset, config, outpath=outpath)
    #WriteImgAn(test_dataset, config, outpath=outpath)
    print("Write complete. Results saved to {}".format(outpath))


if __name__ == '__main__':
  args, unparsed = parser.parse_known_args()
  
  if args.debug:
      print("Wait for debugger attach")
      import ptvsd
      # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
      # Launch applicaiton on remote computer: 
      # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait fcn/train.py
      # Allow other computers to attach to ptvsd at this IP address and port.
      ptvsd.enable_attach(address=('0.0.0.0', args.debug_port), redirect_output=True)
      # Pause the program until a remote debugger is attached

      ptvsd.wait_for_attach()

      print("Debugger attached")

  main(args)
