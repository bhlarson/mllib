# Based on https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb

import argparse
import json
import os
import sys
import shutil
import tempfile
import tensorflow as tf

from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.callbacks import Callback
# Depending on your keras version:-
from tensorflow.keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer

sys.path.insert(0, os.path.abspath(''))
from segment.display import DrawFeatures, WritePredictions
from segment.data import input_fn
from utils.s3 import s3store
from utils.jsonutil import WriteDictJson
from segment.loadmodel import LoadModel

DEBUG = False


#disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')

parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
parser.add_argument('-model_precision', type=str, default='FP16', choices=['FP32', 'FP16', 'INT8'], help='Model Optimization Precision.')

parser.add_argument('--datasetprefix', type=str, default='dataset', help='Dataset prefix')
parser.add_argument('--modelprefix', type=str, default='model', help='Model prefix')

parser.add_argument('--initialmodel', type=str, default='2021-02-24-10-28-35-cocoseg', help='Initial model.  Empty string if no initial model')
parser.add_argument('--temp_savedmodel', type=str, default='./saved_model', help='Temporary path to savedmodel.')

parser.add_argument('-tensorboard_images_max_outputs', type=int, default=2,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('-batch_size', type=int, default=16, help='Number of examples per batch.')               

parser.add_argument('-learning_rate', type=float, default=1e-3, help='Adam optimizer learning rate.')

parser.add_argument('-training_crop', type=json.loads, default='[480, 512]', help='Training crop size [height, width]')
parser.add_argument('-train_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB')
parser.add_argument('-channel_order', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Channels_last = NHWC, Tensorflow default, channels_first=NCHW')

parser.add_argument('-savedmodel', type=str, default='./saved_model', help='Path to savedmodel.')
defaultsavemodeldir = '{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-cocoseg'))
parser.add_argument('-savedmodelname', type=str, default=defaultsavemodeldir, help='Final model')
parser.add_argument('-weights', type=str, default='imagenet', help='Model initiation weights. None prevens loading weights from pre-trained networks')
parser.add_argument('-description', type=str, default='train UNET segmentation network', help='Describe training experament')

parser.add_argument('-saveonnx', type=bool, default=True, help='Save onnx output')


def main(args):

    print('Start Tensorflow to ONNX conversion')

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

    config = {
        'descripiton': args.description,
        'batch_size': args.batch_size,
        'input_shape': [args.training_crop[0], args.training_crop[1], args.train_depth],
        'learning_rate': args.learning_rate,
        'weights': args.weights,
        'channel_order': args.channel_order,
        's3_address':s3def['address'],
        's3_sets':s3def['sets'],
        'initialmodel':args.initialmodel,
    }

    if args.initialmodel is None or len(args.initialmodel) == 0:
        config['initialmodel'] = None

    tempinitmodel = tempfile.TemporaryDirectory(prefix='initmodel', dir='.')
    modelpath = tempinitmodel.name+'/'+config['initialmodel']
    os.makedirs(modelpath)
    file_count = 0
    s3model=config['s3_sets']['model']['prefix']+'/'+config['initialmodel']
    file_count = s3.GetDir(config['s3_sets']['model']['bucket'], s3model, modelpath)

    onnx_filename = "{}/{}.onnx".format(modelpath, args.modelprefix)    
    onnx_req = "python -m tf2onnx.convert --saved-model {} --opset 10 --output {}".format(modelpath, onnx_filename)
    os.system(onnx_req)

    print('Store {} to {}/{}'.format(onnx_filename, s3def['sets']['model']['bucket'],s3model))
    if not s3.PutFile(s3def['sets']['model']['bucket'], onnx_filename, s3model):
        print("s3.PutFile({},{},{} failed".format(s3def['sets']['model']['bucket'], onnx_filename, s3model))

    obj_name = '{}/{}.onnx'.format(s3model, args.modelprefix)
    objurl = s3.GetUrl(s3def['sets']['model']['bucket'], obj_name)

    print("Tensorflow to ONNX  complete. Results stored {}".format(objurl))


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
