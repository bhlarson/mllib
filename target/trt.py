# Based on https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb

import os
import argparse
import platform
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-savedmodel', type=str, default='./saved_model/2020-09-04-05-14-30-dl3', help='Saved model to load if no checkpoint')
parser.add_argument('-trtdir', type=str, default='./trt', help='Path to TensorRT.')
parser.add_argument('-precision_mode', type=str, default='FP16', help='TF-TRT precision mode FP32, FP16 or INT8 supported.')

def main(unparsed):

    print('Platform: {}'.format(platform.platform()))
    print('Tensorflow: {}'.format(tf.__version__))

    savemodelname = os.path.basename(os.path.normpath(FLAGS.savedmodel))
    trtOutPath = '{}/{}-{}'.format(FLAGS.trtdir, savemodelname,FLAGS.precision_mode)

    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(precision_mode="FP16")

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=FLAGS.savedmodel, conversion_params=conversion_params)
    converter.convert()
    converter.save(trtOutPath)

    print("TensorRT Conversion complete. Results saved to {}".format(trtOutPath))


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

  main(unparsed)
