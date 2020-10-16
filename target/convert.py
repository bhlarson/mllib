# Based on https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb

import argparse
import json
import os
import sys
import math
import shutil
import cv2
import tensorflow as tf
#from tensorflow.python.framework.ops import disable_eager_execution
#import tensorflow_model_optimization as tfmot
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.callbacks import Callback
# Depending on your keras version:-
from tensorflow.keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer

#sys.path.insert(0, os.path.abspath(''))
#from segment.display import DrawFeatures, WritePredictions
#from segment.data import input_fn
#from networks.unet import unet_model, unet_compile
import keras2onnx

DEBUG = False


#disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-tflite', action='store_true',help='Run tensorflow lite postprocessing')
parser.add_argument('-trt', action='store_true',help='Run TensorRT postprocessing')
parser.add_argument('-onnx', action='store_true',help='Run ONNX postprocessing')

parser.add_argument('-model_precision', type=str, default='FP16', choices=['FP32', 'FP16', 'INT8'], help='Model Optimization Precision.')
parser.add_argument('-channel_order', type=str, default='channels_first', choices=['channels_first' or 'channels_last'], help='Model Optimization Precision.')

parser.add_argument("-strategy", type=str, default='onedevice', help="Replication strategy. 'mirrored', 'onedevice' now supported ")
parser.add_argument("-devices", type=json.loads, default=["/gpu:0"],  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')

parser.add_argument('-savedmodel', type=str, default='./saved_model/2020-10-16-12-53-11', help='Path to savedmodel.')

parser.add_argument('-trtmodel', type=str, default='./trt', help='Path to TensorRT.')
parser.add_argument('-tbport', type=int, default=6006, help='Tensorboard network port.')
parser.add_argument('-weights', type=str, default='imagenet', help='Model initiation weights. None prevens loading weights from pre-trained networks')


def main(unparsed):

    if FLAGS.tflite: # convert to Tensorflow Lite
        converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.savedmodel)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        #samplefiles = get_samples(FLAGS.sample_dir, FLAGS.match)
        #converter.representative_dataset = lambda:representative_dataset_gen(samplefiles)
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #converter.inference_input_type = tf.uint8  # or tf.uint8
        #converter.inference_output_type = tf.uint8  # or tf.uint8
        tflite_model = converter.convert()
        outflite = '{}/{}-{}.tflite'.format(FALGS.savedmodel,FLAGS.savedmodel, FLAGS.model_precision)
        open(outflite, "wb").write(tflite_model)

    if FLAGS.onnx:
        import keras2onnx

        onnx_name = '{}/{}.onnx'.format(FLAGS.savedmodel, os.path.basename(FLAGS.savedmodel))
        model = tf.keras.models.load_model(FLAGS.savedmodel)
        onnx_model = keras2onnx.convert_keras(model, model.name, debug_mode=True)
        content = onnx_model.SerializeToString()
        keras2onnx.save_model(onnx_model, onnx_name)

    if FLAGS.trt: # Use TF-TRT to convert savedmodel to TensorRT
        trtpath = '{}/{}-{}/'.format(FLAGS.trtmodel, os.path.basename(FLAGS.savedmodel), FLAGS.model_precision)
        from tensorflow.python.compiler.tensorrt import trt_convert as trt

        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        conversion_params = conversion_params._replace(precision_mode=FLAGS.model_precision)

        converter = trt.TrtGraphConverterV2(input_saved_model_dir=FLAGS.savedmodel, conversion_params=conversion_params)
        converter.convert()
        converter.save(trtpath)

        # How do I produce and output a TensorRT .play file with TensorRT
        # https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#worflow-with-savedmodel
        # create_inference_graph has been removed from Tensorflow 2
        '''def input_fn():
            for _ in range(num_runs):
                inputTensor = np.random.normal(size=(1, FLAGS.training_crop[0], FLAGS.training_crop[1], FLAGS.train_depth)).astype(np.float32)
                yield inputTensor
        converter.build(input_fn=input_fn)

        for n in trt_graph.node:
            if n.op == "TRTEngineOp":
                print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
                with tf.gfile.GFile("%s.plan" % (n.name.replace("/", "_")), 'wb') as f:
                f.write(n.attr["serialized_segment"].s)
            else:
                print("Exclude Node: %s, %s" % (n.op, n.name.replace("/", "_")))'''

        print("TF-TRT conversion results saved to {}".format(trtpath))


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
