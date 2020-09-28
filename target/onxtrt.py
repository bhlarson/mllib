# Convert Tensorflow savedmodel to TensorRT optimized model
import os
import sys
import argparse
import platform
import json
from onnx import ModelProto
import tensorrt as trt

sys.path.insert(0, os.path.abspath(''))
import engine as eng

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-onnxmodel', type=str, default='./saved_model/2020-09-28-14-52-11-dl3/2020-09-28-14-52-11-dl3.onnx', help='Saved model to load if no checkpoint')
parser.add_argument('-image_size', type=json.loads, default='[1, 480, 640, 3]', help='Training crop size [height, width]/  [90, 160],[120, 160],[120, 160], [144, 176],[288, 352], [240, 432],[480, 640],[576,1024],[720, 960], [720,1280],[1080, 1920]')
parser.add_argument('-plan', type=str, default='./saved_model/2020-09-28-14-52-11-dl3/trt-480-640.plan', help="TensorRT Plan")


def main(arg):

    print('Platform: {}'.format(platform.platform()))

    engine = eng.build_engine(arg.onnxmodel, shape=arg.image_size)
    eng.save_engine(engine, arg.plan) 
    
    print("TensorRT Conversion complete. Results saved to {}".format(arg.plan))
    

if __name__ == '__main__':
  arg, unparsed = parser.parse_known_args()
  
  if arg.debug:
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

  main(arg)
