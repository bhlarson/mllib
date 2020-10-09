# Convert Tensorflow savedmodel to TensorRT optimized model
import os
import sys
import argparse
import platform
import json
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx import constants, logging, utils, optimizer
from tf2onnx import tf_loader
from onnx import ModelProto
import tensorrt as trt
import engine as eng

sys.path.insert(0, os.path.abspath(''))
from networks.unet import unet_model, unet_compile

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-savedmodel', type=str, default='./saved_model/2020-09-04-05-14-30-dl3', help='Saved model to load if no checkpoint')
parser.add_argument('-image_size', type=json.loads, default='[480, 640]', help='Training crop size [height, width]/  [90, 160],[120, 160],[120, 160], [144, 176],[288, 352], [240, 432],[480, 640],[576,1024],[720, 960], [720,1280],[1080, 1920]')
parser.add_argument('-image_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB')
parser.add_argument('-batch_size', type=int, default=1, help='Batch Size') 
parser.add_argument('-onnxdir', type=str, default='./onnx', help='Path to ONNX models.')
parser.add_argument('-trtdir', type=str, default='./trt', help='Path to TensorRT.')
parser.add_argument('-precision_mode', type=str, default='FP16', help='TF-TRT precision mode FP32, FP16 or INT8 supported.')
parser.add_argument("-output", default='model.onnx', help="output model file")
parser.add_argument("-model_input_names", type=str, default=None, help="model input_names")
parser.add_argument("-model_output_names", type=str, default=None, help="model input_names")

parser.add_argument("--target", default=",".join(constants.DEFAULT_TARGET), choices=constants.POSSIBLE_TARGETS, help="target platform")
parser.add_argument("-continue_on_error", help="continue_on_error", action="store_true")
parser.add_argument("-verbose", "-v", help="verbose output, option is additive", action="count")
parser.add_argument("-inputs_as_nchw", default=None, help="model input_names")
parser.add_argument("-opset", type=int, default=None, help="opset version to use for onnx domain")


def main(args):

    print('Platform: {}'.format(platform.platform()))

    savemodelname = os.path.basename(os.path.normpath(args.savedmodel))
    onnxmodelname = '{}/{}.onnx'.format(args.onnxdir, savemodelname)
    trtenginename = '{}/{}.plan'.format(args.trtdir, savemodelname)
    osstr = 'python -m tf2onnx.convert  --input {} --output {}'.format(args.savedmodel, onnxmodelname)
    print(osstr)
    os.system('python -m tf2onnx.convert  --input {} --output {}'.format(savemodelname, onnxmodelname))

    shape = [args.batch_size , args.image_size[0], args.image_size[1], args.image_depth]
    engine = eng.build_engine_file(onnxmodelname, shape=shape)
    eng.save_engine(engine, trtenginename) 

    print("Conversion complete. Results saved to {}".format(trtOutPath))
    

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

      print("Debugger attach")

  main(args)
