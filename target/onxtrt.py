# Convert Tensorflow savedmodel to TensorRT optimized model
import os
import sys
import argparse
import platform
import json
#from onnx import ModelProto
import tensorrt as trt

sys.path.insert(0, os.path.abspath(''))

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-image_size', type=json.loads, default='[1, 480, 640, 3]', help='Training crop size [height, width]/  [90, 160],[120, 160],[120, 160], [144, 176],[288, 352], [240, 432],[480, 640],[576,1024],[720, 960], [720,1280],[1080, 1920]')
parser.add_argument('-onnxmodel', type=str, default='./saved_model/2020-10-02-00-37-58-dl3/2020-10-02-00-37-58-dl3.onnx', help='Saved model to load if no checkpoint')
parser.add_argument('-plan', type=str, default='./saved_model/2020-10-02-00-37-58-dl3/trt-480-640.plan', help="TensorRT Plan")
parser.add_argument('-trtmodel', type=str, default='./saved_model/2020-10-02-00-37-58-dl3/unet.trt', help='Path to TensorRT.')

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, input_shape = [1,224,224,3]):

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as onnxParser:
        builder.max_workspace_size = (256 << 20)

        #with open(onnx_path, 'rb') as model:
        #    succeeded = onnxParser.parse(model.read())
        #print('onnxParser onnx succeeded={} with {} errors'.format(succeeded, onnxParser.num_errors))
        #for i in range(onnxParser.num_errors):
        #    print('parse error {}: {}'.format(i, onnxParser.get_error(i)))
        
        succeeded = onnxParser.parse_from_file(onnx_path)

        if succeeded:
            network.get_input(0).shape = input_shape
            last_layer = network.get_layer(network.num_layers - 1)
            network.mark_output(last_layer.get_output(0))
            engine = builder.build_cuda_engine(network)
        else:
            engine = None
        
        return engine

def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)
def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

def main(args):

    print('Platform: {}'.format(platform.platform()))
    print('Python: {}'.format(platform.python_version()))

    # Convert using TensorRT python API (https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
    engine = build_engine(onnx_path=args.onnxmodel, input_shape=args.image_size)
    if engine:
        save_engine(engine, args.plan)

    # Convert using trtexec (https://github.com/NVIDIA/TensorRT/blob/master/samples/opensource/trtexec/README.md)
    input_shape = '{}x{}x{}x{}'.format(args.image_size[0], args.image_size[1], args.image_size[2], args.image_size[3])
    os.system('trtexec --onnx={} --saveEngine={} --shapes=input:{} --explicitBatch=1 2>&1'.format(args.onnxmodel, args.trtmodel, input_shape))
    
    print("TensorRT Conversion complete. Results saved to {}".format(args.plan))
    

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
