# Convert Tensorflow savedmodel to TensorRT optimized model
import os
import sys
import argparse
import platform
import json
from tf2onnx import convert
import tensorrt as trt

sys.path.insert(0, os.path.abspath(''))

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-image_size', type=json.loads, default='[1, 224, 224, 3]', help='Training size [batch, height, width, colors]')
parser.add_argument('-savedmodel', type=str, default='./saved_model/2020-11-07-10-37-57-cfy', help='Saved model to load if no checkpoint')
parser.add_argument('-onnxmodel', type=str, default='classify.onnx', help='Saved model to load if no checkpoint')
parser.add_argument('-plan', type=str, default='classify.plan', help="TensorRT Plan")
parser.add_argument('-trtmodel', type=str, default='classify.trt', help='Path to TensorRT.')
parser.add_argument('-precision', type=str, default='undefined', choices=['undefined','fp16', 'int8'], help='Path to TensorRT.')

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, config):

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as onnxParser:
        #builder.max_workspace_size = (256 << 20)

        #with open(onnx_path, 'rb') as model:
        #    succeeded = onnxParser.parse(model.read())
        #print('onnxParser onnx succeeded={} with {} errors'.format(succeeded, onnxParser.num_errors))
        #for i in range(onnxParser.num_errors):
        #    print('parse error {}: {}'.format(i, onnxParser.get_error(i)))
        if config['precision']=='int8':
            builder.int8_mode = True
            builder.strict_type_constraints = True
        elif config['precision']=='fp16':
            builder.fp16_mode = True
            builder.strict_type_constraints = True
        succeeded = onnxParser.parse_from_file(onnx_path)

        if succeeded:
            network.get_input(0).shape = config['image_size'][1:]
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

def tfonnxconvert(config, paths_to_check=None):
    args = ['',
            '--saved-model',
            config['savedmodel'],
            '--output',
            config['onnxmodel']
    ]
    
    """ run case and clean up """
    if paths_to_check is None:
        paths_to_check = [args[-1]]
    sys.argv = args
    convert.main()
    ret = True
    for p in paths_to_check:
        if os.path.exists(p):
            os.remove(p)
        else:
            ret = False
    return ret

def WriteDictJson(outdict, path):
    jsonStr = json.dumps(outdict, sort_keys=False, indent=4)
    f = open(path,"w")
    f.write(jsonStr)
    f.close()

def main(args):

    print('Platform: {}'.format(platform.platform()))
    print('Python: {}'.format(platform.python_version()))


    onnxmodelname = '{}/{}'.format(args.savedmodel, args.onnxmodel)
    planname = '{}/{}'.format(args.savedmodel, args.plan)
    trtname = '{}/{}'.format(args.savedmodel, args.trtmodel)

    config = {
        'savedmodel':args.savedmodel,
        'onnxmodel':onnxmodelname,
        'trtmodel':trtname,
        'platform':platform.platform(),
        'python':platform.python_version(),
        'tf2onnx version': sys.modules['tf2onnx'].__version__,
        'tensorrt version': sys.modules['tensorrt'].__version__,
        'precision':args.precision,
        'image_size': args.image_size
    }

    tf2onx = 'python -m tf2onnx.convert --saved-model {} --output {}'.format(args.savedmodel, onnxmodelname)
    os.system(tf2onx)


    #if tfonnxconvert(config):   

    # Convert using TensorRT python API (https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
    engine = build_engine(onnx_path=onnxmodelname, config=config)
    if engine:
        save_engine(engine, args.plan)

    # Convert using trtexec (https://github.com/NVIDIA/TensorRT/blob/master/samples/opensource/trtexec/README.md)
    input_shape = '{}x{}x{}x{}'.format(args.image_size[0], args.image_size[1], args.image_size[2], args.image_size[3])
    os.system('trtexec --onnx={} --saveEngine={} --shapes=input:{} --explicitBatch=1 2>&1'.format(onnxmodelname, trtname, input_shape))
        
    WriteDictJson(config, '{}/tfonxtrt.json'.format(args.savedmodel))
    print("TensorRT Conversion complete. Results saved to {}".format(trtname))
    

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
