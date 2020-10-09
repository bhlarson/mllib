# Convert Tensorflow savedmodel to TensorRT optimized model
import os
import sys
import argparse
import platform
import json
import tensorflow as tf
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
parser.add_argument('-onxdir', type=str, default='./onnx', help='Path to ONNX models.')
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
    print('Tensorflow: {}'.format(tf.__version__))

    logging.set_level(1)

    savemodelname = os.path.basename(os.path.normpath(args.savedmodel))
    onnxname = '{}/{}.onnx'.format(args.onxdir, savemodelname)
    trtenginename = '{}/{}.plan'.format(args.trtdir, savemodelname)

    graph_def, inputs, outputs = tf_loader.from_saved_model(args.savedmodel, input_names=args.model_input_names, output_names=args.model_output_names)

    print("inputs: %s", inputs)
    print("outputs: %s", outputs)

    with tf.Graph().as_default() as tf_graph:
        const_node_values = None
        tf.import_graph_def(graph_def, name='')
    with tf_loader.tf_session(graph=tf_graph):
        onnx_graph = process_tf_graph(tf_graph,
                             continue_on_error=args.continue_on_error,
                             target=args.target,
                             opset=args.opset,
                             custom_op_handlers={},
                             extra_opset=[],
                             shape_override=None,
                             input_names=inputs,
                             output_names=outputs,
                             inputs_as_nchw=args.inputs_as_nchw)

        model_proto = onnx_graph.make_model(savemodelname)
        with open(onnxname, "wb") as f:
            f.write(model_proto.SerializeToString())

    #onnx_graph = optimizer.optimize_graph(onnx_graph)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#import_onnx_python
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        #builder.max_workspace_size = (256 << 20)
        with open(onnxname, 'rb') as model:
            parser.parse(model.read())
        #network.get_input(0).shape = [args.batch_size, args.image_size[0], args.image_size[1], args.image_depth]
        engine = builder.build_cuda_engine(network)

        buf = engine.serialize()
        with open(trtenginename, 'wb') as f:
            f.write(buf)

        #builder.max_workspace_size = (256 << 20)
        #parser.parse(model_proto.SerializeToString())
        #network.get_input(0).shape = [args.batch_size, args.image_size[0], args.image_size[1], args.image_depth]
        #engine = builder.build_cuda_engine(network)

        #eng.save_engine(engine, trtenginename) 

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
