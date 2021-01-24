# Convert Tensorflow savedmodel to TensorRT optimized model
import os
import sys
import argparse
import platform
import json
import tempfile
import shutil

#from tf2onnx.tfonnx import process_tf_graph
#from tf2onnx import constants, logging, utils, optimizer
#from tf2onnx import tf_loader
#from onnx import ModelProto
#import tensorrt as trt
import engine as eng

sys.path.insert(0, os.path.abspath(''))
from networks.unet import unet_model, unet_compile
from utils.s3 import s3store
from utils.jsonutil import WriteDictJson

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-savedmodel', type=str, default='2021-01-13-18-16-49-cfy', help='Saved model to load if no checkpoint')
parser.add_argument('-work_path', type=str, default='/store/models/convert', help='Saved model to load if no checkpoint')
parser.add_argument('-size_x', type=int, default=224, help='Training image size_x')
parser.add_argument('-size_y', type=int, default=224, help='Training image size_y')
parser.add_argument('-depth', type=int, default=3, help='Training image depth')
parser.add_argument('-channel_order', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Channels_last = NHWC, Tensorflow default, channels_first=NCHW')
parser.add_argument('-image_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB')
parser.add_argument('-batch_size', type=int, default=1, help='Batch Size') 
parser.add_argument('-onxdir', type=str, default='./onnx', help='Path to ONNX models.')
parser.add_argument('-trtdir', type=str, default='./trt', help='Path to TensorRT.')
parser.add_argument('-precision_mode', type=str, default='FP16', help='TF-TRT precision mode FP32, FP16 or INT8 supported.')
parser.add_argument("-output", default='model.onnx', help="output model file")
parser.add_argument("-model_input_names", type=str, default=None, help="model input_names")
parser.add_argument("-model_output_names", type=str, default=None, help="model input_names")

parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
#parser.add_argument("--target", default=",".join(constants.DEFAULT_TARGET), choices=constants.POSSIBLE_TARGETS, help="target platform")
parser.add_argument("-continue_on_error", help="continue_on_error", action="store_true")
parser.add_argument("-verbose", "-v", help="verbose output, option is additive", action="count")
parser.add_argument("-inputs_as_nchw", default=None, help="model input_names")
parser.add_argument("-opset", type=int, default=None, help="opset version to use for onnx domain")

def LoadModel(config, s3, model_dir=None):
    import tensorflow
    model = None 
    print('LoadModel initial model: {}, training directory: {}, '.format(config['initialmodel'], config['training_dir']))
    if config['initialmodel'] is not None:
        tempinitmodel = tempfile.TemporaryDirectory(prefix='initmodel', dir='.')
        modelpath = tempinitmodel.name+'/'+config['initialmodel']
        os.makedirs(modelpath)
        try:
            s3model=config['s3_sets']['model']['prefix']+'/'+config['initialmodel']
            success = s3.GetDir(config['s3_sets']['model']['bucket'], s3model, modelpath)
            model = tensorflow.keras.models.load_model(modelpath) # Load from checkpoint

        except:
            print('Unable to load weghts from http://{}/minio/{}/{}'.format(
                config['s3_address'],
                config['s3_sets']['model']['prefix'],
                modelpath)
            )
            model = None 
        shutil.rmtree(tempinitmodel, ignore_errors=True)

    if model is None:
        if not config['clean'] and config['training_dir'] is not None:
            try:
                model = tensorflow.keras.models.load_model(config['training_dir'])
            except:
                print('Unable to load weghts from {}'.format(config['training_dir']))

    if model is None:

        model = keras.applications.ResNet50V2(include_top=True, weights=config['init_weights'], 
            input_shape=config['shape'], classes=config['classes'], classifier_activation=None)

        if not config['clean'] and config['training_dir'] is not None:
            try:
                model.load_weights(config['training_dir'])
            except:
                print('Unable to load weghts from {}'.format(config['training_dir']))

    if model:
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return model


def main(args):

    print('Platform: {}'.format(platform.platform()))

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
        'initialmodel':args.savedmodel,
        's3_address':s3def['address'],
        's3_sets':s3def['sets'],
        'savedmodel': args.savedmodel,
        'training_dir': None,
        'clean': False,
        'shape': (args.size_y, args.size_x, args.depth),
        'training_dir': None,

    }

    using_tensorflow = True
    using_tensorrt = True

    modelpath = '{}/{}'.format(s3def['sets']['model']['prefix'], config['savedmodel'])
    savedmodelpath = '{}/{}'.format(args.work_path, config['savedmodel'])
    onnxname = '{}/{}.onnx'.format(savedmodelpath, config['savedmodel'])
    trtenginename = '{}/{}.plan'.format(savedmodelpath, config['savedmodel'])

    if using_tensorflow:
        import tensorflow
        import keras2onnx
        import onnx

        print('Tensorflow: {}'.format(tensorflow.__version__))

        model =  LoadModel(config, s3)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        if not s3.GetDir(s3def['sets']['model']['bucket'], modelpath, savedmodelpath):
            print('Failed to load model')

        onnx.save_model(onnx_model, onnxname)
        # Unload models to free memory for ONNX->TRT conversion

    if using_tensorrt:
    
        oscmd = 'trtexec --onnx={} --batch=1 --saveEngine={}  --explicitBatch 2>&1'.format(onnxname, trtenginename)
        os.system(oscmd)


    '''
    trtenginename = '{}/{}.plan'.format(savedmodelpath, config['savedmodel'])

    graph_def, inputs, outputs = tf_loader.from_saved_model(savedmodelpath, input_names=args.model_input_names, output_names=args.model_output_names)

    print("inputs: {}".format(inputs))
    print("outputs: {}".format(outputs))

    with tensorflow.Graph().as_default() as tf_graph:
        const_node_values = None
        tensorflow.import_graph_def(graph_def, name='')
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

        model_proto = onnx_graph.make_model(config['savedmodel'])
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

    '''

    s3.PutFile(s3def['sets']['model']['bucket'], onnxname, modelpath)
    s3.PutFile(s3def['sets']['model']['bucket'], trtenginename, modelpath)

    print("Conversion complete. Results saved to {}".format(trtenginename))
    

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

      print("Debugger attached")

  main(args)
