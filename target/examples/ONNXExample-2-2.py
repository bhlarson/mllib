import os
import sys
import argparse
import json
from tensorflow.keras.applications import ResNet50
import numpy as np
import onnx
import keras2onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-image_size', type=json.loads, default='[1, 224, 224, 3]', help='Training size [batch, height, width, colors]')
parser.add_argument('-savedmodel', type=str, default='./saved_model/2020-11-07-10-37-57-cfy', help='Saved model to load if no checkpoint')
parser.add_argument('-onnxmodel', type=str, default='classify.onnx', help='Saved model to load if no checkpoint')
parser.add_argument('-plan', type=str, default='classify.plan', help="TensorRT Plan")
parser.add_argument('-trtmodel', type=str, default='classify.trt', help='Path to TensorRT.')
parser.add_argument('-precision', type=str, default='undefined', choices=['undefined','fp16', 'int8'], help='Path to TensorRT.')


def main(args):
    model = ResNet50(weights='imagenet')
    dummy_input_batch = np.zeros((32, 224, 224, 3))
    model.predict(dummy_input_batch) # warm up


    onnx_model = keras2onnx.convert_keras(model, model.name)
    model_name = "resnet50_onnx_model.onnx"
    onnx.save_model(onnx_model, model_name)


    os._exit(0) # Shut down all kernels so TRT doesn't fight with Tensorflow for GPU memory - TF monopolizes all GPU memory by default

    get_ipython().system('trtexec --onnx=resnet50_onnx_model.onnx --batch=1 --saveEngine=resnet_engine.trt  --explicitBatch 2>&1')


    dummy_input_batch = np.zeros((32, 224, 224, 3))



    f = open("resnet_engine.trt", "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    output = np.empty(1000, dtype = np.float32)

    # Allocate device memory
    d_input = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    def predict_batch(batch): # result gets copied into output
        # Transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # Execute model
        context.execute_async(1, bindings, stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # Syncronize threads
        stream.synchronize()

    print("Warming up...")

    predict_batch(dummy_input_batch)

    print("Done warming up!")

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