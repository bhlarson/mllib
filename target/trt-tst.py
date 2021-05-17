# tf2-onnx.py -> onnx-trt.py -> trt-tst.py

import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')

def main(args):
    
    USE_FP16 = True
    BATCH_SIZE = 32
    target_dtype = np.float16 if USE_FP16 else np.float32
    dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3), dtype = np.float32) 

    f = open("resnet_engine.trt", "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) # Need to set output dtype to FP16 to enable FP16

    # Allocate device memory
    d_input = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    def predict(batch): # result gets copied into output
        # Transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # Execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # Syncronize threads
        stream.synchronize()
        
        return output

    print("Warming up...")

    predict(dummy_input_batch)

    print("Done warming up!")

    pred = predict(dummy_input_batch) # Check TRT performance

    print ("Prediction: " + str(np.argmax(output)))

    pred = predict(dummy_input_batch)

    pred.shape

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