import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TrtInference:
    def __init__(self, serialized_engine, batch_size, height, width, class_dictionary, in_chanels=3, intype='float32', outtype='uint8'):
        self.outtype = outtype
        self.dummy_input_batch = np.zeros((batch_size, height, width, in_chanels), dtype = np.dtype(intype)) 

        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
        self.output = np.empty([batch_size, len(class_dictionary), height, width], dtype = np.dtype(intype)) # Need to set output dtype to FP16 to enable FP16

        self.d_input = cuda.mem_alloc(self.dummy_input_batch.nbytes)
        self.d_output = cuda.mem_alloc(self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

        # Call to load libraries & allocate memory
        pred = self.predict(self.dummy_input_batch)


    def predict(self, batch): # result gets copied into output
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        segmentations = np.argmax(self.output, axis=1).astype(self.outtype)
        
        return segmentations