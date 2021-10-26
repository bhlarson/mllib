# tf2-onnx.py -> onnx-trt.py -> trt-tst.py

BATCH_SIZE = 32
import argparse
import os
import sys
import json
import shutil
import numpy
import onnx
import tensorrt as trt

sys.path.insert(0, os.path.abspath(''))
from utils.s3 import s3store

__all__ = ['MB', 'GB', 'build_tensorrt_engine']

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def MB(val):
    return val * 1 << 20


def GB(val):
    return val * 1 << 30

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
parser.add_argument('-savedmodelname', type=str, default="segmin", help='Path in S3 model')
parser.add_argument('-targetname', type=str, default="segment_nas_prune_640x640_20211015", help='Final model wiout extension')
parser.add_argument('-workdir', type=str, default="trt", help='Working directory')
parser.add_argument('-onnxname', type=str, default="segment_nas_prune_640x640_20211015.onnx", help='Onnx file name')
parser.add_argument('-workspace_memory', type=int, default=4096, help='trtexec workspace size in megabytes')
parser.add_argument('-batch_size', type=int, default=1, help='Number of examples per batch.') 
parser.add_argument('-image_size', type=json.loads, default='[512 640]', help='Image size') 
parser.add_argument('-fp16',  default=True, help='If set, Generate FP16 model.')



class INT8Calibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, data, cache_file, batch_size=8):
        """
        :param data: numpy array with shape (N, C, H, W)
        :param cache_file:
        :param batch_size:
        """
        trt.IInt8EntropyCalibrator2.__init__(self)

        self._cache_file = cache_file
        self._batch_size = batch_size

        """
        data is numpy array in float32, caution: each image should be normalized
        """
        assert data.ndim == 4 and data.dtype == numpy.float32
        self._data = numpy.array(data, dtype=numpy.float32, order='C')

        self._current_index = 0

        # Allocate enough memory for a whole batch.
        self._device_input = cuda.mem_alloc(self._data[0].nbytes * self._batch_size)

    def get_batch_size(self):
        return self._batch_size

    def get_batch(self, names, p_str=None):
        if self._current_index + self._batch_size > self._data.shape[0]:
            return None

        current_batch = int(self._current_index / self._batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self._batch_size))

        batch = self._data[self._current_index:self._current_index + self._batch_size]
        cuda.memcpy_htod(self._device_input, batch)
        self._current_index += self._batch_size
        return [self._device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self._cache_file):
            with open(self._cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self._cache_file, "wb") as f:
            f.write(cache)


def build_tensorrt_engine(onnx_file_path,
                          engine_save_path,
                          precision_mode='fp32',
                          max_workspace_size=GB(1),  # in bytes
                          max_batch_size=1,
                          min_timing_iterations=2,
                          avg_timing_iterations=2,
                          int8_calibrator=None):
    """
    :param onnx_file_path:
    :param engine_save_path:
    :param precision_mode:
    :param max_workspace_size: The maximum workspace size. The maximum GPU temporary memory which the engine can use at
    :param max_batch_size:
    :param min_timing_iterations:
    :param avg_timing_iterations:
    :param int8_calibrator:
    :return:
    """
    assert os.path.exists(onnx_file_path)
    assert precision_mode in ['fp32', 'fp16', 'int8']

    trt_logger = trt.Logger(trt.Logger.VERBOSE)

    builder = trt.Builder(trt_logger)
    if precision_mode == 'fp16':
        assert builder.platform_has_fast_fp16, 'platform does not support fp16 mode!'
    if precision_mode == 'int8':
        assert builder.platform_has_fast_int8, 'platform does not support int8 mode!'
        assert int8_calibrator is not None, 'calibrator is not provided!'

    network = builder.create_network(EXPLICIT_BATCH)

    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_file_path, 'rb') as onnx_fin:
        parser.parse(onnx_fin.read())

    num_error = parser.num_errors
    if num_error != 0:
        for i in range(num_error):
            temp_error = parser.get_error(i)
            print(temp_error.desc())
        return

    config = builder.create_builder_config()

    if precision_mode == 'int8':
        config.int8_calibrator = int8_calibrator
        config.set_flag(trt.BuilderFlag.INT8)
    elif precision_mode == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        pass

    config.max_workspace_size = max_workspace_size
    config.min_timing_iterations = min_timing_iterations
    config.avg_timing_iterations = avg_timing_iterations
    builder.max_batch_size = max_batch_size
    try:
        engine = builder.build_engine(network, config)
    except:
        print('Engine build unsuccessfully!')
        return False

    if engine is None:
        print('Engine build unsuccessfully!')
        return False

    if not os.path.exists(os.path.dirname(engine_save_path)):
        os.makedirs(os.path.dirname(engine_save_path))

    serialized_engine = engine.serialize()
    with open(engine_save_path, 'wb') as fout:
        fout.write(serialized_engine)

    print('Engine built successfully!')
    return True

def main(args):
    failed = False

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
                 cert_verify=s3def['cert verify'], 
                 cert_path=s3def['cert path']
                 )

    workdir = '{}/{}'.format(args.workdir, args.savedmodelname)
    inobj = '{}/{}/{}'.format(s3def['sets']['model']['prefix'],args.savedmodelname, args.onnxname)
    objpath = '{}/{}'.format(s3def['sets']['model']['prefix'],args.savedmodelname)

    infile = '{}/{}'.format(workdir, args.onnxname)

    if not s3.GetFile(s3def['sets']['model']['bucket'], inobj, infile):
        print('Failed to load {}/{} to {}'.format(s3def['sets']['model']['bucket'], inobj, infile ))
        failed = True
        return failed

    onnx_model = onnx.load(infile)
    inputs = onnx_model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_value = args.batch_size
        input.type.tensor_type.shape.dim[2:3] = args.image_size

    fixedfile = '{}/fixed-{}'.format(workdir, args.onnxname)
    onnx.save_model(onnx_model, fixedfile)

    targetname = args.targetname
    params = ''
    if args.fp16:
        targetname += '-fp16'
        params = '--fp16'
    outfile = '{}/{}.trt'.format(workdir, targetname)
    logfile = '{}/{}-trt.log'.format(workdir, targetname)

    
    # USE_FP16 = True
    # May need to shut down all kernels and restart before this - otherwise you might get cuDNN initialization errors:
    #if USE_FP16:
    #    os.system("trtexec --onnx=resnet50_onnx_model.onnx --saveEngine=resnet_engine.trt  --explicitBatch --fp16")
    #else:
    #    os.system("trtexec --onnx=resnet50_onnx_model.onnx --saveEngine=resnet_engine.trt  --explicitBatch")


    # engine = build_engine(fixedfile)

    succeeded = build_tensorrt_engine(fixedfile,
                            outfile,
                            precision_mode='fp16',
                            max_workspace_size=GB(1),  # in bytes
                            max_batch_size=1,
                            min_timing_iterations=2,
                            avg_timing_iterations=2,
                            int8_calibrator=None)

    if s3.PutFile(s3def['sets']['model']['bucket'], outfile, objpath):
        shutil.rmtree(args.workdir, ignore_errors=True) 

    # trtcmd = "trtexec --onnx=/store/dmp/cl/store/mllib/model/2021-02-19-20-51-59-cocoseg/model.onnx --saveEngine=/store/dmp/cl/store/mllib/model/2021-02-19-20-51-59-cocoseg/model.trt  --explicitBatch --workspace=4096 --verbose  2>&1 | tee trtexe.log"
    print('onnx-trt complete')
    return failed

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