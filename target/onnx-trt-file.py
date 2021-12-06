# based on https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#network_python

BATCH_SIZE = 32
import argparse
import os
import sys
import platform
import tempfile
import json
import numpy
import onnx
import tensorrt as trt

__all__ = ['MB', 'GB', 'build_tensorrt_engine']

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def MB(val):
    return val * 1 << 20


def GB(val):
    return val * 1 << 30

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Process arguments')

    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')

    parser.add_argument('-trtname', type=str, default="test/segment_nas_512x442_20211119_00.trt", help='Final model wiout extension')
    parser.add_argument('-onnxname', type=str, default="test/segment_nas_512x442_20211119_00.onnx", help='Onnx file name')
    parser.add_argument('-workspace_memory', type=int, default=8000, help='trtexec workspace size in megabytes')
    parser.add_argument('-batch_size', type=int, default=1, help='Number of examples per batch.') 
    # parser.add_argument('-image_size', type=json.loads, default='[480, 512]', help='Image size') 
    parser.add_argument('-precision',  type=str, default='fp32', choices=['int8', 'fp16', 'fp32'], help='Output model precision.')

    args = parser.parse_args()
    return args


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
                          precision_mode='fp32',
                          max_workspace_size=GB(1),  # in bytes
                          max_batch_size=1,
                          min_timing_iterations=2,
                          avg_timing_iterations=2,
                          int8_calibrator=None):
    """
    :param onnx_file_path:
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

    logger = trt.Logger(trt.Logger.VERBOSE)

    builder = trt.Builder(logger)
    if precision_mode == 'fp16':
        assert builder.platform_has_fast_fp16, 'platform does not support fp16 mode!'
    if precision_mode == 'int8':
        assert builder.platform_has_fast_int8, 'platform does not support int8 mode!'
        assert int8_calibrator is not None, 'calibrator is not provided!'

    #network = builder.create_network(EXPLICIT_BATCH)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, logger)

    '''with open(onnx_file_path, 'rb') as onnx_fin:
        parser.parse(onnx_fin.read())

    num_error = parser.num_errors
    if num_error != 0:
        for i in range(num_error):
            temp_error = parser.get_error(i)
            print(temp_error.desc())
        return'''

    success = parser.parse_from_file(onnx_file_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        return None

    config = builder.create_builder_config()

    if precision_mode == 'int8':
        config.int8_calibrator = int8_calibrator
        config.set_flag(trt.BuilderFlag.INT8)
    elif precision_mode == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        pass

    config.max_workspace_size = max_workspace_size
    #config.min_timing_iterations = min_timing_iterations
    #config.avg_timing_iterations = avg_timing_iterations
    #builder.max_batch_size = max_batch_size

    '''profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, 
        (args.batch_size, 3, args.image_size[0], args.image_size[1]), 
        (args.batch_size, 3, args.image_size[0], args.image_size[1]), 
        (args.batch_size, 3, args.image_size[0], args.image_size[1]))
    config.add_optimization_profile(profile)'''

    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except:
        print('Engine build unsuccessfully!')

    if serialized_engine is None:
        print('Engine build unsuccessfully!')

    return serialized_engine


def main(args):
    failed = 0

    config = args.__dict__
    config['platform'] = platform.platform()
    config['python'] = platform.python_version()
    config['onnx version'] =sys.modules['onnx'].__version__
    config['tensorrt version'] = sys.modules['tensorrt'].__version__
    config['numpy version'] = sys.modules['numpy'].__version__
    print('onnx-trt-file config={}'.format(config))

    onnx_model = onnx.load(args.onnxname)
    if onnx_model is None:
        print('Failed to load onnx model!')
        failed = -1
        return failed

    inputs = onnx_model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_value = args.batch_size

    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        onnx.save(onnx_model, tmp.name)

        serialized_engine = build_tensorrt_engine(tmp.name,
            precision_mode=args.precision,
            max_workspace_size=MB(args.workspace_memory),  # in bytes
            max_batch_size=args.batch_size,
            min_timing_iterations=2,
            avg_timing_iterations=2,
            int8_calibrator=None)


        with open(args.trtname, 'wb') as file:
            file.write(serialized_engine)

    print('onnx-trt-file convert {} to {}  return: {}'.format(args.onnxname, args.trtname, failed))
    return failed


if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy

        debugpy.listen(address=(args.debug_address, args.debug_port))
        # Pause the program until a remote debugger is attached

        debugpy.wait_for_client()
        print("Debugger attached")

    result = main(args)
    sys.exit(result)
