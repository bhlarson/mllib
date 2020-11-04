
import os
import sys
import argparse
import platform
import json
import numpy as np
import tensorrt as trt
import cv2

sys.path.insert(0, os.path.abspath(''))
import engine as eng
import inference as inf

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-trtmodel', type=str, default='./saved_model/2020-11-04-05-45-02/classify.trt', help='Path to TensorRT.')
parser.add_argument('-image_size', type=json.loads, default='[1, 224, 224, 3]', help='Training size [batch, height, width, colors]')


def crop(image, out_size):
    in_size = image.shape
    out_size = list(out_size)
    h_diff = in_size[0] - out_size[0]
    w_diff = in_size[1] - out_size[1]
    assert h_diff >= 0 or w_diff >= 0, 'At least one side must be longer than or equal to the output size'

    if h_diff > 0 and w_diff > 0:
        h_idx = h_diff//2
        w_idx = w_diff//2
        image = image[h_idx:h_idx + out_size[0], w_idx:w_idx + out_size[1]]
    elif h_diff > 0:
        h_idx = h_diff//2
        image = image[h_idx:h_idx + out_size[0], :]
    elif w_diff > 0:
        w_idx = w_diff//2
        image = image[:, w_idx:w_idx + out_size[1]]

    return image

def zero_pad(image, out_size):
    in_size = image.shape
    out_size = list(out_size)
    h_diff = out_size[0] - in_size[0]
    w_diff = out_size[1] - in_size[1]
    assert h_diff >= 0 or w_diff >= 0, 'At least one side must be shorter than or equal to the output size'

    out_size_max = [max(out_size[0], in_size[0]), max(out_size[1], in_size[1])]
    image_out = np.zeros(out_size_max, dtype=image.dtype)

    if h_diff > 0 and w_diff > 0:
        h_idx = h_diff//2
        w_idx = w_diff//2
        image_out[h_idx:h_idx + in_size[0], w_idx:w_idx + in_size[1]] = image
    elif h_diff > 0:
        h_idx = h_diff//2
        image_out[h_idx:h_idx + in_size[0], :] = image
    elif w_diff > 0:
        w_idx = w_diff//2
        image_out[:, w_idx:w_idx + in_size[1]] = image
    else:
        image_out = image

    return image_out

def resize_with_crop_or_pad(image, out_size):
    if image.shape[0] > out_size[0] or image.shape[1] > out_size[1]:
        image = crop(image, out_size)
    if image.shape[0] < out_size[0] or image.shape[1] < out_size[1]:
        image = zero_pad(image, out_size)

    return image

def main(args):

    print('Platform: {}'.format(platform.platform()))

    trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    f = open(args.trtmodel, "rb")
    engine = trt_runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)

    print('input shape: {}'.format(input_shape))
    print('output shape: {}'.format(output_shape)) 

    images = ['testtrt/000000001761.jpg', 'testtrt/000000119088.jpg', 'testtrt/000000139099.jpg', 'testtrt/000000143998.jpg', 'testtrt/000000222235.jpg', 'testtrt/000000276707.jpg', 'testtrt/000000386134.jpg', 'testtrt/000000428218.jpg', 'testtrt/000000530854.jpg', 'testtrt/000000538067.jpg']

    engine = eng.load_engine(trt_runtime, args.trtmodel)
    print('input shape: {}'.format(engine.get_binding_shape(0)))
    print('output shape: {}'.format(engine.get_binding_shape(1))) 
    output_shape = engine.get_binding_shape(1)
    h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)

    for image in images:
        img = cv2.imread(image,0)
        img = resize_with_crop_or_pad(img, [input_shape[1],input_shape[2]])
        out = inf.do_inference(engine, img.astype(np.uint16), h_input, d_input, h_output, d_output, stream, 1, input_shape[1], input_shape[2])

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