
import os
import sys
import argparse
import platform
import tensorrt as trt
from PIL import Image

sys.path.insert(0, os.path.abspath(''))
import engine as eng
import inference as inf

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-trtmodel', type=str, default='./saved_model/2020-09-28-14-52-11-dl3/trt-480-640.plan', help='Path to TensorRT.')

def main(args):

    print('Platform: {}'.format(platform.platform()))

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    images = ['000000001761.jpg', '000000119088.jpg', '000000139099.jpg', '000000143998.jpg', '000000222235.jpg', '000000276707.jpg', '000000386134.jpg', '000000428218.jpg', '000000530854.jpg', '000000538067.jpg']


    engine = eng.load_engine(trt_runtime, args.trtmodel)
    input_shape = engine.get_binding_shape(0)
    h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float16)


    '''out = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
    out = color_map(out)

    colorImage_trt = Image.fromarray(out.astype(np.uint8))
    colorImage_trt.save(“trt_output.png”)

    semantic_model = keras.models.load_model('/path/to/semantic_segmentation.hdf5')
    out_keras= semantic_model.predict(im.reshape(-1, 3, HEIGHT, WIDTH))

    out_keras = color_map(out_keras)
    colorImage_k = Image.fromarray(out_keras.astype(np.uint8))
    colorImage_k.save(“keras_output.png”)'''


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