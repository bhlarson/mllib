#!/usr/bin/env python
import os, sys, json, argparse
import cv2
import numpy as np
from flask import Flask, render_template, Response
from camera_opencv import Camera
import tflite_runtime.interpreter as tflite
import platform
from datetime import datetime
sys.path.insert(0, os.path.abspath(''))

parser = argparse.ArgumentParser()

parser.add_argument('-debug', action='store_true',help='Wait for debugge attach')
parser.add_argument('-model', 
                    type=str, default='./tflite/2020-09-07-20-23-20-dl3.tflite',
                    #type=str, default='./etpu/2020-05-28-02-16-08-lit_int8_edgetpu.tflite',
                    help='Path to model')
parser.add_argument('-description', 
                    type=str, default='./tflite/description.json',
                    help='Path to segmentation description')
parser.add_argument('-record_dir', type=str, default='./record', help='Path training set tfrecord')
parser.add_argument('-devices', type=json.loads, default=["/gpu:0"],  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')
parser.add_argument('-image_size', type=json.loads, default='[480, 512]', help='Training crop size [height, width]/  [90, 160],[120, 160],[120, 160], [144, 176],[288, 352], [240, 432],[480, 640],[576,1024],[720, 960], [720,1280],[1080, 1920]')
parser.add_argument('-image_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB') 

FLAGS, unparsed = parser.parse_known_args()

config = {
      'input_shape': [FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_depth],
      'area_filter_min': 250,
      'size_divisible': 32,
      }

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

_HEIGHT = 480
_WIDTH = 512
_DEPTH = 3

app = Flask(__name__)

model = None 
infer = None
lut = None

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def PadValid(image, divisible=config['size_divisible']):
    [height, width, depth] = image.shape
    newwidth = int(divisible*math.ceil(float(width)/divisible))
    newheight = int(divisible*math.ceil(float(height)/divisible))
    pad = [[0,newheight-height],[0,newwidth-width], [0,0]]
    if np.max(pad) > 0:
        image = np.pad(image, pad)
    return image
    #return tf.image.pad_to_bounding_box(image, 0, 0, height, width)

def CropOrigonal(image, height, width):
    return image[:height,:width,:]
    #return tf.image.crop_to_bounding_box(image, 0, 0, height, width


def gen(camera):
    """Video streaming generator function."""
    # Load TFLite model and allocate tensors.
    #interpreter = tflite.Interpreter(FLAGS.model, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter = tflite.Interpreter(FLAGS.model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    while True:
        img = camera.get_frame()
        #img = cv2.flip(img, +1)

        imgShape = img.shape
        #Define crop of 2x CNN size and downsize it in tf.image.crop_and_resize
        color =  (0,255,0)
        thickness =  3
        center = np.array([imgShape[1]/2, imgShape[0]/2])
        d =  np.array([_HEIGHT/2,_WIDTH/2])
        p1 = tuple((center-d).astype(int))
        p1 = (max(p1[0],0),max(p1[1],0))
        p2 = tuple((center+d).astype(int))
        p2 = (min(p2[0],imgShape[0]-1),min(p2[1],imgShape[1]-1))
        cv2.rectangle(img,p1,p2,color,thickness)
        crop = cv2.resize(img[p1[1]:p2[1], p1[0]:p2[0]],(_WIDTH,_HEIGHT))

        before = datetime.now()
        interpreter.set_tensor(input_details[0]['index'], [crop.astype(np.float32)])
        interpreter.invoke()
        logits = output_details[0]['quantization'][0]*interpreter.get_tensor(output_details[0]['index'])[0]
        seg = np.argmax(logits, axis=-1).astype(np.uint8)

        seg = [cv2.LUT(seg, lut[:, i]) for i in range(3)]
        seg = np.dstack(seg) 
        imseg = (crop*seg).astype(np.uint8)

        dt = datetime.now()-before

        resultsDisplay = '{:.3f}s'.format(dt.total_seconds())

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imseg, resultsDisplay, (10,25), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # encode as a jpeg image and return it
        frame = cv2.imencode('.jpg', imseg)[1].tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera(config)), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.debug:
        # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        # Launch applicaiton on remote computer: 
        # > python3 -m ptvsd --host 0.0.0.0 --port 3000 --wait predict_imdb.py
        import ptvsd
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=('0.0.0.0', 3000), redirect_output=True)
        # Pause the program until a remote debugger is attached
        print("Wait for debugger attach")
        ptvsd.wait_for_attach()
        print("Debugger Attached")

    trainingsetDescriptionFile = '{}'.format(FLAGS.description)
    trainingsetDescription = json.load(open(trainingsetDescriptionFile))['config']['trainingset']

    config['ignore_label'] = trainingsetDescription['classes']['ignore']
    config['classes'] = trainingsetDescription['classes']['classes']
    config['background'] = trainingsetDescription['classes']['background']
    config['trainingset'] = trainingsetDescription

    lut = np.zeros([256,3], dtype=np.uint8)
    for obj in config['trainingset']['classes']['objects']: # Load RGB colors as BGR
        lut[obj['trainId']][0] = obj['color'][2]
        lut[obj['trainId']][1] = obj['color'][1]
        lut[obj['trainId']][2] = obj['color'][0]
    lut = lut.astype(np.float) * 1/255. # scale colors 0-1
    lut[config['background']] = [1.0,1.0,1.0] # Pass Through
    app.run(host='0.0.0.0', threaded=True)
