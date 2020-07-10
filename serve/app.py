#!/usr/bin/env python
import os, sys, json, argparse
import cv2
import numpy as np
from flask import Flask, render_template, Response
from camera_opencv import Camera
import tensorflow as tf
from datetime import datetime
sys.path.insert(0, os.path.abspath(''))
from segment.display import CreateIman

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true',help='Wait for debugge attach')

#parser.add_argument('--model_dir', type=str, default='./trainings/unet',help='Directory to store training model')
parser.add_argument('--model_dir', type=str, default=None,help='Directory to store training model')
parser.add_argument('--loadsavedmodel', type=str, default='./saved_model/2020-07-09-19-39-37-dl3', help='Saved model to load if no checkpoint')

parser.add_argument('--record_dir', type=str, default='./record', help='Path training set tfrecord')
parser.add_argument("--devices", type=json.loads, default=["/gpu:0"],  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')
parser.add_argument('--image_size', type=json.loads, default='[576,1024]', help='Training crop size [height, width]/  [288, 352],[480, 640],[576,1024],[720, 960],[1080, 1920]')
parser.add_argument('--image_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB') 

FLAGS, unparsed = parser.parse_known_args()

trainingsetDescriptionFile = '{}/description.json'.format(FLAGS.record_dir)
trainingsetDescription = json.load(open(trainingsetDescriptionFile))

config = {
      'input_shape': [FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_depth],
      'ignore_label': trainingsetDescription['classes']['ignore'],
      'classes': trainingsetDescription['classes']['classes'],
      'trainingset': trainingsetDescription,
      'area_filter_min': 250,
      }

app = Flask(__name__)

model = None 
infer = None

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""

    #loaded = tf.saved_model.load_v2(FLAGS.model)
    #loaded = tf.saved_model.load(FLAGS.savedmodel)
    #print(list(loaded.signatures.keys()))
    #infer = loaded.signatures["serving_default"]
    #print(infer.structured_outputs)
    #print (infer.inputs[0])

    while True:
        img = camera.get_frame()
        #print('img.shape={}'.format(img.shape))
        #img = cv2.flip(img, +1)

        tbefore = datetime.now()
        if model is not None:
            ann = model.predict(np.expand_dims(img, axis=0))
            ann = tf.squeeze(ann) # Drop batch dimension
        elif infer is not None:
            outputs = infer(tf.constant(np.expand_dims(img.astype(np.float32), axis=0)))
            ann = tf.squeeze(outputs['conv2d_transpose_4'].numpy()) # Drop batch dimension

        tPredict = datetime.now()
        iman = CreateIman(img, ann, config)

        tAfter = datetime.now()
        dInfer = tPredict-tbefore
        dImAn = tAfter-tPredict

        #outputs['pred_age'].numpy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        resultsDisplay = 'infer:{:.3f}s display:{:.3f}s'.format(dInfer.total_seconds(), dImAn.total_seconds())
        cv2.putText(iman, resultsDisplay, (10,25), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # encode as a jpeg image and return it
        frame = cv2.imencode('.jpg', iman)[1].tobytes()

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

    print(tf.version)

    if FLAGS.model_dir is not None:
        try:
            model = tf.keras.models.load_model(FLAGS.model_dir) # Load from checkpoint
        except:
            print('Unable to load weghts from {}'.format(FLAGS.model_dir))

    if not model and FLAGS.loadsavedmodel:
        try:
            # model = tf.keras.models.load_model(FLAGS.loadsavedmodel) # Load from checkpoint

            loaded = tf.saved_model.load(FLAGS.loadsavedmodel)
            print(list(loaded.signatures.keys()))
            infer = loaded.signatures["serving_default"]
            print(infer.structured_outputs)
            print (infer.inputs[0])
        except:
            print('Unable to load weghts from {}'.format(FLAGS.loadsavedmodel))

    app.run(host='0.0.0.0', threaded=True)
