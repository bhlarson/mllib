import io
import os
import time
import picamera
import picamera.array
from base_camera import BaseCamera

import time

import cv2
from time import sleep
import datetime

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO

from utils import ops as utils_ops
from utils import label_map_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(tensor_dict, image, graph):
  # Run inference
  output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.int64)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


class Camera(BaseCamera):
    @staticmethod
    def frames():
        with picamera.PiCamera() as camera:
            #camera.vflip = True
            #camera.hflip = True
            resX = 768
            resY = 512
            image_np = np.empty((resY, resX, 3), dtype=np.uint8)
            camera.resolution = (resX, resY)

            with   detection_graph.as_default():
              with tf.compat.v1.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.compat.v1.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                  tensor_name = key + ':0'
                  if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name)
                if 'detection_masks' in tensor_dict:
                  # The following processing is only for single image
                  detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                  detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                  # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                  real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                  detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                  detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                  detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                      detection_masks, detection_boxes, image.shape[1], image.shape[2])
                  detection_masks_reframed = tf.cast(
                      tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                  # Follow the convention by adding back the batch dimension
                  tensor_dict['detection_masks'] = tf.expand_dims(
                      detection_masks_reframed, 0)
                image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')              

                # let camera warm up
                # time.sleep(2) camera startup duirng Tensorflow startup

                ''' Works, latency
                stream = io.BytesIO()
                for _ in camera.capture_continuous(stream, 'jpeg',use_video_port=True):
                    # return current frame
                    stream.seek(0)
                    yield stream.read()
                    # reset stream for next frame
                    stream.seek(0)
                    stream.truncate()
                '''
                
                ''' works, latency
                while True:
                    camera.capture(image_np, format='bgr')
                    yield cv2.imencode('.jpg', image_np)[1].tobytes()
                '''

                # 309 MS
                stream = io.BytesIO()
                for _ in camera.capture_continuous(stream, 'bgr',use_video_port=True):
                    # return current frame
                    stream.seek(0)

                    image_np = np.frombuffer(stream.read(), dtype=np.uint8, count=resX*resY*3).reshape((resY, resX,3))

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    #output_dict = run_inference_for_single_image(tensor_dict, image_np_expanded, detection_graph)
                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.int64)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        output_dict['detection_masks'] = output_dict['detection_masks'][0]

                    # Visualization of the results of a detection.
                    for i in range(output_dict['num_detections']):
                        if output_dict['detection_scores'][i] > 0.5:
                            pt1 = (int(output_dict['detection_boxes'][i][1]*image_np.shape[1]), int(output_dict['detection_boxes'][i][0]*image_np.shape[0]))
                            pt2 = (int(output_dict['detection_boxes'][i][3]*image_np.shape[1]), int(output_dict['detection_boxes'][i][2]*image_np.shape[0]))
                            cv2.rectangle(image_np,pt1,pt2,(0,255,0),3)
                            classTxt = '{} ({:2})'.format(category_index[output_dict['detection_classes'][i]]['name'], output_dict['detection_scores'][i])
                            cv2.putText(image_np,classTxt,pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))


                    yield cv2.imencode('.jpg', image_np)[1].tobytes()
                    # reset stream for next frame
                    stream.seek(0)
                    stream.truncate()

