import time
import picamera
import picamera.array
import cv2
from time import sleep
from fractions import Fraction

with picamera.PiCamera() as camera:
    camera.resolution = (768, 512)
    camera.exposure_mode = 'auto'
    camera.exposure_compensation = 0
    camera.start_preview()
    # Set a framerate of 1/6fps, then set shutter
    # speed to 6s and ISO to 800
    #camera.framerate = 1
    #camera.shutter_speed = 1000000
    #camera.exposure_mode = 'off'
    camera.iso = 800
    # Give the camera a good long time to measure AWB
    # (you may wish to use fixed AWB instead)
    sleep(2)
        
    with picamera.array.PiRGBArray(camera) as stream:

        
        camera.capture(stream, format='bgr')
        print('exposure_speed={}'.format(camera.exposure_speed))
        # At this point the image is available as stream.array
        img = stream.array
        cv2.imwrite('cam.png',img)