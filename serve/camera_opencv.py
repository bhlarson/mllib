import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0
    config = None
    def __init__(self, config):
        Camera.config = config
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        for i in range(Camera.video_source,Camera.video_source+4):
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                break
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
        
        if Camera.config is not None:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,Camera.config['input_shape'][1])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT,Camera.config['input_shape'][0])
        while True:
            # read current frame
            _, img = camera.read()

            yield img
