import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0

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

        camera.set(cv2.CAP_PROP_FRAME_WIDTH,1024)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,576)
        while True:
            # read current frame
            _, img = camera.read()

            yield img
