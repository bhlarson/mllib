#!/usr/bin/python3

import math
import random

import numpy as np
import tensorflow as tf

from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance

class Augmentations:
    def __init__(self, format_spec):
        self.format_spec = format_spec

    ##
    # Create a PIL image from a tensorflow image.
    ##
    def load_image(self, image):
        if self.format_spec.channels == 1:
            return Image.fromarray(np.squeeze(image))
        return Image.fromarray(image)

    ##
    # Save a PIL image back to a tensorflow image.
    ##
    def export_image(self, pil_image, image):
        if self.format_spec.channels == 1:
            np.copyto(image, np.expand_dims(np.array(pil_image), axis=2))
        else:
            np.copyto(image, np.array(pil_image))

    ##
    # Flip the image horizontally.
    ##
    def flip_horizontal(self, image):
        return tf.image.random_flip_left_right(image)

    ##
    # Make a few random color adjustments to the image like changing the contrast, brightness
    # saturation etc.
    ##
    def adjust_color(self, image):
        image = tf.image.random_brightness(image, max_delta=0.25)
        image = tf.image.random_contrast(image, lower=0.80, upper=1)
        image = tf.image.random_hue(image, max_delta=0.25)
        image = tf.image.random_saturation(image, lower=0.25, upper=1)
        return image

    ##
    # Translate the image.
    ##
    def translate(self, image, tx_max, ty_max):
        tx = tf.random_uniform([1], minval=(-1 * tx_max), maxval=tx_max, dtype=tf.float32)
        ty = tf.random_uniform([1], minval=(-1 * ty_max), maxval=ty_max, dtype=tf.float32)
        return tf.contrib.image.translate(image, tf.concat([tx, ty], 0))

    ##
    # Rotate the image.
    ##
    def rotate(self, image, rotate_max_degrees):
        rotate_degrees = tf.random_uniform([1], minval=math.radians(-1 * rotate_max_degrees),
                                           maxval=math.radians(rotate_max_degrees))
        return tf.contrib.image.rotate(image, rotate_degrees)

    def randcolor(self):
        if self.format_spec.channels == 1:
            return random.randint(0, 255)
        else:
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    ##
    # Add random noise to the image.
    # Unused - Current implementation starves gpu.
    ##
    def add_noise(self, image):
        def process(image):
            pil_image = self.load_image(image)
            w, h = pil_image.size
            num_px = int(w * h * random.uniform(0, 0.1))
            for i in range(num_px):
                x = random.randint(0, w - 1)
                y = random.randint(0, h -1)
                c = self.randcolor()
                pil_image.putpixel((x, y), c)
            self.export_image(pil_image, image)
            return image

        original_shape = image.get_shape().as_list()
        return tf.reshape(tf.py_func(process, [image], tf.uint8, stateful=False), original_shape)

    ##
    # Cut out parts of the image and replace them with random colored boxes.
    # Unused - current implementation starves gpu.
    ##
    def add_cutouts(self, image, cutouts):
        def process(image):
            pil_image = self.load_image(image)
            w, h = pil_image.size
            for cutout in cutouts:
                canvas = ImageDraw.Draw(pil_image)
                x = random.randint(0, w - cutout - 1)
                y = random.randint(0, h - cutout - 1)
                canvas.rectangle([(x, y), (x + cutout, y + cutout)], fill=self.randcolor())
                del canvas

            self.export_image(pil_image, image)
            return image

        original_shape = image.get_shape().as_list()
        return tf.reshape(tf.py_func(process, [image], tf.uint8, stateful=False), original_shape)

##
# Perform GPU dataset augmentations.
##
def augment(images, params):
    
    if "flip_horizontal" in params and params.flip_horizontal:
        images = tf.image.random_flip_left_right(images)
    if "flip_vertical" in params and params.flip_vertical:
        images = tf.image.random_flip_up_down(images)

    if "tx_max" in params and "ty_max" in params and (params.tx_max > 0 or params.ty_max > 0):
        tx = tf.random_uniform([1], minval=(-1 * params.tx_max), maxval=params.tx_max, dtype=tf.float32)
        ty = tf.random_uniform([1], minval=(-1 * params.ty_max), maxval=params.ty_max, dtype=tf.float32)
        images = tf.contrib.image.translate(images, tf.concat([tx, ty], 0))

    if "rotate_max_degrees" in params and params.rotate_max_degrees>0:
        rotate_degrees = tf.random_uniform([1], minval=math.radians(-1 * params.rotate_max_degrees),
                                           maxval=math.radians(params.rotate_max_degrees))
        images = tf.contrib.image.rotate(images, rotate_degrees)
    
    if "max_dBrightness" in params and params.max_dBrightness > 0
        images = tf.image.random_brightness(images, params.max_dBrightness=0.25)

    if "max_contrast" in params and "min_contrast" in params
        images =tf.image.random_contrast(images, lower=params.min_contrast, upper=params.max_contrast)

    if "max_dhue" in params and params.max_dhue > 0
        images = tf.image.random_hue(images, max_delta=params.max_dhue)

    if "max_saturation" in params and "min_saturation" in params
        images = tf.image.random_saturation(images, lower=params.min_saturation, upper=params.max_saturation)

    return images