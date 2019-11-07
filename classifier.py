#!/usr/bin/python3
# from https://miramar.io/ep1/s1e3.html

import argparse
import augmentations
import classifier_spec_pb2
import glob
import random
import os

import tensorflow as tf
import tflite
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageColor as ImageColor
import google.protobuf.text_format as text_format

NUM_INPUT_THREADS = 6
VALIDATION_REPEATS = -1 # Forever.
SHUFFLE_BUFFER_SIZE = 512
PREFETCH_BUFFER_SIZE = SHUFFLE_BUFFER_SIZE * 4

R"""
python classifier.py --classifier_spec=./classifier_spec.pbtxt [--export_only]
"""
class Classifier:
    def __init__(self, classifier_spec_path, export_only):
        self.classifier_spec = self.load_classifier_spec(classifier_spec_path)
        self.export_only = export_only
        self.font = ImageFont.load_default()
        self.label_map = self.load_label_map(self.classifier_spec.label_map_path)
        self.format_spec = self.load_format_spec(self.classifier_spec.format_spec_path)
        self.num_classes = len(self.label_map)
        self.aug = augmentations.Augmentations(self.format_spec)
        print("Loaded classifier_spec=%s format_spec=%s num_classes=%d" %
              (self.classifier_spec, self.format_spec, self.num_classes))

    ##
    # Load up the classifier spec.
    ##
    def load_classifier_spec(self, classifier_spec_path):
        with open(classifier_spec_path, 'r') as f:
            proto_str = f.read()
            classifier_spec = classifier_spec_pb2.ClassifierSpec()
            text_format.Merge(proto_str, classifier_spec)
            return classifier_spec

    ##
    # Load up a label map proto into a dict.
    ##
    def load_label_map(self, label_map_path):
        label_map = {}
        with open(label_map_path, 'r') as f:
            proto_str = f.read()
            label_map_proto = classifier_spec_pb2.LabelMap()
            text_format.Merge(proto_str, label_map_proto)
            for item in label_map_proto.item:
                label_map[item.id] = item.name;
        return label_map

    ##
    # Load up a format spec.
    ##
    def load_format_spec(self, format_spec_path):
        with open(format_spec_path, 'r') as f:
            proto_str = f.read()
            format_spec = classifier_spec_pb2.FormatSpec()
            text_format.Merge(proto_str, format_spec)
            return format_spec

    ##
    # Perform dataset augmentations.
    ##
    def augment(self, images):
        if self.classifier_spec.augmentations.flip_horizontal:
            images = self.aug.flip_horizontal(images)

        if self.classifier_spec.augmentations.color_adjustment:
            images = self.aug.adjust_color(images)

        if (self.classifier_spec.augmentations.translate_vertical_max_px > 0 or
                self.classifier_spec.augmentations.translate_horizontal_max_px > 0):
            images = self.aug.translate(images,
                        self.classifier_spec.augmentations.translate_horizontal_max_px,
                        self.classifier_spec.augmentations.translate_vertical_max_px)

        if self.classifier_spec.augmentations.rotate_max_degrees > 0:
            images = self.aug.rotate(images, self.classifier_spec.augmentations.rotate_max_degrees)

        return images

    ##
    # Sets up the dataset to read from.
    ##
    def make_dataset_input_fn(self, path, epochs):
        dataset = tf.data.TFRecordDataset([path])

        def parser(record):
            label_key = "image/label"
            bytes_key = "image/encoded"
            parsed = tf.parse_single_example(record, {
                bytes_key : tf.FixedLenFeature([], tf.string),
                label_key : tf.FixedLenFeature([], tf.int64),
            })
            image = tf.decode_raw(parsed[bytes_key], tf.uint8)
            dims = [self.format_spec.width, self.format_spec.height, self.format_spec.channels]
            image = tf.reshape(image, dims)
            return { "image" : image }, parsed[label_key]

        dataset = dataset.map(parser, num_parallel_calls=NUM_INPUT_THREADS)
        dataset = dataset.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.batch(self.classifier_spec.batch_size)
        if epochs > 0:
            dataset = dataset.repeat(epochs)
        else:
            dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    ##
    # An input function for training data.
    ##
    def train_input_fn(self):
        return self.make_dataset_input_fn(self.classifier_spec.train_path,
                                          self.classifier_spec.num_epochs)

    ##
    # An input function for validation data.
    ##
    def eval_input_fn(self):
        return self.make_dataset_input_fn(self.classifier_spec.eval_path,
                                          self.classifier_spec.num_epochs)

    ##
    # Input function that is used by the model server.
    ##
    def serving_input_fn(self):
        feature_placeholders = {
            "image" : tf.placeholder(tf.string, [])
        }
        images = tf.decode_base64(feature_placeholders["image"])
        images = tf.image.decode_jpeg(images, channels=self.format_spec.channels)
        images = tf.expand_dims(images, 0)
        images = tf.image.resize_images(images, [self.format_spec.width, self.format_spec.height])
        features = {
            "image" : images
        }
        return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


    ##
    # Create a copy of the input images with the labels drawn on them.
    ##
    def export_annotated_images(self, images_uint8, labels, predictions, name):
        def draw_image_with_label(images, labels, predictions):
            for i, image in enumerate(images):
                if self.format_spec.channels == 1:
                    pil_image = Image.fromarray(np.squeeze(image))
                else:
                    pil_image = Image.fromarray(image)

                canvas = ImageDraw.Draw(pil_image)
                caption = self.label_map[labels[i]] + "/" + self.label_map[predictions[i]]
                canvas.text((0, 0), text=caption, font=self.font, fill=(255, 0, 0))
                del canvas

                if self.format_spec.channels == 1:
                    np.copyto(image, np.expand_dims(np.array(pil_image), axis=2))
                else:
                    np.copyto(image, np.array(pil_image))
            return images

        images_annotated = tf.py_func(draw_image_with_label,
                [images_uint8, labels, predictions], tf.uint8, stateful=False)
        tf.summary.image(name, images_annotated, max_outputs=4)

    ##
    # Return the channels axis based on the data format.
    ##
    def batchnorm_axis(self):
        if self.classifier_spec.data_format == "NHWC":
            return 3
        return 1

    ##
    # Creates a resnet50 network.
    # https://arxiv.org/pdf/1512.03385.pdf
    ##
    def make_resnet50(self, input_layer, mode):
        data_format = self.classifier_spec.data_format
        is_training = tf.constant(mode == tf.contrib.learn.ModeKeys.TRAIN)
        axis = self.batchnorm_axis()

        # Fixed padding adds in padding based only on the kernel size. This is how cudnn and caffe
        # do it. Tensorflow has a different padding scheme that involves the size of the input.
        def pad(input_layer, kernel_size):
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            if self.classifier_spec.data_format == "NHWC":
                padded = tf.pad(input_layer, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            else:
                padded = tf.pad(input_layer, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
            return padded

        # Conv2d with fancy resnet padding.
        def conv2d(input_layer, filters, kernel_size, stride):
            print("    conv2d input=%s filters=%d, kernel_size=%d stride=%d"
                    % (input_layer.get_shape().as_list(), filters, kernel_size, stride))
            prev = input_layer
            if (stride > 1):
                prev = pad(input_layer, kernel_size)
            prev = tf.contrib.layers.conv2d(inputs=prev, num_outputs=filters,
                    kernel_size=kernel_size, stride=stride,
                    padding=('SAME' if stride == 1 else 'VALID'),
                    activation_fn=None, data_format=data_format)
            return prev

        # A single bottleneck block of resnet50.
        def bottleneck_block(input_layer, filters, stride, projection_shortcut):
            # A single resnet50 bottleneck block looks like this:
            #
            #            |
            #            +------------------------+
            #            |                        |
            #    1x1 conv f filters               |
            #        batch norm                   |
            #          relu                       |
            #            |                        |
            #    3x3 conv f filters               |
            #        batch norm          [1x1 conv, 4f filters] <- projection shortcut
            #          relu                  [batch norm]
            #            |                        |
            #    1x1 conv 4f filters              |
            #        batch norm                   |
            #          relu                       |
            #            |                        |
            #           add-----------------------+
            #            |
            #          relu
            #            |
            prev = input_layer

            if projection_shortcut:
                shortcut = conv2d(input_layer, filters=(filters * 4), kernel_size=1, stride=stride)
                shortcut = tf.layers.batch_normalization(inputs=shortcut, training=is_training,
                                                         axis=axis, fused=False)
            else:
                shortcut = input_layer

            prev = conv2d(input_layer=prev, filters=filters, kernel_size=1, stride=1)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training,
                                                 axis=axis, fused=False)
            prev = tf.nn.relu(prev)

            prev = conv2d(input_layer=prev, filters=filters, kernel_size=3, stride=stride)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training,
                                                 axis=axis, fused=False)
            prev = tf.nn.relu(prev)

            prev = conv2d(input_layer=prev, filters=(4 * filters), kernel_size=1, stride=1)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training,
                                                 axis=axis, fused=False)
            prev = tf.nn.relu(prev)

            prev = tf.add(prev, shortcut)
            return tf.nn.relu(prev)

        # A group of bottlneck blocks.
        def resnet_block(input_layer, filters, stride, num_bottlenecks):
            print("  resnet block input=%s filters=%d, stride=%d, num_bottlenecks=%d"
                    % (input_layer.get_shape().as_list(), filters, stride, num_bottlenecks))
            prev = bottleneck_block(input_layer, filters=filters, stride=stride, projection_shortcut=True)
            for i in range(1, num_bottlenecks):
                prev = bottleneck_block(prev, filters=filters, stride=1, projection_shortcut=False)
            return prev

        # Resnet50 has the following blocks.
        # conv 7x7x64 / s2
        # batch norm
        # relu
        # maxpool 3x3 / s2
        # -------------------------
        # 3 x  64 filters, stride 1 [Block 1]
        # 4 x 128 filters, stride 2 [Block 2]
        # 6 x 256 filters, stride 2 [Block 3]
        # 3 x 512 filters, stride 2 [Block 4]
        # -------------------------
        # average pool 7x7 stride 1
        # fc 2048
        prev = conv2d(input_layer, filters=64, kernel_size=7, stride=2)
        prev = tf.layers.batch_normalization(inputs=prev, training=is_training, axis=axis, fused=False)
        prev = tf.nn.relu(prev)
        prev = tf.contrib.layers.max_pool2d(inputs=prev, kernel_size=3, stride=2, padding='SAME')

        print("Block 1 input=%s" % prev.get_shape().as_list())
        prev = resnet_block(prev, filters=64,  stride=1, num_bottlenecks=3)
        print("Block 2 input=%s" % prev.get_shape().as_list())
        prev = resnet_block(prev, filters=128, stride=2, num_bottlenecks=4)
        print("Block 3 input=%s" % prev.get_shape().as_list())
        prev = resnet_block(prev, filters=256, stride=2, num_bottlenecks=6)
        print("Block 4 input=%s" % prev.get_shape().as_list())
        prev = resnet_block(prev, filters=512, stride=2, num_bottlenecks=3)

        print("avg pool input=%s" % prev.get_shape().as_list())
        prev = tf.contrib.layers.avg_pool2d(inputs=prev, kernel_size=[7, 7], stride=1)

        shape = prev.get_shape().as_list()
        prev = tf.reshape(prev, [-1, shape[1] * shape[2] * shape[3]]) # batch size, fc inputs.
        print("dense input=%s" % prev.get_shape().as_list())
        logits = tf.contrib.layers.fully_connected(inputs=prev, activation_fn=None, num_outputs=len(self.label_map))

        tf.contrib.layers.summarize_activation(logits)
        return logits


    ##
    # Makes the mobilenet v1 architecture.
    # Mobilenet: https://arxiv.org/pdf/1704.04861.pdf
    ##
    def make_mobilenet_v1(self, input_layer, mode):
        is_training = tf.constant(mode == tf.contrib.learn.ModeKeys.TRAIN)
        axis = self.batchnorm_axis()

        def conv_unit(inputs, num_outputs, kernel_size, stride):
            # Conv layers are 3x3 > BatchNorm > ReLu
            print("conv\t\t /s%s \t%s \t%s" % (stride, kernel_size, inputs.get_shape().as_list()))
            prev =  tf.contrib.layers.conv2d(inputs=inputs, num_outputs=num_outputs,
                                             kernel_size=kernel_size, stride=stride,
                                             activation_fn=None, data_format=self.classifier_spec.data_format)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training, axis=axis, fused=False)
            return tf.nn.relu(prev)

        def depthwise_conv_unit(inputs, num_outputs, stride, depth_multiplier=1):
            # Depthwise conv layers are 3x3 depthwise > BatchNorm > ReLu > 1x1 Conv > BatchNorm > ReLu
            print("depthwise_conv\t /s%s \t\t%s" % (stride, inputs.get_shape().as_list()))
            prev = tf.contrib.layers.separable_conv2d(inputs=inputs, depth_multiplier=depth_multiplier, kernel_size=[3, 3],
                                                      stride=stride, num_outputs=None, activation_fn=None,
                                                      data_format=self.classifier_spec.data_format)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training, axis=axis, fused=False)
            prev = tf.nn.relu(prev)
            return conv_unit(inputs=prev, num_outputs=num_outputs, kernel_size=[1, 1], stride=1)

        prev = input_layer
        # tf.nn.relu is the default activation function.
        #   Type / Stride         Filter Shape          Input Size
        #   =============       =================       ==============
        #   Conv / s2           3 × 3 × 3 × 32          224 × 224 × 3  #1
        #   Conv dw / s1        3 × 3 × 32 dw           112 × 112 × 32 #2
        #   Conv / s1           1 × 1 × 32 × 64         112 × 112 × 32
        #   Conv dw / s2        3 × 3 × 64 dw           112 × 112 × 64 #3
        #   Conv / s1           1 × 1 × 64 × 128        56 × 56 × 64
        #   Conv dw / s1        3 × 3 × 128 dw          56 × 56 × 128  #4
        #   Conv / s1           1 × 1 × 128 × 128       56 × 56 × 128
        #   Conv dw / s2        3 × 3 × 128 dw          56 × 56 × 128  #5
        #   Conv / s1           1 × 1 × 128 × 256       28 × 28 × 128
        #   Conv dw / s1        3 × 3 × 256 dw          28 × 28 × 256  #6
        #   Conv / s1           1 × 1 × 256 × 256       28 × 28 × 256
        #   Conv dw / s2        3 × 3 × 256 dw          28 × 28 × 256  #7
        #   Conv / s1           1 × 1 × 256 × 512       14 × 14 × 256
        #5× ---------------------------------------------------------- #8-12
        #   Conv dw / s1        3 × 3 × 512 dw          14 × 14 × 512
        #   Conv / s1           1 × 1 × 512 × 512       14 × 14 × 512
        #   ---------------------------------------------------------
        #
        #   Conv dw / s2        3 × 3 × 512 dw          14 × 14 × 512  #13
        #   Conv / s1           1 × 1 × 512 × 1024      7 × 7 × 512
        #   Conv dw / s1        3 × 3 × 1024 dw         7 × 7 × 1024   #14
        #   Conv / s1           1 × 1 × 1024 × 1024     7 × 7 × 1024
        #
        #   Avg Pool / s1       Pool 7 × 7              7 × 7 × 1024
        #   FC / s1             1024 × 1000             1 × 1 × 1024
        #   Softmax / s1        Classifier              1 × 1 × 1000
        prev = conv_unit(inputs=prev, num_outputs=32, kernel_size=[3, 3], stride=2) #1
        prev = depthwise_conv_unit(inputs=prev, num_outputs=64, stride=1)           #2
        prev = depthwise_conv_unit(inputs=prev, num_outputs=128, stride=2)          #3
        prev = depthwise_conv_unit(inputs=prev, num_outputs=128, stride=1)          #4
        prev = depthwise_conv_unit(inputs=prev, num_outputs=256, stride=2)          #5
        prev = depthwise_conv_unit(inputs=prev, num_outputs=256, stride=1)          #6
        prev = depthwise_conv_unit(inputs=prev, num_outputs=512, stride=2)          #7
        print("--")
        for i in range(0, 5):
            prev = depthwise_conv_unit(inputs=prev, num_outputs=512, stride=1)      #8-12
        print("--")
        prev = depthwise_conv_unit(inputs=prev, num_outputs=1024, stride=2)         #13
        prev = depthwise_conv_unit(inputs=prev, num_outputs=1024, stride=1)         #14

        prev = tf.contrib.layers.avg_pool2d(inputs=prev, kernel_size=[7, 7], stride=1)
        print(prev.get_shape().as_list())
        shape = prev.get_shape().as_list()
        prev = tf.reshape(prev, [-1, shape[1] * shape[2] * shape[3]]) # batch size, fc inputs.
        print(prev.get_shape().as_list())
        prev = tf.contrib.layers.fully_connected(inputs=prev, num_outputs=1024)
        print(prev.get_shape().as_list())
        logits = tf.contrib.layers.fully_connected(inputs=prev, activation_fn=None, num_outputs=len(self.label_map))
        tf.contrib.layers.summarize_activation(logits)

        return logits


    ##
    # Creates a mobilenet v2 network.
    # https://arxiv.org/pdf/1801.04381.pdf
    ##
    def make_mobilenet_v2(self, input_layer, mode):
        is_training = tf.constant(mode == tf.contrib.learn.ModeKeys.TRAIN)
        axis = self.batchnorm_axis()

        def conv_unit(inputs, num_outputs, kernel_size, stride, linear=False):
            print("  conv\t\t\t /s%s \t%s \t%s" % (stride, kernel_size, inputs.get_shape().as_list()))
            prev = tf.contrib.layers.conv2d(inputs=inputs, num_outputs=num_outputs,
                                            kernel_size=kernel_size, stride=stride,
                                            activation_fn=None,
                                            data_format=self.classifier_spec.data_format)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training, axis=axis, fused=False)
            if not linear:
                prev = tf.nn.relu6(prev)
            return prev

        def depthwise_conv_unit(inputs, num_outputs, stride, depth_multiplier=1):
            print("  depthwise_conv\t /s%s \t\t%s" % (stride, inputs.get_shape().as_list()))
            prev = tf.contrib.layers.separable_conv2d(inputs=inputs, depth_multiplier=depth_multiplier,
                                                      kernel_size=[3, 3], stride=stride,
                                                      num_outputs=None, activation_fn=None,
                                                      data_format=self.classifier_spec.data_format)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training, axis=axis, fused=False)
            return tf.nn.relu6(prev)

        def bottleneck_unit(inputs, num_outputs, stride, depth_multiplier=1):
            print(" bottleneck filters=%d, stride=%d" % (num_outputs, stride))
            # conv_1x1 relu6 > depthwise_3x3/s1 relu6 > conv 1x1 linear > add_input
            # conv_1x1 relu6 > depthwise_3x3/s2 relu6 > conv_1x1 linear
            prev = conv_unit(inputs, num_outputs=num_outputs, kernel_size=[1, 1], stride=1)
            prev = depthwise_conv_unit(prev, num_outputs=num_outputs, stride=stride, depth_multiplier=depth_multiplier)
            prev = conv_unit(prev, num_outputs=num_outputs, kernel_size=[1, 1], stride=1, linear=True)
            if stride == 1:
                shortcut = conv_unit(inputs, num_outputs=num_outputs, kernel_size=[1, 1], stride=1, linear=True)
                prev = tf.add(shortcut, prev)
            return prev

        def bottleneck_block(inputs, num_outputs, stride, repeats):
            # Each n-sequence has a stride-s unit followed n-1 stride-1 units.
            print("bottleneck filters=%d, stride=%d repeats=%d" % (num_outputs, stride, repeats))
            prev = bottleneck_unit(inputs, num_outputs, stride)
            for i in range(repeats - 1):
                prev = bottleneck_unit(prev, num_outputs, 1)
            return prev

        # Input             Operator        t       c       n       s
        # ==========        ==========     ===     ====    ===     ===
        # 224x224x3         conv2d 3x3      -       32      1       2
        # 112x112x32        bottleneck      1       16      1       1
        # 112x112x16        bottleneck      6       24      2       2
        # 56x56x24          bottleneck      6       32      3       2
        # 28x28x32          bottleneck      6       64      4       2
        # 14x14x64          bottleneck      6       96      3       1
        # 14x14x96          bottleneck      6       160     3       2
        # 7x7x160           bottleneck      6       320     1       1
        # 7x7x320           conv2d 1x1      -       1280    1       1
        # 7x7x1280          avgpool 7x7     -       -       1       -
        # 1x1x1280          conv2d 1x1      -       k       -

        prev = input_layer
        print("input")
        prev = conv_unit(prev, num_outputs=32, kernel_size=[3, 3], stride=2)
        prev = bottleneck_block(prev, num_outputs=16, stride=1, repeats=1)
        prev = bottleneck_block(prev, num_outputs=24, stride=2, repeats=2)
        prev = bottleneck_block(prev, num_outputs=32, stride=2, repeats=3)
        prev = bottleneck_block(prev, num_outputs=64, stride=2, repeats=4)
        prev = bottleneck_block(prev, num_outputs=96, stride=1, repeats=3)
        prev = bottleneck_block(prev, num_outputs=160, stride=2, repeats=3)
        prev = bottleneck_block(prev, num_outputs=320, stride=1, repeats=1)
        prev = conv_unit(prev, num_outputs=1280, kernel_size=[1, 1], stride=1)
        prev = tf.contrib.layers.avg_pool2d(inputs=prev, kernel_size=[7, 7], stride=1)

        print(prev.get_shape().as_list())
        shape = prev.get_shape().as_list()
        prev = tf.reshape(prev, [-1, shape[1] * shape[2] * shape[3]]) # batch size, fc inputs.
        print(prev.get_shape().as_list())
        logits = tf.contrib.layers.fully_connected(inputs=prev, activation_fn=None, num_outputs=len(self.label_map))
        tf.contrib.layers.summarize_activation(logits)
        return logits


    ##
    # This implements a simple convolutional neural network architecture.
    ##
    def model_fn(self, features, labels, mode):
        images_uint8 = tf.identity(features["image"], "whc_input")

        # Convert the range of the pixels from 0 - 255 to 0.0 - 1.0
        input_layer = tf.image.convert_image_dtype(images_uint8, tf.float32)

        if self.classifier_spec.data_format =="NHWC":
            prev = tf.transpose(input_layer, [0, 2, 1, 3], name="nhwc_input")
            # Perform dataset augmentation during training only (only supported on nhwc).
            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                prev = self.augment(prev)
        else:
            prev = tf.transpose(input_layer, [0, 3, 2, 1], name="nchw_input")

        # Shift the images to the range [-1.0, 1.0)
        prev = tf.subtract(prev, 0.5)
        prev = tf.multiply(prev, 2.0)

        # Add on the appropriate classifier graph.
        if self.classifier_spec.classifier == "mobilenet_v1":
            logits = self.make_mobilenet_v1(prev, mode)
        elif self.classifier_spec.classifier == "mobilenet_v2":
            logits = self.make_mobilenet_v2(prev, mode)
        elif self.classifier_spec.classifier == "resnet_50":
            logits = self.make_resnet50(prev, mode)
        else:
            raise RuntimeError("Unknown classifier architecture")

        predictions = {
            # Returns the highest prediction from the output of logits.
            tf.contrib.learn.PredictionKey.CLASSES : tf.argmax(logits, axis=1, name="predicted_classes"),
            # Softmax squashes arbitrary real values into a range of 0 to 1 and returns the
            # probabilities of each of the classes. Name is used for logging.
            tf.contrib.learn.PredictionKey.PROBABILITIES : tf.nn.softmax(logits, name="predicted_probability")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            exports = {
                'predictions' : tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=exports)

        predicted_labels = predictions[tf.contrib.learn.PredictionKey.CLASSES]

        accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_labels)
        precision = tf.metrics.precision(labels=labels, predictions=predicted_labels)
        recall = tf.metrics.recall(labels=labels, predictions=predicted_labels)

        onehot_labels = tf.one_hot(indices=labels, depth=len(self.label_map))
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            tf.summary.scalar("accuracy", accuracy[1])
            tf.summary.scalar("precision", precision[1])
            tf.summary.scalar("recall", recall[1])
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.classifier_spec.learning_rate)
            # This is needed when you use batch normalization, see the documentation.
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        self.export_annotated_images(images_uint8, labels, predicted_labels,
                                     "predictions_annotated")
        eval_summary_hook = tf.train.SummarySaverHook(save_secs=2,
                output_dir=self.classifier_spec.model_dir + "/eval",
                scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, evaluation_hooks=[eval_summary_hook],
                                          eval_metric_ops={
                                            "accuracy": accuracy,
                                            "precision": precision,
                                            "recall": recall
                                          })

    ##
    # Export the model from an estimator.
    ##
    def export(self, session, estimator):
        print("Exporting SavedModel")
        export_path = os.path.join(self.classifier_spec.model_dir, "exported")
        export_dir = estimator.export_savedmodel(export_path, self.serving_input_fn).decode("utf-8")
        print("Exported SavedModel to %s" % export_dir)
        print("Exporting tflite model")
        converter = tflite.TFLiteConverter.from_saved_model(export_dir, input_arrays=["whc_input"],
            output_arrays = ["predicted_classes", "predicted_probability"])
        tflite_model = converter.convert()
        open(os.path.join(export_dir, "model.tflite"), "wb").write(tflite_model)


    ##
    # The entry point of the application.
    ##
    def run(self):
        tf.logging.set_verbosity(tf.logging.INFO)

        # Standard initialization op.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Use only 90% of the gpu to keep the system responsive.
        config = tf.ConfigProto()
        config.log_device_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        with tf.Session(config=config) as session:
            # Initialize vars.
            session.run(init_op)
            # Setup input queue threads.
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coordinator, sess=session)
            # Build the estimator.
            estimator = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=self.classifier_spec.model_dir)

            if self.export_only:
                self.export(session, estimator)
                return

            # Train / eval loop.
            try:
                num_train_steps = self.format_spec.num_train / self.classifier_spec.batch_size
                num_eval_steps = self.format_spec.num_eval / self.classifier_spec.batch_size
                while not coordinator.should_stop():
                    # Train a few steps, eval a few steps.
                    estimator.train(input_fn=self.train_input_fn, steps=num_train_steps)
                    estimator.evaluate(input_fn=self.eval_input_fn, steps=num_eval_steps)
            except tf.errors.OutOfRangeError:
                print("Done with training.")
            finally:
                coordinator.request_stop()
                coordinator.join(threads)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mobilenet Classifier")
    parser.add_argument("--classifier_spec", type=str, default="./classifier_spec_mobilenet1.pbtxt", help="The path to the classifier spec pbtxt.")
    parser.add_argument("--export_only", type=bool, default=False, help="Just export a saved model.")
    parser.add_argument("--start_gpu", type=int, default=1, help="Start GPU.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs.")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    devices = ','.join(str(x) for x in range(args.start_gpu, args.start_gpu+args.num_gpus))
    print('devices={}'.format(devices))
    os.environ["CUDA_VISIBLE_DEVICES"]=devices  # specify which GPU(s) to be used

    app = Classifier(classifier_spec_path=args.classifier_spec, export_only=args.export_only)
    app.run()