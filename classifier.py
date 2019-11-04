#!/usr/bin/python3

import argparse
import augmentations
import classifier_spec_pb2
import glob
import random
import os

import tensorflow as tf
import tensorflow.contrib.lite as tflite
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
    def augment(self, image):
        if self.classifier_spec.augmentations.flip_horizontal:
            image = self.aug.flip_horizontal(image)

        if self.classifier_spec.augmentations.color_adjustment:
            image = self.aug.adjust_color(image)

        if (self.classifier_spec.augmentations.translate_vertical_max_px > 0 or
                self.classifier_spec.augmentations.translate_horizontal_max_px > 0):
            image = self.aug.translate(image,
                        self.classifier_spec.augmentations.translate_horizontal_max_px,
                        self.classifier_spec.augmentations.translate_vertical_max_px)

        if self.classifier_spec.augmentations.rotate_max_degrees > 0:
            image = self.aug.rotate(image, self.classifier_spec.augmentations.rotate_max_degrees)

        return image

    ##
    # Sets up the dataset to read from.
    ##
    def make_dataset_input_fn(self, path, epochs, should_augment):
        dataset = tf.data.TFRecordDataset([path])

        def parser(record):
            label_key = "image/label"
            bytes_key = "image/encoded"
            parsed = tf.parse_single_example(record, {
                bytes_key : tf.FixedLenFeature([], tf.string),
                label_key : tf.FixedLenFeature([], tf.int64),
            })
            # Takes the raw byte array and reshapes it into W x H x C
            image = tf.decode_raw(parsed[bytes_key], tf.uint8)
            dims = [self.format_spec.width, self.format_spec.height, self.format_spec.channels]
            image = tf.reshape(image, dims)
            # Apply random transformations to the image.
            if should_augment:
                image = self.augment(image)
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
                                          self.classifier_spec.num_epochs,
                                          should_augment=True)

    ##
    # An input function for validation data.
    ##
    def eval_input_fn(self):
        return self.make_dataset_input_fn(self.classifier_spec.eval_path,
                                          self.classifier_spec.num_epochs,
                                          should_augment=False)

    ##
    # Input function that is used by the saved model.
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
    # Makes the mobilenet v1 architecture.
    # Mobilenet: https://arxiv.org/pdf/1704.04861.pdf
    ##
    def make_mobilenet_v1(self, input_layer, mode):
        is_training = tf.constant(mode == tf.contrib.learn.ModeKeys.TRAIN)
        if self.classifier_spec.data_format == "NHWC":
            axis = 3
        else:
            axis = 1

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
    parser.add_argument("--export_only", type=bool, default=True, help="Just export a saved model.")
    parser.add_argument("--start_gpu", type=int, default=1, help="Start GPU.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs.")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    devices = ','.join(str(x) for x in range(args.start_gpu, args.start_gpu+args.num_gpus))
    print('devices={}'.format(devices))
    os.environ["CUDA_VISIBLE_DEVICES"]=devices  # specify which GPU(s) to be used

    app = Classifier(classifier_spec_path=args.classifier_spec, export_only=args.export_only)
    app.run()