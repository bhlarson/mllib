#!/usr/bin/python3

import argparse
import io
import classifier_spec_pb2
import os
import progressbar
import PIL
import random

import google.protobuf.text_format as text_format
import tensorflow as tf

R"""
python caltech_to_example.py \
    --images= \
    --output_train=/train.tfrecord \
    --output_eval=/eval.tfrecord \
    --label_map=/label_map.pbtxt \
    --format_spec=/format_spec.pbtxt \
    --width=256 \
    --height=256 \
    --channels=3
"""
class Converter:
    def __init__(self, images, output_train, output_eval, label_map_path,
                 format_spec_path, width, height, channels):
        self.images = images
        self.output_train = output_train
        self.output_eval = output_eval
        self.label_map_path = label_map_path
        self.format_spec_path = format_spec_path
        self.width = width
        self.height = height
        self.channels = channels

    def make_tf_example(self, label, img_path):
        with open(img_path, "rb") as f:
            encoded_image = f.read()
            with PIL.Image.open(img_path) as image:
                scaled_image = image.resize((self.width, self.height), PIL.Image.ANTIALIAS)
                if (self.channels == 3):
                    image_bytes = scaled_image.convert('RGB').tobytes()
                elif (self.channels == 1):
                    image_bytes = scaled_image.convert('L').tobytes()
                else:
                    raise ValueError("Unsupported format. %d" % self.channels)

                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                    'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                }))
                scaled_image.close()
        return tf_example

    def write_examples(self, writer, examples):
        i = 0
        bar = progressbar.ProgressBar(max_value=len(examples), redirect_stdout=True)
        for (label, path) in bar(examples):
            tf_example = self.make_tf_example(label, path)
            writer.write(tf_example.SerializeToString())
            i = i + 1
            bar.update(i)

    def save_label_map(self, label_map, path):
        label_map_proto = classifier_spec_pb2.LabelMap()
        for key, value in label_map.items():
            label = label_map_proto.item.add()
            label.id = value
            label.name = key
        with open(path, 'w') as f:
            f.write(text_format.MessageToString(label_map_proto))

    def save_format_spec(self, format_spec_path):
        format_spec = classifier_spec_pb2.FormatSpec()
        format_spec.width = self.width
        format_spec.height = self.height
        format_spec.channels = self.channels
        format_spec.num_train = self.num_train
        format_spec.num_eval = self.num_eval
        with open(format_spec_path, 'w') as f:
            f.write(text_format.MessageToString(format_spec))

    def main(self):
        i = 0
        label_map = {}
        input_pairs = []

        bar = progressbar.ProgressBar(redirect_stdout=True)
        print("Enumerating...")
        for (full_path, _, filenames) in bar(os.walk(self.images)):
            label = full_path.replace(self.images, '')
            if len(label) == 0:
                continue
            if len(filenames) == 0:
                continue

            label = label.replace('\\', '');
            label_map[label] = i

            for filename in filenames:
                input_pairs.append((i, os.path.join(full_path, filename)))
            i = i + 1
            bar.update(i)

        random.shuffle(input_pairs)
        num_examples = len(input_pairs)
        self.num_train = int((90 * num_examples) / 100)
        self.num_eval = num_examples - self.num_train

        print("Writing training data...")
        with tf.python_io.TFRecordWriter(self.output_train) as train_writer:
            self.write_examples(train_writer, input_pairs[0:self.num_train])

        print("Writing evaluation data...")
        with tf.python_io.TFRecordWriter(self.output_eval) as eval_writer:
            self.write_examples(eval_writer, input_pairs[self.num_train:])

        print("Writing label map")
        self.save_label_map(label_map, self.label_map_path)
        self.save_format_spec(self.format_spec_path)
        print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caltech256 to Tf.Example format converter")
    parser.add_argument("--images", type=str, default="C:\data\datasets\Caltech256")
    parser.add_argument("--output_train", type=str, default="./Caltech256/train.tfrecord")
    parser.add_argument("--output_eval", type=str, default="./Caltech256/eval.tfrecord")
    parser.add_argument("--label_map", type=str, default="./Caltech256/label_map.pbtxt")
    parser.add_argument("--format_spec", type=str, default="./Caltech256/format_spec.pbtxt")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--channels", type=int, default=3)
    args = parser.parse_args()
    app = Converter(args.images, args.output_train, args.output_eval,
                    args.label_map, args.format_spec,
                    args.width, args.height, args.channels)
    app.main()