"""
Move classifier.py to reusable_custom_tf_estimator.py template
"""


import tensorflow as tf
import tflite
import numpy as np

import os, sys, json, argparse, glob
from random import shuffle
from resnet50 import make_resnet50
from model_mobilenet_v1 import mobilenet_v1_model_fn, default_model_params
import shutil

# use Feature Import/Output to have unified schema
from fio import FIO

print('tensorflow.__version={}'.format(tf.__version__))

NUM_INPUT_THREADS = 6
VALIDATION_REPEATS = -1 # Forever.
SHUFFLE_BUFFER_SIZE = 512
PREFETCH_BUFFER_SIZE = SHUFFLE_BUFFER_SIZE * 4

def ParseInputs():
    parser = argparse.ArgumentParser(description="Mobilenet Classifier")
    parser.add_argument("--dim", "--dimensions", nargs=3, type=int, default=[256,256,3], metavar=('height','width','colors'), help="Input image height, width, and colors.  For example: --dim 256,256,3")
    parser.add_argument("--num_classes", type=int, default=257, help="num_classes")  
    parser.add_argument("--model_dir", type=str, default="./cmodel", help="Path to save and load model checkpoints")
    parser.add_argument("--classifier_spec", type=str, default="./classifier_spec_mobilenet1.pbtxt", help="The path to the classifier spec pbtxt.")
    parser.add_argument("--train_path", type=str, default="./Caltech256/train.tfrecord", help="path to tfrecord file containing training set")
    parser.add_argument("--eval_path", type=str, default="./Caltech256/eval.tfrecord", help="path to tfrecord file containing eval set")
    parser.add_argument("--num_epochs", type=int, default=16, help="Number of training epocs.")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size.")   
    parser.add_argument("--export_only", type=bool, default=False, help="Just export a saved model.")
    parser.add_argument("--start_gpu", type=int, default=0, help="Start GPU.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs.")
    parser.add_argument("--clean", type=bool, default = False, help="Clean model directory before train")


    parser.add_argument('--train_epochs', type=int, default=26,
                        help='Number of training epochs: '
                            'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                            'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                            'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                            'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                            'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                            'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')

    parser.add_argument('--epochs_per_eval', type=int, default=1,
                        help='The number of training epochs to run between evaluations.')

    parser.add_argument('--tensorboard_images_max_outputs', type=int, default=6,
                        help='Max number of batch elements to generate for Tensorboard.')

    #parser.add_argument('--batch_size', type=int, default=4, help='Number of examples per batch.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,help='Adam optimizer learning rate.')

    parser.add_argument('--max_iter', type=int, default=30,help='Number of maximum iteration used for "poly" learning rate policy.')

    parser.add_argument('--debug', action='store_true',help='Whether to use debugger to track down bad values during training.')

    return parser.parse_args()

'''
NOTE:
try the following mode_predict function and then in `model_fn` replace
  
   if mode == tf.estimator.ModeKeys.PREDICT: 
      return mode_predict(MODEL)
      
 with
 
   mode_predict(MODEL)
   if mode == tf.estimator.ModeKeys.PREDICT:
      return MODEL['PREDICT_SPEC']
      

This will result in an error when running train_and_evaluate because the spec
will trigger when initiated, rather than when returned.
      
 -----------------------
 
def mode_predict(model):
    # do the predictions here
    model['predictions'] = {'labels': model['net_logits']}

    model['export_outputs'] = {
        k: tf.estimator.export.PredictOutput(v) for k, v in model['predictions'].items()
    }
    
    spec = tf.estimator.EstimatorSpec(
        mode           = model['mode'],
        predictions    = model['predictions'],
        export_outputs = model['export_outputs']
    )
    model['PREDICT_SPEC'] = spec
    return model

'''

"""## metrics_fn"""

def metrics_fn(model):
    """Produce metrics of the model to monitor during training.

    Args:
        model (dict): a `dict` containing the model

    Returns:
        model (`dict`): an update `dict` containg the metrics

    .. _EstimatorSpec:
        https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec

    """
    # here you extract variables for easier reading
    labels = model['labels']
    predicted = model['predictions']['labels']

    # here you calculate your metrics
    mae = tf.metrics.mean_absolute_error(labels=labels, predictions=predicted, name='mea_op')
    mse = tf.metrics.mean_squared_error(labels=labels, predictions=predicted, name='mse_op')

    # here you add your metrics (or anything else) to tf.summary to be monitored
    tf.summary.scalar('mae', mae[1])
    tf.summary.scalar('mse', mse[1])

    # here you add the above metrics to the model
    metrics = {'mae': mae, 'mse': mse}
    model['metrics'] = metrics
    return model

"""## model_fn"""

def model_fn(features, labels, mode, params):
    MODEL = {'features': features, 'labels': labels, 'mode': mode, 'params': params}

    # send the features through the graph
    MODEL = build_fn(MODEL)

    # prediction
    MODEL['predictions'] = {'labels': MODEL['net_logits']}

    MODEL['export_outputs'] = {
        k: tf.estimator.export.PredictOutput(v) for k, v in MODEL['predictions'].items()
    }

    
    if mode == tf.estimator.ModeKeys.PREDICT: 
      return mode_predict(MODEL)

    # calculate the loss
    MODEL = loss_fn(MODEL)

    # calculate all metrics and send them to tf.summary
    MODEL = metrics_fn(MODEL)

    if mode == tf.estimator.ModeKeys.EVAL: 
      return mode_eval(MODEL)

    if mode == tf.estimator.ModeKeys.TRAIN: 
      return mode_train(MODEL)

# from previous S.O. Question: 
# https://stackoverflow.com/questions/52641737/tensorflow-1-10-custom-estimator-early-stopping-with-train-and-evaluate


class Learn:
    def __init__(self, args):
        self.args = args
        self.estimator = {}

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        devices = ','.join(str(x) for x in range(self.args.start_gpu, self.args.start_gpu+self.args.num_gpus))
        print('devices={}'.format(devices))
        os.environ["CUDA_VISIBLE_DEVICES"]=devices  # specify which GPU(s) to be used

    def create_model(self):
        # Set up a RunConfig to only save checkpoints once per training cycle.
        if self.estimator =={}:
            run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
            self.estimator = tf.estimator.Estimator(
                model_fn=mobilenet_v1_model_fn,
                model_dir=self.args.model_dir,
                config=run_config,
                params=default_model_params(self.args))

    def serving_input_fn(self):
        shape = [_WIDTH, _HEIGHT, _DEPTH]
        features = {
            "image" : tf.FixedLenFeature(shape=shape, dtype=tf.uint8),
        }
        return tf.estimator.export.build_parsing_serving_input_receiver_fn(features)

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
            dims = self.args.dim
            image = tf.reshape(image, dims)
            return { "image" : image }, parsed[label_key]

        dataset = dataset.map(parser, num_parallel_calls=NUM_INPUT_THREADS)
        dataset = dataset.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.batch(self.args.batch_size)
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
        return self.make_dataset_input_fn(self.args.train_path,
                                          self.args.num_epochs)

    ##
    # An input function for validation data.
    ##
    def eval_input_fn(self):
        return self.make_dataset_input_fn(self.args.eval_path,
                                          self.args.num_epochs)

    def train(self):
        # os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1' # Using the Winograd non-fused algorithms provides a small performance boost.

        if self.args.clean:
            shutil.rmtree(self.args.model_dir, ignore_errors=True)

        tf.logging.set_verbosity(tf.logging.INFO)

        self.create_model()

        for _ in range(self.args.train_epochs // self.args.epochs_per_eval):
            tensors_to_log = {
            #'learning_rate': 'learning_rate',
            #'cross_entropy': 'cross_entropy',
            #'train_px_accuracy': 'train_px_accuracy',
            }

            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=10)
            train_hooks = [logging_hook]
            eval_hooks = None

            if self.args.debug:
                debug_hook = tf_debug.LocalCLIDebugHook()
                train_hooks.append(debug_hook)
                eval_hooks = [debug_hook]

        tf.logging.info("Start training.")
        self.estimator.train(
            input_fn=self.train_input_fn,
            hooks=train_hooks,
            # steps=1  # For debug
        )

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

        self.estimator.export_saved_model('saved_model', self.serving_input_fn)

    def eval(self):
        self.create_model()

        tf.logging.info("Start evaluation.")
        # Evaluate the model and print results
        eval_results = self.estimator.evaluate(
            # Batch size must be 1 for testing because the images' size differs
            input_fn=lambda: input_fn(False, FLAGS.data_dir, 1),
            hooks=eval_hooks,
            # steps=1  # For debug
        )
        print(eval_results)

if __name__ == "__main__":
    args = ParseInputs()
    learn = Learn(args)
    learn.train()
    learn.eval()




