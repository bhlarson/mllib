"""
Move classifier.py to reusable_custom_tf_estimator.py template
"""


import tensorflow as tf
import tflite
import numpy as np

import os, sys, json, argparse, glob
from random import shuffle
from resnet50 import make_resnet50
from model_mobilenet_v1 import mobilenet_v1_model_fn

# use Feature Import/Output to have unified schema
from fio import FIO

tf.__version__

def ParseInputs():
    parser = argparse.ArgumentParser(description="Mobilenet Classifier")
    parser.add_argument("--classifier_spec", type=str, default="./classifier_spec_mobilenet1.pbtxt", help="The path to the classifier spec pbtxt.")
    parser.add_argument("--export_only", type=bool, default=False, help="Just export a saved model.")
    parser.add_argument("--start_gpu", type=int, default=1, help="Start GPU.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs.")

    return parser.parse_args()

def mode_train(model):
    """How to train the model.

    Args:
        model (dict): a `dict` containing the model

    Returns:
        spec (`EstimatorSpec`_): Ops and objects returned from a model_fn and passed to an Estimator

    .. _EstimatorSpec:
        https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec

    """
    # extract variables for easier reading here
    global_step   = tf.train.get_global_step()
    learning_rate = model['params']['learning_rate']
    loss          = model['loss']

    # do the training here
    model['optimizer'] = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    model['train_op'] = model['optimizer'].minimize(loss, global_step=global_step)

    spec = tf.estimator.EstimatorSpec(
        mode            = model['mode'],
        loss            = model['loss'],
        train_op        = model['train_op'],
        eval_metric_ops = model['metrics'],
        predictions     = model['predictions'],
        export_outputs  = model['export_outputs']
    )
    return spec

"""### eval"""

def mode_eval(model:dict):
    """How to evaluate the model.

    Args:
        model (dict): a `dict` containing the model

    Returns:
        spec (`EstimatorSpec`_): Ops and objects returned from a model_fn and passed to an Estimator

    .. _EstimatorSpec:
        https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec

    """
    # do the evaluation here

    spec = tf.estimator.EstimatorSpec(
        mode            = model['mode'],
        loss            = model['loss'],
        eval_metric_ops = model['metrics'],
        predictions     = model['predictions'],
        export_outputs  = model['export_outputs']
    )
    return spec

"""### predict"""

def mode_predict(model):
    """How to predict given the model.

    Args:
        model (dict): a `dict` containing the model

    Returns:
        spec (`EstimatorSpec`_): Ops and objects returned from a model_fn and passed to an Estimator

    .. _EstimatorSpec:
        https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec

    """
    # do the predictions here

    spec = tf.estimator.EstimatorSpec(
        mode           = model['mode'],
        predictions    = model['predictions'],
        export_outputs = model['export_outputs']
    )
    return spec

  
  
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

"""Note: ideally the `MODEL['predictions']` / `MODEL['export_outputs']` would be done in `mode_predict` function, which would return the model along with the prediction `spec`. However, the instant a estimator spec is made, (regardless of scope it seems), it will evalulate that spec. So that is silly.

## serving_fn
"""

serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
    {'input_tensors': tf.placeholder(tf.float32, I_SHAPE(None), name="input_tensors")})

def serving_input_receiver_fn():
  input_tensors = tf.placeholder(tf.float32, I_SHAPE(None), name="input_tensors")


  features = {'input_tensors' : input_tensors} # this is the dict that is then passed as "features" parameter to your model_fn
  receiver_tensors = {'input_tensors': input_tensors} # As far as I understand this is needed to map the input to a name you can retrieve later
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

"""## Run Config"""

run_config = tf.estimator.RunConfig(**{
    "model_dir": MODEL_DIR,
    "keep_checkpoint_max": 5
})

"""## Run Params"""

run_params = {
    "batch_size": 5,
    "kernel": 3,
    "learning_rate": 0.001,
}

"""# Estimator

## init estimator
"""

est = tf.estimator.Estimator(
    model_fn = model_fn,
    config = run_config,
    params = run_params,
)

"""## define exporter"""

# from previous S.O. Question: 
# https://stackoverflow.com/questions/52641737/tensorflow-1-10-custom-estimator-early-stopping-with-train-and-evaluate

exporter = tf.estimator.BestExporter(
    name="best_exporter",
    serving_input_receiver_fn=serving_input_receiver_fn,
    # event_file_pattern="model_*", # <--- doesn't do anything?
    exports_to_keep=5
) # this will keep the 5 best checkpoints

"""## define train and eval spec"""

eval_run_params = {**run_params, 'mode': 'eval'}

train_fn = lambda: input_fn(DATASET['train'], run_params)
valid_fn = lambda: input_fn(DATASET['valid'], eval_run_params)
test_fn  = lambda: input_fn(DATASET['test'],  eval_run_params)


# early_stop = tf.contrib.estimator.stop_if_no_decrease_hook(est, 'loss', 10)

train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=200, hooks=[
#     early_stop
])
eval_spec  = tf.estimator.EvalSpec( input_fn=valid_fn, exporters=exporter)

"""# Run"""

run = tf.estimator.train_and_evaluate(est, train_spec, eval_spec)

"""## export saved model"""
est.export_savedmodel('./here', serving_input_receiver_fn)

# what does "Pass your op to the equivalent parameter main_op instead." mean?

"""## evaluate with test set"""
test_res = est.evaluate(input_fn=test_fn)

test_res

"""# Predict with trained model"""

to_predict = random_onehot((1, SEQUENCE_LENGTH, SEQUENCE_CHANNELS))\
            .astype(tf_type_string(I_DTYPE))
pred_features = {'input_tensors': to_predict}
pred_ds = tf.data.Dataset.from_tensor_slices(pred_features)

predicted = est.predict(lambda: pred_ds, yield_single_examples=True)

'''
Why does this throw an error?

Also, how would one load the saved model and predict, rather than use the current runtime instance?
'''

# next(predicted)

to_predict = random_onehot((2, SEQUENCE_LENGTH, SEQUENCE_CHANNELS))\
            .astype(tf_type_string(I_DTYPE))
pred_features = {'input_tensors': to_predict}

def predict_input_fn(data, batch_size=2):
  dataset = tf.data.Dataset.from_tensor_slices(data)
  return dataset.batch(batch_size).prefetch(None)

predicted = est.predict(lambda: predict_input_fn(pred_features), yield_single_examples=False)
next(predicted)
# predicted

import os
from tensorflow.contrib import predictor

predict_fn = predictor.from_saved_model('./here/{}'.format(os.listdir('./here')[0]))



predict_fn(pred_features)


class Learn:
    def __init__(self, args):
        self.args = args

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
        #config.gpu_options.per_process_gpu_memory_fraction = 0.9

        with tf.Session(config=config) as session:
            # Initialize vars.
            session.run(init_op)
            # Setup input queue threads.
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coordinator, sess=session)
            # Build the estimator.
            estimator = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=self.classifier_spec.model_dir)

            if self.export_only:
                self.export(estimator)
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
    args = ParseInputs()

    learn = Learn(args)





