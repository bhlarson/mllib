"""
Move classifier.py to reusable_custom_tf_estimator.py template
"""


import tensorflow as tf
import tensorflow.contrib.lite as tflite
import numpy as np

import os, sys, json, argparse, glob
from random import shuffle

# use Feature Import/Output to have unified schema
from fio import FIO

tf.__version__

"""## Constants"""

'''
Here we define all the constants in relation to the model flow 
i.e. things related to the data the model handels rather than how the 
model is constructed

In this example we are undertaking a multilabel problem with a sequence of
shape [SEQUENCE_LENGTH, SEQUENCE_CHANNELS] and we working with NUMBER_OF_LABELS 
labels ([SEQUENCE_LENGTH, NUMBER_OF_LABELS])
'''
SEQUENCE_LENGTH = 20
SEQUENCE_CHANNELS = 7
NUMBER_OF_LABELS = 4
NUMBER_OF_EXAMPLES = 100

'''
SCHEMA is a concept from the fio library to have a unified tf record 
write / read interface.

All the features of our data are defined here (even if they are not used in the
model).
'''
SCHEMA = {
    
    'Name': {'length': 'fixed', 'dtype': tf.string,  'shape': []},
    
    'Sequence': {
        'length': 'fixed',
        'dtype': tf.float32,
        'shape': [SEQUENCE_LENGTH, SEQUENCE_CHANNELS],
        'encode': 'channels',
        'data_format': 'channels_last'
    },
    
    'Labels': {
        'length': 'fixed',
        'dtype': tf.float32,
        'shape': [SEQUENCE_LENGTH, NUMBER_OF_LABELS],
        'encode': 'channels',
        'data_format': 'channels_last'
    }
}

# which features from our SCHEMA are the input / output
I_FEATURE = 'Sequence'
O_FEATURE = 'Labels'

# the input and output types
I_DTYPE = SCHEMA[I_FEATURE]['dtype']
O_DTYPE = SCHEMA[O_FEATURE]['dtype']

# function to get the input / output shapes. Since these are dependent on the 
# batch size, and that might change per experiment, these are lambdas
I_SHAPE = lambda bs: (bs, SCHEMA[I_FEATURE]["shape"][0],  SCHEMA[I_FEATURE]["shape"][1])
O_SHAPE = lambda bs: (bs, SCHEMA[O_FEATURE]["shape"][0],  SCHEMA[O_FEATURE]["shape"][1])

fio = FIO(
    schema = SCHEMA,
    etype = 'sequence_example',
    sequence_features = [I_FEATURE, O_FEATURE]
)

MODEL_DIR = './test'
 

# all of the file names (write one example per record)
FILE_NAMES = [f'sequence_{i}.tfrecord' for i in range(NUMBER_OF_EXAMPLES)]

# how to partition our datset into train, validation and test sets
DATA_RATIOS = [0.7, 0.2, 0.1]

"""# Data

## Utils
"""

def tf_type_string(tf_type:str): 
  return str(tf_type).replace("<dtype: \'", '').replace("\'>", '')


def random_encode_multilabels(array:list, number_of_labels:int):
  labels = [
      [1 for i in range(number_of_non_zeros)] + 
      [0 for i in range(number_of_labels - number_of_non_zeros)] 
      for number_of_non_zeros in array
  ]
  from random import shuffle
  for l in labels:
    shuffle(l)
  return labels

def random_multilabels(shape):
  batch_size, length, labels = shape
  return np.array([
      random_encode_multilabels(num_nonzeros, labels) for num_nonzeros in 
      np.random.randint(1, labels, (batch_size, length))
  ])

def random_encode_onehot(array:list, number_of_classes:int):
  hot = [[0 for i in range(number_of_classes)] for element in array]
  for which, encoded in enumerate(hot):
    encoded[array[which]] = 1
  return hot

def random_onehot(shape):
  batch_size, length, channels = shape
  return np.array([
      random_encode_onehot(hot_channel, channels) for hot_channel in 
      np.random.randint(0, channels-1, (batch_size, length))
  ])

def partition_files(files, train=1, valid=0, test=0):
  n = len(files)
  
  shuffle(files)
  
  a = int(n * (train))
  b = int(n * (train + valid))
  c = int(n * (train + valid + test))
  
  return {
      'train': files[:a],
      'valid': files[a:b],
      'test':  files[b:]
  }

"""## Make dummy data"""

# our sequences are fixed length with binary channels that, in this case,
# happen to never be 1 at the same instance
sequences = random_onehot((NUMBER_OF_EXAMPLES, SEQUENCE_LENGTH, SEQUENCE_CHANNELS))\
            .astype(tf_type_string(I_DTYPE))
  
# randomly make multilabels
seqlabels = random_multilabels((NUMBER_OF_EXAMPLES, SEQUENCE_LENGTH, NUMBER_OF_LABELS))\
            .astype(tf_type_string(O_DTYPE))

"""### see what data looks like"""

# (features, labels)
(sequences[0], seqlabels[0])

"""## Write dummy data to TF Records"""

for i in range(NUMBER_OF_EXAMPLES):
  # give our examples a name
  name = f'sequence_{i}'
  
  schema = {'Name': name, 'Sequence': sequences[i], 'Labels': seqlabels[i]}
  
  # writing tf records is so much easier like this
  example = fio.to_example(schema)
  
  # each example gets a record
  with tf.python_io.TFRecordWriter(f'{name}.tfrecord') as writer:
    writer.write(example.SerializeToString())

"""## Split data"""

'''
DATASET is a dictionary of keys 'train', 'valid', and test', each of which 
corresponds to a list of strings indicating full paths to TF Record files.
'''
DATASET = partition_files(FILE_NAMES, *DATA_RATIOS)

print('DS\t# example')
for key in DATASET:
  print(key, len(DATASET[key]), sep='\t')

"""# Architecture Functions

## Loss fn
"""

def multilabel_loss(outputs, targets):
    # Note: sigmoid_cross_entropy_with_logits applies sigmoid to the logits
    with tf.variable_scope('multilabel_loss'):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=outputs))

  
  
def loss_fn(model):
    """How to calculate the loss of the model.

    Args:
        model (dict): a `dict` containing the model

    Returns:
        model (dict): an updated `dict` containing the loss of the model

    """

    # here you extract what you need to calculate the loss
    logits = model['net_outputs']
    labels = model['labels']

    # here you calculate the loss
    loss = multilabel_loss(logits, labels)

    # add loss to model
    model['loss'] = loss
    return model

"""# Estimator Functions

## build_fn
"""

def conv1d_to_labels(inputs, labels:int, kernel_size:int):
    '''
    Takes <inputs> tensor and calls conv1d with <labels> number of filters using
    <kernels_size> and padding='same'. Ideal to reshape <input> into correct
    shape for semantic segmentation problems
    '''
    with tf.variable_scope('conv1d_to_labels'):
        x = tf.layers.conv1d(inputs, labels, kernel_size, name="conv", padding="same", reuse=tf.AUTO_REUSE)
        return x


def build_fn(model):
  '''
  build_fn serves to construct the architecture / wire the network for all by the
  last activation function.
  '''

  # here we extract what is needed for building the graph
  features = model['features']
  params   = model['params']
  x = features['input_tensors']
  
  # NEEDED because TF Records are the worst
  # x.set_shape(I_SHAPE(params['batch_size']))  
  if model['mode'] != 'predict':
    x.set_shape(I_SHAPE(None))
 
  # here you wire how features go throught the graph
  # in this toy example, we just reshape (via a convolution) to match the labels
  x = conv1d_to_labels(x, NUMBER_OF_LABELS, params['kernel'])

  # here you store the outputs of the graph
  model['net_outputs'] = x # our loss automatically applies the sigmoid for us
  model['net_logits']  = tf.nn.sigmoid(x) # <--- the actual logits
  return model

'''
tf.data.TFRecordDataset(FILE_NAMES)                                   \ # dataset from files
.map(lambda record: fio.from_record(record))                          \ # use the schema defined once to load from tf records
.map(lambda context, features: fio.reconstitute((context, features))) \ # undo the dumb forced wrapping of tf records
.batch(2).make_one_shot_iterator().get_next()                         \ # set batch and make iterator
'''

"""## input_fn"""

def input_fn(filenames:list, params):
  mode = params['mode'] if 'mode' in params else 'train'
  batch_size = params['batch_size']
 
  
  shuffle(filenames) # <--- far more efficient than tf dataset shuffle
  dataset = tf.data.TFRecordDataset(filenames)
  
  # using fio's SCHEMA fill the TF Feature placeholders with values
  dataset = dataset.map(lambda record: fio.from_record(record))
  
  # using fio's SCHEMA restructure and unwrap (if possible) features (because tf records require wrapping everything into a list)
  dataset = dataset.map(lambda context, features: fio.reconstitute((context, features)))
  
  # dataset should be a tuple of (features, labels)
  dataset = dataset.map(lambda context, features: ( 
      {"input_tensors": features[I_FEATURE]}, # features
      features[O_FEATURE]                     # labels
    )
  ) 
  
  
  if mode == 'train':
    # during evaluation, we do not want to repeat forever
    dataset = dataset.repeat()
    
  # dataset = dataset.batch(batch_size)
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  
  return dataset.make_one_shot_iterator().get_next()

"""## mode_fns

### train
"""

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

est.

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

predict_fn



