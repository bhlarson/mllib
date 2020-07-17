import tensorflow as tf
import tensorflow_datasets as tfds

def input_fn(split=tfds.Split.TRAIN, dataset = 'iris', batch_size=32):
  dataset = tfds.load(dataset, split=split, as_supervised=True)
  dataset = dataset.map(lambda features, labels: ({'dense_input':features}, labels))
  dataset = dataset.batch(batch_size).repeat()
  return dataset

def input_fn_test():
    for features_batch, labels_batch in input_fn().take(1):
        print(features_batch)
        print(labels_batch)