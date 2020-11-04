import argparse
import json
import os
import sys
import shutil
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from datetime import datetime
import matplotlib.pyplot as plt

print('Python Version {}'.format(sys.version))
print('Tensorflow version {}'.format(tf.__version__))
print('GPU Available: {}'.format(tf.test.is_gpu_available()))
if(tf.test.is_gpu_available()):
  print('GPU Devices: {}'.format(tf.test.gpu_device_name()))

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debugger attach')
parser.add_argument('-clean', action='store_true', help='If set, delete model directory at startup.')
parser.add_argument('-batch_size', type=int, default=32, help='Number of examples per batch.')
parser.add_argument('-classes', type=int, default=5, help='Number of examples per batch.')
parser.add_argument('-size_x', type=int, default=224, help='Training image size_x')
parser.add_argument('-size_y', type=int, default=224, help='Training image size_y')
parser.add_argument('-depth', type=int, default=3, help='Training image depth')
parser.add_argument('-epochs', type=int, default=3, help='Training epochs')
parser.add_argument('-model_dir', type=str, default='./trainings/unetcoco',help='Directory to store training model')
parser.add_argument('-savedmodel', type=str, default='./saved_model', help='Path to savedmodel.')
parser.add_argument('-loadsavedmodel', type=str, default=None, help='Saved model to load if no checkpoint')
parser.add_argument('-learning_rate', type=float, default=1e-3, help='Adam optimizer learning rate.')
parser.add_argument('-dataset', type=str, default='tf_flowers', choices=['tf_flowers'], help='Model Optimization Precision.')

parser.add_argument("-strategy", type=str, default='mirrored', help="Replication strategy. 'mirrored', 'onedevice' now supported ")
parser.add_argument("-devices", type=json.loads, default=None,  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')
defaultsavemodeldir = '{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-cfy'))
parser.add_argument('-savedmodelname', type=str, default=defaultsavemodeldir, help='Final model')
parser.add_argument('-weights', type=str, default='imagenet', help='Model initiation weights. None prevens loading weights from pre-trained networks')

def WriteDictJson(outdict, path):

    jsonStr = json.dumps(outdict, sort_keys=False)
    f = open(path,"w")
    f.write(jsonStr)
    f.close()
       
    return True

def LoadModel(config, model_dir=None, loadsavedmodel=None):
    model = None 

    if loadsavedmodel is not None and len(loadsavedmodel)>0:
        try:
            model = tf.keras.models.load_model(loadsavedmodel) # Load from checkpoint

        except:
            print('Unable to load weghts from {}'.format(loadsavedmodel))
            model = None 

    if model is None:
        if not config['clean'] and model_dir is not None:
            try:
                model = tf.keras.models.load_model(model_dir)
            except:
                print('Unable to load weghts from {}'.format(model_dir))

    if model is None:

        model = keras.applications.ResNet50V2(
            include_top=True, weights=config['init_weights'], input_shape=config['shape'],
            classifier_activation='softmax'
        )

        if model_dir is not None and not config['clean']:
            model.load_weights(model_dir)

    if model:
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model

def prepare_image(image, label, config):
    image = tf.image.resize_with_crop_or_pad(image, config['shape'][0], config['shape'][1])
    return image, label

def input_fn(config, split):
    dataset, metadata = tfds.load('tf_flowers', with_info=True, split=split, shuffle_files=True, as_supervised=True)
    dataset = dataset.map(lambda features, label: prepare_image(features, label, config) , num_parallel_calls = 10)
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.prefetch(config['batch_size'])
    #dataset = dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # Test new autotune prefetch 
    dataset = dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # Test new autotune prefetch 
    return dataset, metadata

def graph_history(epochs,loss,val_loss,savedmodelpath):
    plt.figure()
    if loss:
        plt.plot(epochs, loss, 'r', label='Training loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig('{}/training.svg'.format(savedmodelpath))

def main(args):
    tf.config.experimental_run_functions_eagerly(False)
    config = {
        'dataset':args.dataset,
        'batch_size': args.batch_size,
        'shape': (args.size_y, args.size_x, args.depth),
        'split': tfds.Split.TRAIN,
        'classes': 0, # load classes from dataset
        'learning_rate': args.learning_rate,
        'init_weights':'imagenet',
        'clean': args.clean,
        'epochs':args.epochs,
        'trining':'train[:80%]',
        'validation':'train[80%:]',
    }

    strategy = None
    if(args.strategy == 'mirrored'):
        strategy = tf.distribute.MirroredStrategy(devices=args.devices)

    else:
        device = "/gpu:0"
        if args.devices is not None and len(args.devices > 0):
            device = args.devices[0]

        strategy = tf.distribute.OneDeviceStrategy(device=device)

    print('{} distribute with {} GPUs'.format(args.strategy,strategy.num_replicas_in_sync))

    if args.clean:
        shutil.rmtree(args.model_dir, ignore_errors=True)

    savedmodelpath = '{}/{}'.format(args.savedmodel, args.savedmodelname)
    if not os.path.exists(savedmodelpath):
        os.makedirs(savedmodelpath)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    with strategy.scope():
        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.model_dir, monitor='loss',verbose=0,save_weights_only=False,save_freq='epoch')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.model_dir, histogram_freq='epoch')
        callbacks = [
            #save_callback,
            #tensorboard_callback
        ]

        model =  LoadModel(config, args.model_dir)

        # Display model
        model.summary()

        train_set, datasetdata = input_fn(config, split=config['trining'])
        val_set, _ = input_fn(config, split=config['validation'])

        model_history = model.fit(train_set, 
                                  validation_data=val_set,
                                  epochs=args.epochs, 
                                  callbacks=callbacks)
        history = model_history.history

        history = model_history.history
        if 'loss' in history:
            loss = model_history.history['loss']
        else:
            loss = []
        if 'val_loss' in history:
            val_loss = model_history.history['val_loss']
        else:
            val_loss = []

        model_description = {'config':config,
                             'results': history
                             }
        epochs = range(config['epochs'])

        graph_history(epochs,loss,val_loss,savedmodelpath)

    model.save(savedmodelpath, save_format='tf')
    model_description = {'config':config,
                         'results': history
                        }
    WriteDictJson(model_description, '{}/description.json'.format(savedmodelpath))

    print("Training complete. Results saved to {}".format(savedmodelpath))

if __name__ == '__main__':
  args, unparsed = parser.parse_known_args()
  
  if args.debug:
      print("Wait for debugger attach")
      import ptvsd
      # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
      # Launch applicaiton on remote computer: 
      # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait fcn/train.py
      # Allow other computers to attach to ptvsd at this IP address and port.
      ptvsd.enable_attach(address=('0.0.0.0', 3000), redirect_output=True)
      # Pause the program until a remote debugger is attached

      ptvsd.wait_for_attach()

      print("Debugger attached")

  main(args)