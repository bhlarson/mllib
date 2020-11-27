# Based on https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb

import argparse
import json
import os
import sys
import math
import shutil
import cv2
import tempfile
import tensorflow as tf
#from tensorflow.python.framework.ops import disable_eager_execution
#import tensorflow_model_optimization as tfmot
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.callbacks import Callback
# Depending on your keras version:-
from tensorflow.keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer

sys.path.insert(0, os.path.abspath(''))
from segment.display import DrawFeatures, WritePredictions
from segment.data import input_fn
from networks.unet import unet_model, unet_compile
from utils.s3 import s3store

DEBUG = False


#disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-clean', action='store_true', help='If set, delete model directory at startup.')
parser.add_argument('-min', action='store_true', help='If set, minimum training to generate output.')

parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
parser.add_argument('-model_precision', type=str, default='FP16', choices=['FP32', 'FP16', 'INT8'], help='Model Optimization Precision.')
parser.add_argument('-channel_order', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Channels_last = NHWC, Tensorflow default, channels_first=NCHW')

parser.add_argument('-trainingset_dir', type=str, default='/store/training/coco', help='Path training set tfrecord')
parser.add_argument('-training_dir', type=str, default='./trainings/unetcoco',help='Training directory.  Empty string for auto-generated tempory directory')
parser.add_argument('-checkpoint', type=str, default='train.ckpt',help='Directory to store training model')

parser.add_argument('--datasetprefix', type=str, default='dataset', help='Dataset prefix')
parser.add_argument('--trainingsetprefix', type=str, default='trainingset', help='Trainingset prefix')
parser.add_argument('--modelprefix', type=str, default='model', help='Model prefix')

parser.add_argument('--trainingset', type=str, default='coco', help='training set')
parser.add_argument('--initialmodel', type=str, default='2020-11-26-11-56-17-cocoseg', help='Initial model.  Empty string if no initial model')
parser.add_argument('--temp_savedmodel', type=str, default='./saved_model', help='Temporary path to savedmodel.')

parser.add_argument('-epochs', type=int, default=10, help='Number of training epochs')

parser.add_argument('-prune_epochs', type=int, default=0, help='Number of pruning epochs')

parser.add_argument('-tensorboard_images_max_outputs', type=int, default=2,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('-batch_size', type=int, default=16, help='Number of examples per batch.')               

parser.add_argument('-learning_rate', type=float, default=1e-3, help='Adam optimizer learning rate.')

parser.add_argument("-strategy", type=str, default='mirrored', help="Replication strategy. 'mirrored', 'onedevice' now supported ")
parser.add_argument("-devices", type=json.loads, default=None,  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')

parser.add_argument('-training_crop', type=json.loads, default='[480, 512]', help='Training crop size [height, width]')
parser.add_argument('-train_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB') 

parser.add_argument('-savedmodel', type=str, default='./saved_model', help='Path to savedmodel.')
defaultsavemodeldir = '{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-cocoseg'))
parser.add_argument('-savedmodelname', type=str, default=defaultsavemodeldir, help='Final model')
parser.add_argument('-weights', type=str, default='imagenet', help='Model initiation weights. None prevens loading weights from pre-trained networks')

def WriteDictJson(outdict, path):

    jsonStr = json.dumps(outdict, sort_keys=False)
    f = open(path,"w")
    f.write(jsonStr)
    f.close()
       
    return True

def LoadModel(config, s3):
    model = None 
    print('LoadModel initial model: {}, training directory: {}, '.format(config['initialmodel'], config['training_dir']))
    if config['initialmodel'] is not None:
        tempinitmodel = tempfile.TemporaryDirectory(prefix='initmodel', dir='.')
        modelpath = tempinitmodel.name+'/'+config['initialmodel']
        os.makedirs(modelpath)
        try:
            s3model=config['mllibsets']['model']['prefix']+'/'+config['initialmodel']
            success = s3.GetDir(config['mllibsets']['model']['bucket'], s3model, modelpath)
            model = tf.keras.models.load_model(modelpath) # Load from checkpoint

        except:
            print('Unable to load weghts from http://{}/minio/{}/{}'.format(
                config['s3_address'],
                config['mllibsets']['model']['prefix'],
                modelpath)
            )
            model = None 
        shutil.rmtree(tempinitmodel, ignore_errors=True)

    if model is None:
        if not config['clean'] and config['training_dir'] is not None:
            try:
                model = tf.keras.models.load_model(config['training_dir'])
            except:
                print('Unable to load weghts from {}'.format(config['training_dir']))

    if model is None:
        model = unet_model(classes=config['classes'], 
                           input_shape=config['input_shape'], 
                           weights=config['weights'], 
                           channel_order=config['channel_order'])

        if not config['clean'] and config['training_dir'] is not None:
            model.load_weights(config['training_dir'])

    if model:
        model = unet_compile(model, learning_rate=config['learning_rate'])
    

    return model


def make_image_tensor(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Adapted from https://github.com/lanpa/tensorboard-pytorch/
    """
    if len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    else:
        height, width = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)

def graph_history(epochs,loss,val_loss,savedmodelpath):
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig('{}/training.svg'.format(savedmodelpath))

class TensorboardWriter:

    def __init__(self, outdir):
        assert (os.path.isdir(outdir))
        self.outdir = outdir
        self.writer = tf.summary.create_file_writer(self.outdir, flush_millis=10000)

    def save_image(self, tag, image, global_step=None):
        image_tensor = make_image_tensor(image)
        with self.writer.as_default():
            tf.Summary(value=[tf.Summary.Value(tag=tag, image=image_tensor)])

    def close(self):
        """
        To be called in the end
        """
        self.writer.close()

def main(args):

    print('Start training')

    creds = {}
    with open(args.credentails) as json_file:
        creds = json.load(json_file)
    if not creds:
        print('Failed to load credentials file {}. Exiting'.format(args.credentails))

    s3def = creds['s3'][0]
    s3 = s3store(s3def['address'], s3def['access key'], s3def['secret key'])

    if args.clean:
        shutil.rmtree(config['training_dir'], ignore_errors=True)
    
    trainingset = '{}/{}/'.format(s3def['sets']['trainingset']['prefix'] , args.trainingset)
    print('Load training set {}/{} to {}'.format(s3def['sets']['trainingset']['bucket'],trainingset,args.trainingset_dir ))
    s3.Mirror(s3def['sets']['trainingset']['bucket'], trainingset, args.trainingset_dir)

    if args.weights is not None and args.weights.lower() == 'none' or args.weights == '':
        args.weights = None

    trainingsetDescriptionFile = '{}/description.json'.format(args.trainingset_dir)
    trainingsetDescription = json.load(open(trainingsetDescriptionFile))

    config = {
        'batch_size': args.batch_size,
        'traningset': trainingset,
        'trainingset description': trainingsetDescription,
        'input_shape': [args.training_crop[0], args.training_crop[1], args.train_depth],
        'classScale': 0.001, # scale value for each product class
        'augment_rotation' : 15., # Rotation in degrees
        'augment_flip_x': False,
        'augment_flip_y': True,
        'augment_brightness':0.,
        'augment_contrast': 0.,
        'augment_shift_x': 0.1, # in fraction of image
        'augment_shift_y': 0.1, # in fraction of image
        'scale_min': 0.5, # in fraction of image
        'scale_max': 2.0, # in fraction of image
        'ignore_label': trainingsetDescription['classes']['ignore'],
        'classes': trainingsetDescription['classes']['classes'],
        'epochs': args.epochs,
        'area_filter_min': 25,
        'learning_rate': args.learning_rate,
        'weights': args.weights,
        'channel_order': args.channel_order,
        'clean': args.clean,
        's3_address':s3def['address'],
        'mllibsets':s3def['sets'],
        'initialmodel':args.initialmodel,
        'training_dir': args.training_dir,
    }

    if args.trainingset is None or len(args.trainingset) == 0:
        config['trainingset'] = None
    if len(args.initialmodel) == 0:
        config['initialmodel'] = None
    if args.training_dir is None or len(args.training_dir) == 0:
        config['training_dir'] = tempfile.TemporaryDirectory(prefix='train', dir='.')


    strategy = None
    if(args.strategy == 'mirrored'):
        strategy = tf.distribute.MirroredStrategy(devices=args.devices)

    else:
        device = "/gpu:0"
        if args.devices is not None and len(args.devices > 0):
            device = args.devices[0]

        strategy = tf.distribute.OneDeviceStrategy(device=device)

    print('{} distribute with {} GPUs'.format(args.strategy,strategy.num_replicas_in_sync))

    savedmodelpath = '{}/{}'.format(args.savedmodel, args.savedmodelname)
    if not os.path.exists(savedmodelpath):
        os.makedirs(savedmodelpath)
    if not os.path.exists(config['training_dir']):
        os.makedirs(config['training_dir'])

    # ## Train the model
    # Now, all that is left to do is to compile and train the model. The loss being used here is `losses.SparseCategoricalCrossentropy(from_logits=True)`. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and `losses.SparseCategoricalCrossentropy(from_logits=True)` is the recommended loss for 
    # such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.

    with strategy.scope(): # Apply training strategy 
        model =  LoadModel(config, s3) 

        # Display model
        model.summary()

        train_dataset = input_fn('train', args.trainingset_dir, config)
        val_dataset = input_fn('val', args.trainingset_dir, config)

        #earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=3, verbose=0, mode='auto')
        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=config['training_dir'], monitor='loss',verbose=0,save_weights_only=False,save_freq='epoch')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config['training_dir'], histogram_freq=100)
        callbacks = [
            save_callback,
            tensorboard_callback
        ]
        #file_writer = tf.summary.create_file_writer(config['training_dir'])

        # Save plot of model model
        # Failing with "AttributeError: 'dict' object has no attribute 'name'" when returning multiple outputs 
        #tf.keras.utils.plot_model(model, to_file='{}unet.png'.format(savedmodelpath), show_shapes=True)

        train_images = config['batch_size'] # Guess training set if not provided
        val_images = config['batch_size']

        for dataset in trainingsetDescription['sets']:
            if(dataset['name']=="train"):
                train_images = dataset["length"]
            if(dataset['name']=="val"):
                val_images = dataset["length"]
        steps_per_epoch=int(train_images/config['batch_size'])        
        validation_steps=int(val_images/config['batch_size'])
              
        if(args.min):
            steps_per_epoch= min(5, steps_per_epoch)
            validation_steps=min(5, validation_steps)
            config['epochs'] = 1


        if config['epochs'] > 0:

            print("Fit model to data")
            model_history = model.fit(train_dataset, 
                                    validation_data=val_dataset,
                                    epochs=config['epochs'],
                                    steps_per_epoch=steps_per_epoch,
                                    validation_steps=validation_steps,
                                    callbacks=callbacks)

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

            '''if args.prune_epochs > 0:
                end_step = int(math.ceil(train_images / args.batch_size)) * args.prune_epochs
                pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                                    initial_sparsity=0.0, final_sparsity=0.5,
                                    begin_step=0, end_step=end_step)

                def pruning_layers(layer):
                    layers_to_prune=['sequential_3','concatenate_3']
                    if layer.name in layers_to_prune:
                        return tfmot.sparsity.keras.prune_low_magnitude(layer, pruning_schedule=pruning_schedule)
                    return layer

                # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
                # to the layers of the model.
                pruning_model = tf.keras.models.clone_model(
                    model,
                    clone_function=pruning_layers,
                )

                #pruning_model = tfmot.sparsity.keras.prune_low_magnitude(
                #    model, pruning_schedule=pruning_schedule)

                #pruning_model = unet_compile(pruning_model, learning_rate=config['learning_rate'])
                pruning_model.summary()

                callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
                #callbacks.append(tfmot.sparsity.keras.PruningSummaries(log_dir=config['training_dir']))
                prune_history = pruning_model.fit(train_dataset, epochs=args.prune_epochs,
                                        steps_per_epoch=int(train_images/config['batch_size']),
                                        validation_steps=VALIDATION_STEPS,
                                        validation_data=val_dataset,
                                        callbacks=callbacks)'''

        else:
            model_description = {'config':config,
                            }

    print("Create saved model")
    model.save(savedmodelpath, save_format='tf')
    WriteDictJson(model_description, '{}/description.json'.format(savedmodelpath))

    # Make some predictions. In the interest of saving time, the number of epochs was kept small, but you may set this higher to achieve more accurate results.
    WritePredictions(train_dataset, model, config, outpath=savedmodelpath, imgname='train_img')
    WritePredictions(val_dataset, model, config, outpath=savedmodelpath, imgname='val_img')

    # Kubeflow Pipeline results
    results = model_description
    WriteDictJson(results, '{}/results.json'.format(savedmodelpath))

    saved_name = '{}/{}'.format(s3def['sets']['model']['prefix'] , args.savedmodelname)
    print('Save model to {}/{}'.format(s3def['sets']['model']['bucket'],saved_name))
    if s3.PutDir(s3def['sets']['model']['bucket'], savedmodelpath, saved_name):
        shutil.rmtree(savedmodelpath, ignore_errors=True)

    if args.clean or args.training_dir is None or len(args.training_dir) == 0:
        shutil.rmtree(config['training_dir'], ignore_errors=True)

    print("Segmentation training complete. Results saved to http://{}/minio/{}/{}".format(s3def['address'], s3def['sets']['model']['bucket'],saved_name))


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
