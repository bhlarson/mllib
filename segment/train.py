# Based on https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb

import argparse
import json
import os
import sys
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('')), '..')))
sys.path.insert(0, os.path.abspath(''))
from segment.data import input_fn
from segment.display import WritePredictions
from networks.unet import unet_model


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('--dataset_dir', type=str, default='./dataset',help='Directory to store training model')
parser.add_argument('--saveonly', action='store_true', help='Do not train.  Only produce saved model')

parser.add_argument('--record_dir', type=str, default='record', help='Path training set tfrecord')
parser.add_argument('--model_dir', type=str, default='./trainings/unet',help='Directory to store training model')
parser.add_argument('--loadsavedmodel', type=str, default='./saved_model/2020-06-27-14-40-22-dl3', help='Saved model to load if no checkpoint')

parser.add_argument('--clean_model_dir', type=bool, default=True,
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=2,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--batch_size', type=int, default=8, help='Number of examples per batch.')
parser.add_argument('--crops', type=int, default=1, help='Crops/image/step')                

parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='Adam optimizer learning rate.')

parser.add_argument("--strategy", type=str, default='onedevice', help="Replication strategy. 'mirrored', 'onedevice' now supported ")
parser.add_argument("--devices", type=json.loads, default=["/gpu:0"],  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')

parser.add_argument('--training_crop', type=json.loads, default='[720, 960]', help='Training crop size [height, width]')
parser.add_argument('--train_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB') 

defaultfinalmodelname = '{}-dl3'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
parser.add_argument('--finalmodel', type=str, default=defaultfinalmodelname, help='Final model')

parser.add_argument('--savedmodel', type=str, default='./saved_model', help='Path to fcn savedmodel.')
defaultsavemodelname = '{}-dl3'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
parser.add_argument('--savedmodelname', type=str, default=defaultsavemodelname, help='Final model')
parser.add_argument('--tbport', type=int, default=6006, help='Tensorboard network port.')

def WriteDictJson(outdict, path):

    jsonStr = json.dumps(outdict, sort_keys=False)
    f = open(path,"w")
    f.write(jsonStr)
    f.close()
       
    return True

def main(unparsed):
    trainingsetDescriptionFile = '{}/description.json'.format(FLAGS.record_dir)
    trainingsetDescription = json.load(open(trainingsetDescriptionFile))

    config = {
        'batch_size': FLAGS.batch_size,
        'trainingset': trainingsetDescription,
        'input_shape': [FLAGS.training_crop[0], FLAGS.training_crop[1], FLAGS.train_depth],
        'classScale': 0.001, # scale value for each product class
        'augment_rotation' : 5., # Rotation in degrees
        'augment_flip_x': False,
        'augment_flip_y': True,
        'augment_brightness':0.,
        'augment_contrast': 0.,
        'augment_shift_x': 0.0, # in fraction of image
        'augment_shift_y': 0.0, # in fraction of image
        'scale_min': 0.75, # in fraction of image
        'scale_max': 1.25, # in fraction of image
        'ignore_label': trainingsetDescription['classes']['ignore'],
        'classes': trainingsetDescription['classes']['classes'],
        'image_crops': FLAGS.crops,
        'epochs': FLAGS.epochs,
        'area_filter_min': 25,
        }

    # ## Train the model
    # Now, all that is left to do is to compile and train the model. The loss being used here is `losses.SparseCategoricalCrossentropy(from_logits=True)`. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and `losses.SparseCategoricalCrossentropy(from_logits=True)` is the recommended loss for 
    # such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.

    model = None 
    if FLAGS.model_dir:
        try:
            model = tf.keras.models.load_model(FLAGS.model_dir) # Load from checkpoint
        except:
            print('Unable to load weghts from {}'.format(FLAGS.model_dir))

    if not model and FLAGS.loadsavedmodel:
        try:
            model = tf.keras.models.load_model(FLAGS.loadsavedmodel) # Load from checkpoint
        except:
            print('Unable to load weghts from {}'.format(FLAGS.loadsavedmodel))

    if not model:
        print('Unable to load weghts.  Restart training.')
        model = unet_model(config['classes'], config.input_shape)
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    # Display model
    model.summary()


    train_dataset = input_fn(True, FLAGS.record_dir, config)
    test_dataset = input_fn(False, FLAGS.record_dir, config)

    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.model_dir,verbose=1,save_weights_only=False,save_freq='epoch',period=1)

    outpath = '{}/{}/'.format(FLAGS.savedmodel, FLAGS.savedmodelname)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Save plot of model model
    tf.keras.utils.plot_model(model, to_file='{}unet.png'.format(outpath), show_shapes=True)

    VALIDATION_STEPS = 100
    if not FLAGS.saveonly:
        model_history = model.fit(train_dataset, epochs=config['epochs'],
                                  steps_per_epoch=int(trainingsetDescription['sets'][0]['length']/config['batch_size']),
                                  validation_steps=VALIDATION_STEPS,
                                  validation_data=test_dataset,
                                  callbacks=[save_callback])

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

        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'bo', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig('{}training.svg'.format(outpath))

    else:
        model_description = {'config':config,
                          }

    WriteDictJson(model_description, '{}descrption.json'.format(outpath))

    model.save(outpath)

    # Let's make some predictions. In the interest of saving time, the number of epochs was kept small, but you may set this higher to achieve more accurate results.
    WritePredictions(test_dataset, model, config, outpath=outpath)
    print("Segmentation training complete. Results saved to {}".format(outpath))


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  
  if FLAGS.debug:
      print("Wait for debugger attach")
      import ptvsd
      # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
      # Launch applicaiton on remote computer: 
      # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait fcn/train.py
      # Allow other computers to attach to ptvsd at this IP address and port.
      ptvsd.enable_attach(address=('0.0.0.0', 3000), redirect_output=True)
      # Pause the program until a remote debugger is attached

      ptvsd.wait_for_attach()

      print("Debugger attach")

  main(unparsed)
