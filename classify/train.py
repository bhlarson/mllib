import argparse
import json
import os
import sys
import shutil
import tempfile
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from datetime import datetime
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append('utils')
from s3 import s3store
from jsonutil import WriteDictJson
from data import input_fn

print('Python Version {}'.format(sys.version))
print('Tensorflow version {}'.format(tf.__version__))
print('GPU Available: {}'.format(tf.test.is_gpu_available()))
if(tf.test.is_gpu_available()):
  print('GPU Devices: {}'.format(tf.test.gpu_device_name()))

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debugger attach')
parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
parser.add_argument('-clean', action='store_true', help='If set, delete model directory at startup.')
parser.add_argument('-min', action='store_true', help='If set, minimum training to generate output.')
parser.add_argument('-min_steps', type=int, default=3, help='Minimum steps')

parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
parser.add_argument('-batch_size', type=int, default=32, help='Number of examples per batch.')
parser.add_argument('-size_x', type=int, default=224, help='Training image size_x')
parser.add_argument('-size_y', type=int, default=224, help='Training image size_y')
parser.add_argument('-depth', type=int, default=3, help='Training image depth')
parser.add_argument('-channel_order', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Channels_last = NHWC, Tensorflow default, channels_first=NCHW')
parser.add_argument('-epochs', type=int, default=20, help='Training epochs')
parser.add_argument('-model_dir', type=str, default='./trainings/resnet-classify',help='Directory to store training model')
parser.add_argument('-savedmodel', type=str, default='./saved_model', help='Path to savedmodel.')
parser.add_argument('-training_dir', type=str, default='./trainings/classify',help='Training directory.  Empty string for auto-generated tempory directory')
parser.add_argument('--trainingset', type=str, default='2021-01-12-08-26-56-cocoseg', help='training set')
parser.add_argument('-trainingset_dir', type=str, default='/store/training/2021-01-12-08-26-56-cocoseg', help='Path training set tfrecord')

parser.add_argument('--initialmodel', type=str, default='', help='Initial model.  Empty string if no initial model')

parser.add_argument('-learning_rate', type=float, default=1e-3, help='Adam optimizer learning rate.')
parser.add_argument('-dataset', type=str, default='tf_flowers', choices=['tf_flowers'], help='Model Optimization Precision.')

parser.add_argument("-strategy", type=str, default='mirrored', help="Replication strategy. 'mirrored', 'onedevice' now supported ")
parser.add_argument("-devices", type=json.loads, default=None,  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')
defaultsavemodeldir = '{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-cfy'))
parser.add_argument('-savedmodelname', type=str, default=defaultsavemodeldir, help='Final model')
parser.add_argument('-weights', type=str, default='imagenet', help='Model initiation weights. None prevens loading weights from pre-trained networks')
parser.add_argument('-description', type=str, default='train UNET segmentation network', help='Describe training experament')

def LoadModel(config, s3, model_dir=None):
    model = None 
    print('LoadModel initial model: {}, training directory: {}, '.format(config['initialmodel'], config['training_dir']))
    if config['initialmodel'] is not None:
        tempinitmodel = tempfile.TemporaryDirectory(prefix='initmodel', dir='.')
        modelpath = tempinitmodel.name+'/'+config['initialmodel']
        os.makedirs(modelpath)
        try:
            s3model=config['s3_sets']['model']['prefix']+'/'+config['initialmodel']
            success = s3.GetDir(config['s3_sets']['model']['bucket'], s3model, modelpath)
            model = tf.keras.models.load_model(modelpath) # Load from checkpoint

        except:
            print('Unable to load weghts from http://{}/minio/{}/{}'.format(
                config['s3_address'],
                config['s3_sets']['model']['prefix'],
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

        model = keras.applications.ResNet50V2(include_top=True, weights=config['init_weights'], 
            input_shape=config['shape'], classes=config['classes'], classifier_activation=None)

        if not config['clean'] and config['training_dir'] is not None:
            try:
                model.load_weights(config['training_dir'])
            except:
                print('Unable to load weghts from {}'.format(config['training_dir']))

    if model:
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return model

def prepare_image(image, label, config):
    image = tf.image.resize_with_crop_or_pad(image, config['shape'][0], config['shape'][1])
    return image, label

'''def input_fn(config, split):
    dataset, metadata = tfds.load('tf_flowers', with_info=True, split=split, shuffle_files=True, as_supervised=True)
    dataset = dataset.map(lambda features, label: prepare_image(features, label, config) , num_parallel_calls = 10)
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.prefetch(config['batch_size'])

    dataset = dataset.shuffle(buffer_size=10*config['batch_size'])
    dataset = dataset.repeat(config['epochs'])

    dataset = dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # Test new autotune prefetch 
    return dataset, metadata'''

def graph_history(loss,val_loss,savedmodelpath):
    
    plt.figure()
    if loss:
        i_loss = range(1,len(loss)+1,1)
        plt.plot(i_loss, loss, 'r', label='Training loss')
    if val_loss:
        i_val_loss = range(1,len(val_loss)+1,1)
        plt.plot(i_val_loss, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.savefig('{}/training.svg'.format(savedmodelpath))

def PrepareInference(dataset, model):
    for image, label in dataset.take(1):
        logits = model.predict(image) 

def CreatePredictions(dataset, model, config, outpath, imgname, num=1):
    i = 0
    dtSum = 0.0
    for image, label in dataset.take(num):
      initial = datetime.now()
      logits = model.predict(image)
      classification = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
      dt = (datetime.now()-initial).total_seconds()
      dtSum += dt
      for j in range(config['batch_size']):
        filename = '{}/{}{}.png'.format(outpath, imgname, i)
        ax = plt.imshow(image[j])
        plt.autoscale(True)
        plt.axis('off')
        plt.gca().get_legend().set_visible(False)
        plt.tight_layout()
        datastr = 'label:{}, pred:{} conf:{:.4f}, dt:{:.4f}'.format(
            label[j].numpy(),
            classification[j].numpy(),
            logits[j][classification[j].numpy()],
            dt/config['batch_size'])
        plt.title(datastr)
        plt.savefig(filename)
        i=i+1

        print (datastr)
    print ("Average time {}".format(dtSum/i) )

def main(args):
    #tf.config.experimental_run_functions_eagerly(False)
    print('Start training')

    creds = {}
    with open(args.credentails) as json_file:
        creds = json.load(json_file)
    if not creds:
        print('Failed to load credentials file {}. Exiting'.format(args.credentails))

    s3def = creds['s3'][0]
    s3 = s3store(s3def['address'], 
                 s3def['access key'], 
                 s3def['secret key'], 
                 tls=s3def['tls'], 
                 cert_verify=s3def['cert_verify'], 
                 cert_path=s3def['cert_path']
                 )

    trainingset = '{}/{}/'.format(s3def['sets']['trainingset']['prefix'] , args.trainingset)
    print('Load training set {}/{} to {}'.format(s3def['sets']['trainingset']['bucket'],trainingset,args.trainingset_dir ))
    s3.Mirror(s3def['sets']['trainingset']['bucket'], trainingset, args.trainingset_dir)

    trainingsetDescriptionFile = '{}/description.json'.format(args.trainingset_dir)
    trainingsetDescription = json.load(open(trainingsetDescriptionFile))

    training_percent = 0.8
    config = {
        'descripiton': args.description,
        'traningset': trainingset,
        'trainingset description': trainingsetDescription,
        'dataset':args.dataset,
        'batch_size': args.batch_size,
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
        'input_shape': [args.size_y, args.size_x, args.depth],
        'shape': (args.size_y, args.size_x, args.depth),
        'split': tfds.Split.TRAIN,
        'classes': trainingsetDescription['classes']['classes'],
        'learning_rate': args.learning_rate,
        'init_weights':None,
        'clean': args.clean,
        'epochs':args.epochs,
        'training_percent':training_percent,
        'training':'train[:{}%]'.format(int(100*training_percent)),
        'validation':'train[{}%:]'.format(int(100*training_percent)),
        'channel_order': args.channel_order,
        #'training':'train[:80%]',
        #'validation':'train[80%:]',
        's3_address':s3def['address'],
        's3_sets':s3def['sets'],
        'initialmodel':args.initialmodel,
        'training_dir': args.training_dir,
    }

    if len(args.initialmodel) == 0:
        config['initialmodel'] = None
    if args.training_dir is None or len(args.training_dir) == 0:
        config['training_dir'] = tempfile.TemporaryDirectory(prefix='train', dir='.')

    if args.clean:
        shutil.rmtree(config['training_dir'], ignore_errors=True)


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

    with strategy.scope():
        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.model_dir, monitor='loss',verbose=0,save_weights_only=False,save_freq='epoch')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.model_dir, histogram_freq='epoch')
        callbacks = [
            #save_callback,
            #tensorboard_callback
        ]

        #train_dataset, datasetdata = input_fn(config, split=config['training'])
        #val_dataset, _ = input_fn(config, split=config['validation'])

        train_dataset = input_fn('train', args.trainingset_dir, config)
        val_dataset = input_fn('val', args.trainingset_dir, config)

        #for images, labels in train_dataset.take(1):
        #    for i in range(images.shape[0]):
        #        print(labels[i].numpy())

        # config['classes'] = datasetdata.features['label'].num_classes
        #train_images = int(datasetdata.splits.total_num_examples*config['training_percent'])
        #val_images = int(datasetdata.splits.total_num_examples*(1.0-config['training_percent']))

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
            steps_per_epoch= min(args.min_steps, steps_per_epoch)
            validation_steps=min(args.min_steps, validation_steps)
            config['epochs'] = 1        
        
        model =  LoadModel(config, s3, args.model_dir)

        # Display model
        model.summary()

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
        #epochs = config['epochs']

        graph_history(loss,val_loss,savedmodelpath)

    print("Create saved model")
    model.save(savedmodelpath, save_format='tf')
    
    PrepareInference(dataset=train_dataset, model=model)
    CreatePredictions(dataset=train_dataset, model=model, config=config, outpath=savedmodelpath, imgname='train')
    CreatePredictions(dataset=val_dataset, model=model, config=config, outpath=savedmodelpath, imgname='val')

    model_description = {'config':config,
                         'results': history
                        }
    WriteDictJson(model_description, '{}/description.json'.format(savedmodelpath))

    # Save confusion matrix: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    saved_name = '{}/{}'.format(s3def['sets']['model']['prefix'] , args.savedmodelname)
    print('Save model to {}/{}'.format(s3def['sets']['model']['bucket'],saved_name))
    #if s3.PutDir(s3def['sets']['model']['bucket'], savedmodelpath, saved_name):
    #    shutil.rmtree(savedmodelpath, ignore_errors=True)

    if args.clean or args.training_dir is None or len(args.training_dir) == 0:
        shutil.rmtree(config['training_dir'], ignore_errors=True)

    #print("Classification training complete. Results saved to http://{}/minio/{}/{}".format(s3def['address'], s3def['sets']['model']['bucket'],saved_name))


if __name__ == '__main__':
  args, unparsed = parser.parse_known_args()
  
  if args.debug:
      print("Wait for debugger attach")
      import ptvsd
      # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
      # Launch applicaiton on remote computer: 
      # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait fcn/train.py
      # Allow other computers to attach to ptvsd at this IP address and port.
      ptvsd.enable_attach(address=('0.0.0.0', args.debug_port), redirect_output=True)
      # Pause the program until a remote debugger is attached

      ptvsd.wait_for_attach()

      print("Debugger attached")

  main(args)