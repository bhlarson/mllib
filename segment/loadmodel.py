
import os
import shutil
import tempfile
import tensorflow as tf
from utils.s3 import s3store
from networks.unet import unet_model, unet_compile

def LoadModel(config, s3):
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
            print('Unable to load weghts from http://{}/minio/{}/{}/'.format(
                config['s3_address'],
                config['s3_sets']['model']['bucket'],
                s3model
            ))
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
            try:
                model.load_weights(config['training_dir'])
            except:
                print('Unable to load training weghts from {}'.format(config['training_dir']))


    if model:
        model = unet_compile(model, learning_rate=config['learning_rate'])
    

    return model