# Based on https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb

import argparse
import json
import os
import sys
import math
import shutil
import cv2
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

DEBUG = False


#disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-saveonly', action='store_true', help='Do not train.  Only produce saved model')
parser.add_argument('-tflite', action='store_true',help='Run tensorflow lite postprocessing')
parser.add_argument('-trt', action='store_true',help='Run TensorRT postprocessing')
parser.add_argument('-onnx', action='store_true',help='Run ONNX postprocessing')
parser.add_argument('-clean_model_dir', action='store_true', help='If set, delete model directory at startup.')

parser.add_argument('-dataset_dir', type=str, default='./dataset',help='Directory to store training model')
parser.add_argument('-model_precision', type=str, default='FP16', choices=['FP32', 'FP16', 'INT8'], help='Model Optimization Precision.')
parser.add_argument('-chanel_order', type=str, default='channels_first', choices=['channels_first' or 'channels_last'], help='Model Optimization Precision.')

parser.add_argument('-record_dir', type=str, default='cocorecord', help='Path training set tfrecord')
#parser.add_argument('-record_dir', type=str, default='cityrecord', help='Path training set tfrecord')
parser.add_argument('-model_dir', type=str, default='./trainings/unetcoco',help='Directory to store training model')
#parser.add_argument('-model_dir', type=str, default='./trainings/unetcity',help='Directory to store training model')
#parser.add_argument('-loadsavedmodel', type=str, default='./saved_model/2020-09-30-23-04-02-dl3', help='Saved model to load if no checkpoint')
parser.add_argument('-loadsavedmodel', type=str, default=None, help='Saved model to load if no checkpoint')



parser.add_argument('-epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('-prune_epochs', type=int, default=0, help='Number of pruning epochs')

parser.add_argument('-tensorboard_images_max_outputs', type=int, default=2,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('-batch_size', type=int, default=32, help='Number of examples per batch.')
parser.add_argument('-crops', type=int, default=1, help='Crops/image/step')                

parser.add_argument('-learning_rate', type=float, default=1e-3,
                    help='Adam optimizer learning rate.')

parser.add_argument("-strategy", type=str, default='onedevice', help="Replication strategy. 'mirrored', 'onedevice' now supported ")
parser.add_argument("-devices", type=json.loads, default=["/gpu:0"],  help='GPUs to include for training.  e.g. None for all, [/cpu:0], ["/gpu:0", "/gpu:1"]')

parser.add_argument('-training_crop', type=json.loads, default='[480, 512]', help='Training crop size [height, width]')
parser.add_argument('-train_depth', type=int, default=3, help='Number of input colors.  1 for grayscale, 3 for RGB') 

defaultfinalmodelname = '{}-dl3'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
parser.add_argument('-finalmodel', type=str, default=defaultfinalmodelname, help='Final model')

parser.add_argument('-savedmodel', type=str, default='./saved_model', help='Path to savedmodel.')
parser.add_argument('-onxdir', type=str, default='./onnx', help='Path to ONNX models.')
defaultsavemodelname = '{}-dl3'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
parser.add_argument('-savedmodelname', type=str, default=defaultsavemodelname, help='Final model')
parser.add_argument('-trtmodel', type=str, default='./trt', help='Path to TensorRT.')
parser.add_argument('-resultspath', type=str, default='/', help='Kubeflow pipeline model output')
parser.add_argument('-tbport', type=int, default=6006, help='Tensorboard network port.')
parser.add_argument('-weights', type=str, default='imagenet', help='Model initiation weights. None prevens loading weights from pre-trained networks')

def WriteDictJson(outdict, path):

    jsonStr = json.dumps(outdict, sort_keys=False)
    f = open(path,"w")
    f.write(jsonStr)
    f.close()
       
    return True

def LoadModel(config):
    model = None 

    if FLAGS.loadsavedmodel is not None:
        try:
            model = tf.keras.models.load_model(FLAGS.loadsavedmodel, compile=False) # Load from checkpoint

        except:
            print('Unable to load weghts from {}'.format(FLAGS.loadsavedmodel))
            model = None 

    if model is None:
        print('Unable to load weghts.  Restart training.')
        model = unet_model(classes=config['classes'], input_shape=config['input_shape'], learning_rate=config['learning_rate'], weights=config['weights'], chanel_order=config['chanel_order'])

        if FLAGS.model_dir:
            try:
                model.load_weights(FLAGS.model_dir)
            except:
                print('Unable to load weghts from {}'.format(FLAGS.model_dir))

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

class ImageWriterCallback(Callback):

    def __init__(self, config):
        self.config = config

    @tf.function(autograph=not DEBUG)
    def on_train_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

        img = tf.keras.preprocessing.image.array_to_img(self.model.inputs[0])
        array = tf.keras.preprocessing.image.img_to_array(img)

        img = self.model.inputs[0].numpy()
        seg = tf.argmax(self.model.outputs[0], axis=-1, name='argmax').numpy()

        [batch,_,_,_] = img.shape()
        for i in range(batch):
            iman = DrawFeatures(img[i], seg[i], self.config)

            iman = cv2.cvtColor(iman, cv2.COLOR_RGB2BGR)
            cv2.imwrite('{}/pred{}{}.png'.format(FLAGS.model_dir, batch, i), iman)

    def on_test_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )


def main(unparsed):

    if FLAGS.loadsavedmodel is not None and FLAGS.loadsavedmodel.lower() == 'none' or FLAGS.weights == '':
        FLAGS.loadsavedmodel = None

    if FLAGS.weights is not None and FLAGS.weights.lower() == 'none' or FLAGS.weights == '':
        FLAGS.weights = None  

    trainingsetDescriptionFile = '{}/description.json'.format(FLAGS.record_dir)
    trainingsetDescription = json.load(open(trainingsetDescriptionFile))

    config = {
        'batch_size': FLAGS.batch_size,
        'trainingset': trainingsetDescription,
        'input_shape': [FLAGS.training_crop[0], FLAGS.training_crop[1], FLAGS.train_depth],
        'classScale': 0.001, # scale value for each product class
        'augment_rotation' : 0., # Rotation in degrees
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
        'image_crops': FLAGS.crops,
        'epochs': FLAGS.epochs,
        'area_filter_min': 25,
        'learning_rate': FLAGS.learning_rate,
        'weights': FLAGS.weights,
        'chanel_order': FLAGS.chanel_order
        }

    if FLAGS.clean_model_dir:
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

    savedmodelpath = '{}/{}/'.format(FLAGS.savedmodel, FLAGS.savedmodelname)
    if not os.path.exists(savedmodelpath):
        os.makedirs(savedmodelpath)
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    # ## Train the model
    # Now, all that is left to do is to compile and train the model. The loss being used here is `losses.SparseCategoricalCrossentropy(from_logits=True)`. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and `losses.SparseCategoricalCrossentropy(from_logits=True)` is the recommended loss for 
    # such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.

    model =  LoadModel(config) 

    # Display model
    model.summary()

    train_dataset = input_fn('train', FLAGS.record_dir, config)
    val_dataset = input_fn('val', FLAGS.record_dir, config)

    #earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=3, verbose=0, mode='auto')
    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.model_dir, monitor='loss',verbose=0,save_weights_only=False,save_freq='epoch')
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.model_dir, histogram_freq=100)
    callbacks = [
        #earlystop_callback,
        save_callback,
        #tensorboard_callback
        #tf.keras.callbacks.TensorBoard(FLAGS.model_dir)
        #,ImageWriterCallback(config)]
    ]
    #file_writer = tf.summary.create_file_writer(FLAGS.model_dir)

    # Save plot of model model
    # Failing with "AttributeError: 'dict' object has no attribute 'name'" when returning multiple outputs 
    #tf.keras.utils.plot_model(model, to_file='{}unet.png'.format(savedmodelpath), show_shapes=True)

    train_images = 5000 # Guess training set if not provided
    for dataset in trainingsetDescription['sets']:
        if(dataset['name']=="train"):
            train_images = dataset["length"]


    VALIDATION_STEPS = 100
    if not FLAGS.saveonly:
        if config['epochs'] > 0:
            model_history = model.fit(train_dataset, epochs=config['epochs'],
                                    steps_per_epoch=int(train_images/config['batch_size']),
                                    validation_steps=VALIDATION_STEPS,
                                    validation_data=val_dataset,
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

            plt.figure()
            plt.plot(epochs, loss, 'r', label='Training loss')
            plt.plot(epochs, val_loss, 'bo', label='Validation loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.ylim([0, 1])
            plt.legend()
            plt.savefig('{}training.svg'.format(savedmodelpath))

        '''if FLAGS.prune_epochs > 0:
            end_step = int(math.ceil(train_images*FLAGS.crops / FLAGS.batch_size)) * FLAGS.prune_epochs
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
            #callbacks.append(tfmot.sparsity.keras.PruningSummaries(log_dir=FLAGS.model_dir))
            prune_history = pruning_model.fit(train_dataset, epochs=FLAGS.prune_epochs,
                                    steps_per_epoch=int(train_images/config['batch_size']),
                                    validation_steps=VALIDATION_STEPS,
                                    validation_data=val_dataset,
                                    callbacks=callbacks)'''

    else:
        model_description = {'config':config,
                          }

    model.save(savedmodelpath)
    WriteDictJson(model_description, '{}description.json'.format(savedmodelpath))

    # Make some predictions. In the interest of saving time, the number of epochs was kept small, but you may set this higher to achieve more accurate results.
    WritePredictions(val_dataset, model, config, outpath=savedmodelpath)

    # Kubeflow Pipeline results
    results = model_description
    WriteDictJson(results, '{}/results.json'.format(savedmodelpath))

    if FLAGS.tflite: # convert to Tensorflow Lite
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodelpath)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        #samplefiles = get_samples(FLAGS.sample_dir, FLAGS.match)
        #converter.representative_dataset = lambda:representative_dataset_gen(samplefiles)
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #converter.inference_input_type = tf.uint8  # or tf.uint8
        #converter.inference_output_type = tf.uint8  # or tf.uint8
        tflite_model = converter.convert()
        outflite = '{}{}.tflite'.format(savedmodelpath,FLAGS.savedmodelname)
        open(outflite, "wb").write(tflite_model)

    if FLAGS.trt: # convert to TensorRT
        trtpath = '{}/{}-{}/'.format(FLAGS.trtmodel, FLAGS.savedmodelname, FLAGS.model_precision)
        from tensorflow.python.compiler.tensorrt import trt_convert as trt

        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        conversion_params = conversion_params._replace(precision_mode=FLAGS.model_precision)

        converter = trt.TrtGraphConverterV2(input_saved_model_dir=savedmodelpath, conversion_params=conversion_params)
        converter.convert()
        converter.save(trtpath)

        # How do I produce and output a TensorRT .play file with TensorRT
        # https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#worflow-with-savedmodel
        # create_inference_graph has been removed from Tensorflow 2
        '''def input_fn():
            for _ in range(num_runs):
                inputTensor = np.random.normal(size=(1, FLAGS.training_crop[0], FLAGS.training_crop[1], FLAGS.train_depth)).astype(np.float32)
                yield inputTensor
        converter.build(input_fn=input_fn)

        for n in trt_graph.node:
            if n.op == "TRTEngineOp":
                print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
                with tf.gfile.GFile("%s.plan" % (n.name.replace("/", "_")), 'wb') as f:
                f.write(n.attr["serialized_segment"].s)
            else:
                print("Exclude Node: %s, %s" % (n.op, n.name.replace("/", "_")))'''

    if FLAGS.onnx:
        import keras2onnx

        onnx_name = '{}{}.onnx'.format(savedmodelpath, FLAGS.savedmodelname)
        onnx_model = keras2onnx.convert_keras(model, model.name, debug_mode=True)
        content = onnx_model.SerializeToString()
        keras2onnx.save_model(onnx_model, onnx_name)

    print("Segmentation training complete. Results saved to {}".format(savedmodelpath))


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

      print("Debugger attached")

  main(unparsed)
