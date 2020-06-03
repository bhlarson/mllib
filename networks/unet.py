# TensorFlow and tf.keras
import sys
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./model_fcn',
                    help='Base directory for the model.')
parser.add_argument('--train_dir', type=str, default='/store/Datasets/dogsvscats/train',
                    help='Base directory for the model.')
parser.add_argument('--debug', type=bool, default=False, help='Break for remote debugger.')


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def unet(num_classes=2, pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)

    filters = 64
    conv1 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    filters = 2*filters
    conv2 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    filters = 2*filters
    conv3 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    filters = 2*filters
    conv4 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    filters = 2*filters
    conv5 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    filters = filters/2
    up6 = Conv2D(filters=filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    filters = filters/2
    up7 = Conv2D(filters=filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    filters = filters/2
    up8 = Conv2D(filters=filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    filters = filters/2
    up9 = Conv2D(filters=filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(filters=filters, kernel_size=(3,3) activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(filters=num_classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(filters=num_classes, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def VGG16Classification(num_classes=2, input_shape=(224,224,3), learning_rate=1e-4):
    model = Sequential()
    
    # Convolution layers
    filters = 64
    model.add(Conv2D(input_shape=input_shape, filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    filters = 2*filters
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    filters = 2*filters
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    filters = 2*filters
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    #filters = 2*filters
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=filters, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=num_classes, activation="softmax"))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    model.summary()

    return model

def DogvsCat(path, target_size=(224,224), batch_size=50, validation_split=0.2, horizontal_flip=True, normalize=True, zoom_range=0.1, rotation_range=10, shift_range=0.1):
    trdata = ImageDataGenerator(validation_split=validation_split, 
        horizontal_flip=horizontal_flip, 
        samplewise_center=normalize, 
        samplewise_std_normalization=normalize,
        zoom_range=zoom_range,
        rotation_range=rotation_range,
        width_shift_range=shift_range,
        height_shift_range=shift_range,
        fill_mode="reflect"
        )

    train_generator = trdata.flow_from_directory(
        path, 
        subset='training',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = trdata.flow_from_directory(
        path,
        subset='validation',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, validation_generator

def main(FLAGS):
    print(tf.__version__)

    model = VGG16Classification()

    train_generator, validation_generator = DogvsCat(FLAGS.train_dir)

    history = model.fit_generator(
        train_generator,
        epochs=2,
        validation_data=validation_generator)

    print(history)

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.debug:
        # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        # Launch applicaiton on remote computer: 
        # > python3 -m ptvsd --host 0.0.0.0 --port 3000 --wait predict_imdb.py
        import ptvsd
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=('0.0.0.0', 3000), redirect_output=True)
        # Pause the program until a remote debugger is attached
        print("Wait for debugger attach")
        ptvsd.wait_for_attach()
        print("Debugger Attached")

    main(FLAGS)
    print('complete')