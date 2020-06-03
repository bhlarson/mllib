import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow_datasets as tfds

def model_dfn(num_classes=2, input_shape=(224,224,3), learning_rate=1e-4):
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

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=Adam(lr=learning_rate))
    model.summary()

    return model
