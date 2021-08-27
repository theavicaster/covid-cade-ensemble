train_dir = '/content/data-folders/train'
test_dir = '/content/data-folders/test'

import numpy as np
import os
from glob import glob
import math
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Flatten, concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Permute
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

data_format = K.image_data_format()
K.set_image_data_format(data_format)
np.random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_of_classes = 3

# Replace with Inception ResNet V2 or DenseNet 201
from tensorflow.keras.applications import InceptionV3

model = InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3),
)

flat = Flatten()(den_model.layers[-1].output)
dense1 = Dense(256, activation='relu')(flat)
drop = Dropout(0.5)(dense1)
dense2 = Dense(256, activation='relu')(drop)
output = Dense(num_of_classes, activation='softmax')(dense2)

from keras.models import Model
model = Model(inputs=den_model.inputs, outputs=output)

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                                   #samplewise_center=True,
                                   #samplewise_std_normalization=True
                                   )
test_datagen = ImageDataGenerator(
                                   #samplewise_center=True,
                                   #samplewise_std_normalization=True
                                 )


training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    seed = 42
   )

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle = False, 
    seed = 42
   )

sgd = optimizers.SGD(learning_rate = 0.01, momentum = 0.9, clipnorm = 1.0)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

checkpoint = ModelCheckpoint(monitor='val_accuracy', verbose=1, filepath="model.h5", save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, min_delta = 0.0005,
                              patience=20, min_lr=0.0001, verbose = 1)
callbacks_list = [checkpoint,reduce_lr]

H = model.fit_generator(
    training_set,
    steps_per_epoch=435,
    epochs=50,
    validation_data = test_set,
    validation_steps = 50,
    callbacks=callbacks_list)