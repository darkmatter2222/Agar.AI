import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time
import json
import math

import datetime

class custom_loss_check(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch Start, saving...')
        self.model.save(f'N:\\Projects\\Agar.AI\\Models\\MultiClassEpochStart.h5')

    def on_epoch_end(self, epoch, logs=None):
        if 'loss' not in logs or math.isnan(logs['loss']):
            print('loss is NaN, stopping early, reverting to head model')
            self.model.stop_training = True
            self.model = tf.keras.models.load_model('N:\\Projects\\Agar.AI\\Models\\MultiClassEpochStart.h5')
        if 'val_loss' not in logs or math.isnan(logs['val_loss']):
            print('val_loss is NaN, stopping early, reverting to head model')
            self.model.stop_training = True
            self.model = tf.keras.models.load_model('N:\\Projects\\Agar.AI\\Models\\MultiClassEpochStart.h5')

# Pull in all data
base_dir = 'N:\\Projects\\Agar.AI\\Training Data'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')

# Pull in all data
base_dir = 'N:\\Projects\\Agar.AI\\Training Data'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')

ratio_multi = 2
target_ratio = (192 * ratio_multi, 97 * ratio_multi)

build_new_model = False

runs = 200

for run in range(0, runs):
    print(f'Run: {run}')
    if build_new_model:
        # Our input feature map is 200x200x3: 200x200 for the image pixels, and 3 for
        # the three color channels: R, G, and B
        img_input = layers.Input(shape=(target_ratio[0], target_ratio[1], 3))

        # First convolution extracts 16 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(16, 3, activation='relu')(img_input)
        x = layers.MaxPooling2D(2)(x)

        # Second convolution extracts 32 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Third convolution extracts 64 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Flatten feature map to a 1-dim tensor so we can add fully connected layers
        x = layers.Flatten()(x)

        # Create a fully connected layer with ReLU activation and 512 hidden units
        x = layers.Dense(512, activation='relu')(x)

        # Create output layer with a single node and sigmoid activation
        output = layers.Dense(37, activation='sigmoid')(x)

        # Create model:
        # input = input feature map
        # output = input feature map + stacked convolution/max pooling layers + fully
        # connected layer + sigmoid output layer
        model = Model(img_input, output)

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=0.001),
                      metrics=['acc'])
    else:
        # Load Model
        model = tf.keras.models.load_model('N:\\Projects\\Agar.AI\\Models\\MultiClassBestCheckpoint.h5')

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            train_dir,  # This is the source directory for training images
            target_size=target_ratio,  # All images will be resized to 200x200
            batch_size=100,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='categorical')

    # Flow validation images in batches of 20 using val_datagen generator
    validation_generator = val_datagen.flow_from_directory(
            validation_dir,
            target_size=target_ratio,
            batch_size=100,
            class_mode='categorical')

    tensor_board = tf.keras.callbacks.TensorBoard(log_dir=os.path.realpath('..') + "\\Logs\\{}".format(time.time()))
    model_save = tf.keras.callbacks.ModelCheckpoint(
        'N:\\Projects\\Agar.AI\\Models\\MultiClassBestCheckpoint.h5',
        monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=2, mode='auto',
                                                  baseline=None, restore_best_weights=False)
    early_term_on_nan = tf.keras.callbacks.TerminateOnNaN()


    classes = train_generator.class_indices
    with open('N:\\Projects\\Agar.AI\\Models\\Classes.json', 'w') as outfile:
        json.dump(classes, outfile)
    print(classes)

    history = model.fit_generator(
          train_generator,
          steps_per_epoch=100,  # 2000 images = batch_size * steps
          epochs=3,
          callbacks=[tensor_board, model_save, early_term_on_nan, custom_loss_check()],
          validation_data=validation_generator,
          validation_steps=40,  # 1000 images = batch_size * steps
          verbose=1)

    model.save(f'N:\\Projects\\Agar.AI\\Models\\MultiClassV1.h5')