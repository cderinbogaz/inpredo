import os
import sys

# If you like to speed up training process with GPU, first install PlaidML and then uncomment the following line.
# Otherwise it will fallback to tensorflow.

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 10
else:
  epochs = 1000

# BEFORE STARTING TRAINING YOU NEED TO MANUALLY TAKE 20 PERCENENT OF THE TRAINING DATA AND PUT IT INTO VALIDATION FOLDER
# I was too lazy to do it in the code.

train_data_dir = './data/train/'
validation_data_dir = './data/validation/'

# Input the size of your sample images
img_width, img_height = 150, 150
# Enter the number of samples, training + validation
nb_train_samples = 13204
nb_validation_samples = 1412
nb_filters1 = 32
nb_filters2 = 32
nb_filters3 = 64
conv1_size = 3
conv2_size = 2
conv3_size = 5
pool_size = 2
# We have 2 classes, buy and sell
classes_num = 2
batch_size = 128
lr = 0.001
chanDim =3


model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ='same', input_shape=(img_height, img_width , 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Convolution2D(nb_filters3, conv3_size, conv3_size, border_mode ='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.rmsprop(),
                      metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    #rescale=1. / 255,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(
    #rescale=1. / 255,
    horizontal_flip=False)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    #shuffle=True,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    #shuffle=True,
    class_mode='categorical')

"""
Tensorboard log
"""
target_dir = "./models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./src/models/model.h5')
model.save_weights('./src/models/weights.h5')

checkpoint = ModelCheckpoint(target_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    shuffle=True,
    validation_data=validation_generator,
    callbacks=callbacks_list,
    validation_steps=nb_validation_samples//batch_size)


