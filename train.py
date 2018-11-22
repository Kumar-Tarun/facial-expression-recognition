import pandas as pd
import numpy as np
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import utils
from keras import regularizers
from keras.utils import to_categorical
from prepare import get_data, plot

data = get_data()

train_x = data["train_x"]
train_y = data["train_y"]
val_x = data["val_x"]
val_y = data["val_y"]
test_x = data["test_x"]
test_y = data["test_y"]

train1_x = data["train1_x"]
train1_y = data["train1_y"]
val1_x = data["val1_x"]
val1_y = data["val1_y"]
test1_x = data["test1_x"]
test1_y = data["test1_y"]


# Network 1
mod = Sequential()

mod.add(Conv3D(64, kernel_size = (3, 5, 5), strides = (1, 1, 1), padding = 'same', data_format = 'channels_last', input_shape = (6, 64, 64, 1), name = "Input1"))
mod.add(Dropout(0.3))
mod.add(Activation('relu'))
mod.add(MaxPooling3D(pool_size = (2, 2, 2), strides = (1, 2, 2)))
mod.add(Conv3D(64, kernel_size = (3, 5, 5), strides = (1, 1, 1), padding = 'same', data_format = 'channels_last'))
mod.add(Dropout(0.3))
mod.add(Activation('relu'))
mod.add(MaxPooling3D(pool_size = (2, 2, 2), strides = (1, 2, 2)))
mod.add(Flatten())
mod.add(Dense(256, kernel_regularizer = regularizers.l2(0.35)))
mod.add(Dropout(0.5))
mod.add(Activation('relu'))
mod.add(Dense(256, kernel_regularizer = regularizers.l2(0.35)))
mod.add(Dropout(0.55))
mod.add(Activation('relu'))
mod.add(Dense(7, name = "last_layer1"))
mod.add(Activation('softmax', name = "Output1"))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=0.00002, verbose=1)
                              
model_checkpoint = ModelCheckpoint(filepath = 'weights1.hdf5', verbose = 1, save_best_only = True)
                              
optimizer = Adam(lr = 0.0001)

mod.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
history = mod.fit(train_x, train_y, epochs = 60, batch_size = 16, callbacks = [reduce_lr, model_checkpoint], verbose = 2, validation_data = (val_x, val_y))

plot(history)

# Network 2
model = Sequential()

model.add(Dense(256, input_shape = (612,), kernel_regularizer = regularizers.l2(0.25), name = "Input2"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(512, kernel_regularizer = regularizers.l2(0.25)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(7, name = "last_layer2"))
model.add(Activation('softmax', name = "Output2"))

optimizer = Adam(lr = 0.0005)

model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=0.00001, verbose=1)
                            
model_checkpoint = ModelCheckpoint(filepath = 'weights2.hdf5', verbose = 1, save_best_only = True)
                              

history = model.fit(train1_x, train1_y, epochs = 50, batch_size = 16, callbacks = [reduce_lr, model_checkpoint], verbose = 2, validation_data = (val1_x, val1_y))

plot(history)
