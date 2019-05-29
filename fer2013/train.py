from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import regularizers
from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

train_x = np.load('ferdata/train_x.npy')
train_y = np.load('ferdata/train_y.npy')
train_y = to_categorical(train_y, 7)

def create_model(shape):
	input_shape = _obtain_input_shape(shape,
	                                  default_size=224,
	                                  min_size=48,
	                                  data_format=K.image_data_format(),
	                                  require_flatten=False)

	img_input = Input(shape = input_shape)
	x = Conv2D(64, kernel_size = (3,3), padding = 'same', name = 'conv1_1')(img_input)
	x = BatchNormalization(name = 'bn1')(x)
	x = Dropout(0.5, name = 'dp1')(x)
	x = Activation('relu', name = 'ac1')(x)

	x = Conv2D(64, kernel_size = (3,3), name = 'conv1_2')(x)
	x = BatchNormalization(name = 'bn2')(x)
	x = Dropout(0.5, name = 'dp2')(x)
	x = Activation('relu', name = 'ac2')(x)
	x = MaxPooling2D(pool_size = (2,2), name = 'pool2')(x)

	x = Conv2D(128, kernel_size = (3,3), padding = 'same', name = 'conv2_1')(x)
	x = Dropout(0.5, name = 'dp3')(x)
	x = Activation('relu', name = 'ac3')(x)

	x = Conv2D(128, kernel_size = (3,3), name = 'conv2_2')(x)
	x = Dropout(0.5, name = 'dp4')(x)
	x = Activation('relu', name = 'ac4')(x)
	x = MaxPooling2D(pool_size = (2,2), name = 'pool4')(x)

	x = Conv2D(128, kernel_size = (3,3), name = 'conv2_3')(x)
	x = Dropout(0.5, name = 'dp6')(x)
	x = Activation('relu', name = 'ac6')(x)
	x = MaxPooling2D(pool_size = (2,2), name = 'pool5')(x)

	x = Flatten(name = 'flatten')(x)
	x = Dense(128, kernel_regularizer = regularizers.l2(0.01), name = 'dense1')(x)
	x = Dropout(0.5, name = 'dp5')(x)
	x = Activation('relu', name = 'ac5')(x)
	x = Dense(7, activation = 'softmax', name = 'dense2')(x)

	model = Model(inputs = img_input, outputs = x)

	return model

model = create_model(train_x.shape)

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 0.0005), metrics = ['accuracy'])

model_checkpoint = ModelCheckpoint(filepath = 'weights.hdf5', verbose = 1, save_best_only = True, save_weights_only = True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=0.00002, verbose=1)
                              
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)

history = model.fit(train_x, train_y, epochs = 200, batch_size = 16, verbose = 2, validation_split = 0.1, callbacks = [model_checkpoint])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()