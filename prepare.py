import pandas as pd
import numpy as np
from keras import utils
import matplotlib.pyplot as plt
from keras.utils import to_categorical

def get_data():

	#Network1
	train1_x = np.load('../input/erdata/train1_x.npy')
	train1_x = train1_x.reshape((train1_x.shape[0], train1_x.shape[1]))
	val1_x = np.load('../input/erdata/val1_x.npy')
	val1_x = val1_x.reshape((val1_x.shape[0], val1_x.shape[1]))
	test1_x = np.load('../input/erdata/test1_x.npy')
	test1_x = test1_x.reshape((test1_x.shape[0], test1_x.shape[1]))
	train1_y = np.load('../input/erdata/train1_y.npy')
	val1_y = np.load('../input/erdata/val1_y.npy')
	test1_y = np.load('../input/erdata/test1_y.npy')

	#Network2
	train_x = np.load('../input/erdata/train_x.npy')/255
	val_x = np.load('../input/erdata/val_x.npy')/255
	test_x = np.load('../input/erdata/test_x.npy')/255
	train_y = np.load('../input/erdata/train_y.npy')
	val_y = np.load('../input/erdata/val_y.npy')
	test_y = np.load('../input/erdata/test_y.npy')

	# print(train_x.shape)
	# print(test_x.shape)
	# print(val_x.shape)

	train_y = to_categorical(train_y, num_classes = 7)
	val_y = to_categorical(val_y, num_classes = 7)
	test_y = to_categorical(test_y, num_classes = 7)

	train1_y = to_categorical(train1_y, num_classes = 7)
	val1_y = to_categorical(val1_y, num_classes = 7)
	test1_y = to_categorical(test1_y, num_classes = 7)


	train_indices = np.arange(train_y.shape[0])
	val_indices = np.arange(val_y.shape[0])
	test_indices = np.arange(test_y.shape[0])

	np.random.seed(1)

	np.random.shuffle(train_indices)
	np.random.shuffle(test_indices)
	np.random.shuffle(val_indices)

	train_x = train_x[train_indices]
	train_y = train_y[train_indices]
	val_x = val_x[val_indices]
	val_y = val_y[val_indices]
	test_x = test_x[test_indices]
	test_y = test_y[test_indices]

	train1_x = train1_x[train_indices]
	train1_y = train1_y[train_indices]
	val1_x = val1_x[val_indices]
	val1_y = val1_y[val_indices]
	test1_x = test1_x[test_indices]
	test1_y = test1_y[test_indices]

	return {"train_x": train_x, "train_y": train_y, "val_x": val_x, "val_y": val_y, "test_x": test_x, "test_y": test_y
			"train1_x": train1_x, "train1_y": train1_y, "val1_x": val1_x, "val1_y": val1_y, "test1_x": test1_x, "test1_y": test1_y}						}


def plot(history):

	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
