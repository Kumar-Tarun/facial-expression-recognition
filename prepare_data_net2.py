import os
import cv2
import dlib
import numpy as np
from keras.preprocessing import image
from helper import facial_landmarks

PATH2 = 'Normalized-Flipped/'
PATH1 = 'Emotion/'

# Emotion contains the labels for the examples in a file corresponding to the name of last frame of the example

def normalize(point, std, nose):
	a = (point - nose)/std
	return a

for imagefile in sorted(os.listdir(PATH2)):

	for folder in sorted(os.listdir(PATH2+imagefile+'/')):
		label_path = PATH1+imagefile+'/'+folder+'/'

		b = len(os.listdir(label_path))
		X = np.empty((612, 1))
		i=0
		if(b==0):
			print(len(os.listdir(PATH5+imagefile+'/'+folder+'/')))
			continue
		file = open(label_path+os.listdir(label_path)[0]) 
		label = file.read()[3]
		for imagename in sorted(os.listdir(PATH2+imagefile+'/'+folder+'/')):
			img = cv2.imread(PATH2+imagefile+'/'+folder+'/'+imagename)
			img = cv2.resize(img, (64, 64))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			dlib_rect = dlib.rectangle(0, 0, 64, 64)
			fac = facial_landmarks(img, dlib_rect)[17:]
			a = np.empty((51))
			b = np.empty((51))
			j=0
			for (x, y) in fac:
				a[j] = x
				b[j] = y
				j+=1
			nose_x = a[13]
			nose_y = b[13]
			x_noise = np.random.normal(0, 0.01, 51)
			y_noise = np.random.normal(0, 0.01, 51)
			j=0
			a = np.std(a, dtype = np.float64)
			b = np.std(b, dtype = np.float64)
			for (x, y) in fac:
				Y[i] = normalize(x, a, nose_x) + x_noise[j]
				i+=1
				Y[i] = normalize(y, b, nose_y) + y_noise[j]
				i+=1
				j+=1
			name = imagename
		# Different names for different types of augmentation
		np.save(label+'g/'+'flno3'+name[:-4], X)

		
			


