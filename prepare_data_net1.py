import os
import cv2
import dlib
import numpy as np
from keras.preprocessing import image
from helper import facial_landmarks

PATH2 = 'Normalized-Flipped/'
PATH1 = 'Emotion/'

# Emotion contains the labels for the examples in a file corresponding to the name of last frame of the example

for imagefile in sorted(os.listdir(PATH2)):

	for folder in sorted(os.listdir(PATH2+imagefile+'/')):
		label_path = PATH1+imagefile+'/'+folder+'/'
		b = len(os.listdir(label_path))
		X = np.empty((6, 64, 64, 1))
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
			img = image.img_to_array(img)
			X[i,] = img
			i+=1
			name = imagename
		# Different names for different types of augmentation
		np.save(label+'/'+'fl'+name[:-4], X)
