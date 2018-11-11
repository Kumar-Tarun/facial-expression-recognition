import os
import numpy
import math
import cv2

PATH2 = 'Faces/'
PATH1 = 'Normalized-Faces/'
for imagefile in sorted(os.listdir(PATH2)):
	os.mkdir(PATH1+imagefile, 0777)

	for folder in sorted(os.listdir(PATH2+imagefile+'/')):
		if(folder == '_DS_Store'):
			continue

		os.mkdir(PATH1+imagefile+'/'+folder, 0777)
		temp_path = PATH1+imagefile+'/'+folder+'/'
		main_path = PATH2+imagefile+'/'+folder+'/'
		frame_length = len(os.listdir(main_path))
		frames = sorted(os.listdir(main_path))
		a = (frame_length-2)/4
		b = a
		image = cv2.imread(main_path+frames[0])
		cv2.imwrite(temp_path+frames[0], image)
		for i in range(4):
			rounded_up = int(math.ceil(a))
			rounded_down = int(math.floor(a))
			if((a - rounded_down) >= 0.5):
				image = cv2.imread(main_path+frames[rounded_up])
				cv2.imwrite(temp_path+frames[rounded_up], image)
			elif((a - rounded_down) < 0.5):
				image = cv2.imread(main_path+frames[rounded_down])
				cv2.imwrite(temp_path+frames[rounded_down], image)
			a+=b
		image = cv2.imread(main_path+frames[frame_length-1])
		cv2.imwrite(temp_path+frames[frame_length-1], image)



		 
		 
