import numpy as np
import os

val_subs = ['S064', 'S092', 'S098', 'S106', 'S114', 'S120', 'S127', 'S135', 'S151', 'S160', 'S506']
test_subs = ['S022', 'S042', 'S054', 'S066', 'S078', 'S094', 'S110', 'S116', 'S122', 'S149', 'S501']

val_x = []
val_y = []
train_x = []
train_y = []
test_x = []
test_y = []

i=0
j=0
k=0
for t in range(1, 8):
	images = sorted(os.listdir(str(t)+'g/'))
	for file in images:
		s1 = file[:2]
		s2 = file[:4]
		s3 = file[:3]
		s4 = file[:5]
		temp = np.load(str(t)+'g/'+file)
		if(s4 == 'flno1' or s4 == 'flno2' or s4 == 'flno3' or s4 == 'ri-10' or s4 == 'ro-10'):
			if(file[5:9] in val_subs):
				val_x.append(temp)
				val_y.append(t-1)
				i+=1
			elif(file[5:9] in test_subs):
				test_x.append(temp)
				test_y.append(t-1)
				j+=1
			else:
				train_x.append(temp)
				train_y.append(t-1)
				k+=1
		elif(s2 == 'ri-5' or s2 == 'ri15' or s2 == 'ro15' or s2 == 'ro-5'):
			if(file[4:8] in val_subs):
				val_x.append(temp)
				val_y.append(t-1)
				i+=1
			elif(file[4:8] in test_subs):
				test_x.append(temp)
				test_y.append(t-1)
				j+=1
			else:
				train_x.append(temp)
				train_y.append(t-1)
				k+=1
		elif(s3 == 'no1' or s3 == 'no2' or s3 == 'no3'):
			if(file[3:7] in val_subs):
				val_x.append(temp)
				val_y.append(t-1)
				i+=1
			elif(file[3:7] in test_subs):
				test_x.append(temp)
				test_y.append(t-1)
				j+=1
			else:
				train_x.append(temp)
				train_y.append(t-1)
				k+=1
		elif(s1 == 'fa' or s1 == 'fl'):
			if(file[2:6] in val_subs):
				val_x.append(temp)
				val_y.append(t-1)
				i+=1
			elif(file[2:6] in test_subs):
				test_x.append(temp)
				test_y.append(t-1)
				j+=1
			else:
				train_x.append(temp)
				train_y.append(t-1)
				k+=1



val_x = np.array(val_x)
val_y = np.array(val_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
train_x = np.array(train_x)
train_y = np.array(train_y)

print(i)
print(j)
print(k)

np.save('val/val1_x', val_x)
np.save('val/val1_y', val_y)
np.save('test/test1_x', test_x)
np.save('test/test1_y', test_y)
np.save('train/train1_x', train_x)
np.save('train/train1_y', train_y)
