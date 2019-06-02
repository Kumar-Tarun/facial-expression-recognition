import os
import numpy as np
import pandas as pd

path = 'fer2013/fer2013.csv'

x = pd.read_csv(path)

length = x['pixels'].values.shape[0]

y = []
z = []

for i in range(length):
	y.append([int(s) for s in x['pixels'].values[i].split(' ')])
	z.append(x['emotion'].values[i])

y = np.array(y)
y = np.reshape(y, (y.shape[0], 48, 48, 1))
y = y.astype(np.float32)/255.0

z = np.array(z)
np.save('ferdata/train_x.npy', y)
np.save('ferdata/train_y.npy', z)
print(y.shape)
print(z.shape)



