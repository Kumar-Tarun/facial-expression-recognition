import pandas as pd
import numpy as np
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import utils
from keras import regularizers
from prepare import get_data
import matplotlib.pyplot as plt

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

model1 = load_model('../input/weights/weights1.hdf5')
model2 = load_model('../input/weights/weights2.hdf5')

input1 = model1.input
input2 = model2.input
output1 = model1.output
output2 = model2.output
pred1 = model1.get_layer('last_layer1').output
pred2 = model2.get_layer('last_layer2').output

for layer in model1.layers[:15]:
    layer.trainable = False
    
for layer in model2.layers[:8]:
    layer.trainable = False
    
x = Add()([pred1, pred2])
x = Dropout(0.3, name = "aaa")(x)
#x = Dense(7, name = "aa")(x)
x = Activation('softmax', name = 'Output3')(x)

model = Model(inputs = [input1, input2], outputs = [output1, output2, x])

losses = {'Output1': 'categorical_crossentropy', 'Output2': 'categorical_crossentropy', 'Output3': 'categorical_crossentropy'}

lossweights = {'Output1': 1.0, 'Output2': 1.0, 'Output3': 0.1}

optimizer = Adam(lr = 0.0005)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
#                             patience=2, min_lr=0.00001, verbose=1)


model.compile(loss = losses, loss_weights = lossweights, optimizer = optimizer, metrics = ['accuracy'])

input_train_data = {'Input1_input': train_x, 'Input2_input': train1_x}
output_train_data = {'Output1': train_y, 'Output2': train1_y, 'Output3': train_y}

input_val_data = {'Input1_input': val_x, 'Input2_input': val1_x}
output_val_data = {'Output1': val_y, 'Output2': val1_y, 'Output3': val_y}
model_checkpoint = ModelCheckpoint(filepath = 'weightjt.hdf5', verbose = 1, save_best_only = True)

history = model.fit(input_train_data, output_train_data, epochs = 25, batch_size = 32, callbacks = [model_checkpoint], validation_data = (input_val_data, output_val_data), verbose = 2)


input_test_data = {'Input1_input': test_x, 'Input2_input': test1_x}
output_test_data = {'Output1': test_y, 'Output2': test1_y, 'Output3': test_y}

score = model.evaluate(input_test_data, output_test_data)

print(score)

# predictions1 = model1.predict(test_x)
# predictions2 = model2.predict(test1_x)

# pred = (predictions1 + predictions2)/2

# pred = np.argmax(pred, 1)
