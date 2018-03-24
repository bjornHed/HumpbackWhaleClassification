import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Lambda, Flatten, Conv2D
from keras import optimizers
import matplotlib.image as mpimg
from skimage import color
from skimage.transform import resize

import matplotlib.pyplot as plt
import random

# Utils
batch_size = 1
epochs = 1

def hamming_weight(vector):
    weight = 0
    for i in range(4250):
        if(vector[i] > 0):
            weight += 1
    return weight

def mini_batch_generator():
    size = 9040
    nbr_batches = size//batch_size
    r = range(size)

    for _ in range(nbr_batches):
        x_batch = np.empty((batch_size, 100, 50, 1))
        y_batch = np.empty((batch_size, 4250))

        for _ in range(batch_size):
            i = random.choice(r)
            img = mpimg.imread("train/" + data.iloc[i].Image)
            img = color.rgb2gray(img)
            img = resize(img, (100, 50))
            np.append(x_batch, np.reshape(img, (100, 50, 1)))
            tmp_y_batch = np.zeros((1,4250))
            np.put(tmp_y_batch, data.iloc[i].Id, 1)
            np.append(y_batch, tmp_y_batch)

        yield x_batch, y_batch

# Read input
data = pd.read_csv("train.csv")
data = data[data.Id != "new_whale"]

id_to_whale = {}
whale_to_id = {}
counter = 0

for whale in data.Id:
    int_id = whale_to_id.get(whale)
    if int_id is None:
        id_to_whale[counter] = whale
        whale_to_id[whale] = counter
        int_id = counter
        counter += 1

for i in range(len(data.Id)):
    data.iloc[i].Id = whale_to_id[data.iloc[i].Id]


data_labels = data['Id']
# Preprocess data
one_hot_labels = keras.utils.to_categorical(data_labels, num_classes=4250)
print(one_hot_labels.shape)

# Define the model
model = Sequential()
model.add(Conv2D(batch_size, kernel_size=(5,5), input_shape=(100,50,1), padding='Same', activation='relu'))
model.add(Conv2D(batch_size, kernel_size=(5,5), padding='Same', activation="relu"))
model.add(Conv2D(64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(4250, activation='softmax'))
model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

# Train the model
for e in range(epochs):
    for x_batch, y_batch in mini_batch_generator():
        print("Weight",hamming_weight(y_batch[-1]))
        print(y_batch.shape)
        history = model.fit(x=x_batch,
                            y=y_batch,
                            batch_size=batch_size,
                            verbose=2,
                            epochs=1)

# Evaluate
score = model.evaluate(eval_data,one_hot_eval_labels)
print('Score: ',score)

# Summarize and plot history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# Create output from test
#predictions = model.predict_classes(test_data)

#submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                        # "Label": predictions})
#submissions.to_csv("results.csv", index=False, header=True)
