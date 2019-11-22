import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import cv2
import os
import random
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def CreateBatch(num) :
    
    images = []
    labels = []
    
    dim = (96,96)
    a = np.random.randint(0,2, num)

    for i in range(num) :

        if(a[i]==0) : 
            folder = './tensor'
        else: 
            folder = './algo'

        filename = random.choice(os.listdir(folder)) 
    
        img = cv2.imread(os.path.join(folder,filename))        
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        images.append(img)
        labels.append(a[i])    

    images = np.asarray(images)
    labels = np.asarray(labels)

    return images, labels


X, y =  CreateBatch(20)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(96,96,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

# compile the keras model
#FIND THE RIGHT LOSS FUNCTION: https://keras.io/losses/
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=20, verbose=1)
# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(1):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
