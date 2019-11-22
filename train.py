import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import cv2
import os
import random
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation


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
        images.append(img/255) #from 0-255 to 0-1
        labels.append(a[i])    

    images = np.asarray(images)
    labels = np.asarray(labels)

    return images, labels


X, y =  CreateBatch(300)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(96,96,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=50, verbose=1)



model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

'''
predictions = model.predict_classes(X)

for i in range(1):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
'''