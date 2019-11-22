import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import cv2
import os
import random
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.models import model_from_json

dim = (96,96)
img = cv2.imread('./algo/3.png')        
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
img = np.asarray(img)
img = np.expand_dims(img, 0)

#images.append(img/255) 
#labels.append(a[i])    


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

res = loaded_model.predict_classes(img)
print(res)

#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
