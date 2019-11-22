import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense
import os
import random
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.models import model_from_json

cap = cv2.VideoCapture(0)

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

i = 0
while(True):
    
    ret, frame = cap.read()    
    cv2.imshow('frame',frame)
   
    img = cv2.resize(frame, (96,96), interpolation = cv2.INTER_AREA)
    img = np.asarray(img)
    img = np.expand_dims(img, 0)

    res = loaded_model.predict_classes(img)
    #print(res[0])

    if(res[0]==0) :
        print('tensor')
    else :
        print('algo')


    i += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()