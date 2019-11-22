import numpy as np
import cv2
import os
import random

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

    sha = np.asarray(images).shape

    print('test')    

CreateBatch(20)
