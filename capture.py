import numpy as np
import cv2

cap = cv2.VideoCapture(0)

i = 0
j = 0
while(True):
    
    ret, frame = cap.read()    
    cv2.imshow('frame',frame)
    
    if (i%20 == 0) and (i>0) :
        cv2.imwrite(f'./tensor/{j}.png',frame)
        j+=1  


    i += 1
    print(j)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if j>100 :
        break

cap.release()
cv2.destroyAllWindows()