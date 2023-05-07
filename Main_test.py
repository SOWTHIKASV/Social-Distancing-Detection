

import tkinter as tk
from tkinter import *
from tkinter import messagebox as mb
import time


def face_detection():

    from keras.models import load_model
    import cv2
    import numpy as np
    from pygame import mixer
    mixer.init()
    sound = mixer.Sound('s1.wav')
    sound2 = mixer.Sound('s1.wav')
    sound1 = mixer.Sound('m1.wav')
    
    model = load_model('model-090.model')
    
    face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    cap=cv2.VideoCapture(0)
    labels_dict={0:'MASK',1:'f MASK'}
    color_dict={0:(0,255,0),1:(0,0,255)}
      
    
    while(True):
    
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_clsfr.detectMultiScale(gray,1.3,5)  
    
        for (x,y,w,h) in faces:
        
            face_img=gray[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            result=model.predict(reshaped)
    
            label=np.argmax(result,axis=1)[0]
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],4)
            cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],4)
            cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_ITALIC, 1,(255,255,255),4)
            
            if(labels_dict[label] =='MASK'):
               print("No Beep")
            elif(labels_dict[label] =='f MASK'):
                    sound1.play()
                    print("Beep") 
            
        cv2.imshow('Detection App',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




def social_distancing():
    import cv2
    from scipy.spatial import distance as dist
    from pygame import mixer
    mixer.init()
    sound2 = mixer.Sound('s1.wav')
    cap = cv2.VideoCapture(0)
    face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        status , photo = cap.read()
        face_cor = face_model.detectMultiScale(photo)
        l = len(face_cor)
        photo = cv2.putText(photo, str(len(face_cor))+" Face", (80, 80), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        stack_x = []
        stack_y = []
        stack_x_print = []
        stack_y_print = []
        global D
        
        if len(face_cor) == 0:
            pass
        else:
            for i in range(0,len(face_cor)):
                x1 = face_cor[i][0]
                y1 = face_cor[i][1]
                x2 = face_cor[i][0] + face_cor[i][2]
                y2 = face_cor[i][1] + face_cor[i][3]

                mid_x = int((x1+x2)/2)
                mid_y = int((y1+y2)/2)
                stack_x.append(mid_x)
                stack_y.append(mid_y)
                stack_x_print.append(mid_x)
                stack_y_print.append(mid_y)

                #photo = cv2.circle(photo, (mid_x, mid_y), 3 , [255,0,0] , -1)
                photo = cv2.rectangle(photo , (x1, y1) , (x2,y2) , [0,255,0] , 2)

            if len(face_cor) == 2:
                D = int(dist.euclidean((stack_x.pop(), stack_y.pop()), (stack_x.pop(), stack_y.pop())))
                #photo = cv2.line(photo, (stack_x_print.pop(), stack_y_print.pop()), (stack_x_print.pop(), stack_y_print.pop()), [0,0,255], 2)
            else:
                D = 0

            if D<250 and D!=0:
                photo = cv2.putText(photo, "!!Social Distance Violent!!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,2, [0,0,255] , 4)
                sound2.play()
                print("Social Distance Violent!!")
                time.sleep(3)
            photo = cv2.putText(photo, str(D/10) + " cm", (300, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255) , 2, cv2.LINE_AA)

            cv2.imshow('Image' , photo)
            if cv2.waitKey(100) == 13:
                break

    cv2.destroyAllWindows()
    
    
def call():
    res=mb.askquestion('Press Yes for Socal Distance', 'Press Yes for Social Distance Application')
    if res == 'yes' :
        social_distancing()
    else :
        face_detection()
call()

