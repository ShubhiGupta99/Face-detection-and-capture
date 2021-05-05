
import cv2,os
import shutil
import numpy as np
from tkinter import*

win=Tk()
#w, h = win1.winfo_screenwidth(), win1.winfo_screenheight()
#win1.geometry("%dx%d" % (w, h))
win.title("Welcome to face detection")
win['background']="#ff7e67"
win.geometry("500x300")


def TakeImages(name):        

    i=0
    if(i==0):
        cam = cv2.VideoCapture(0)
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        sampleNum=0
        directoryName="TrainingImage\%s" %(name)
        os.mkdir(os.path.join("TrainingImage/",name))
        
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),8)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite(str(directoryName)+"\%d.jpg" %(sampleNum) , gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>=50:
                break
        cam.release()
        cv2.destroyAllWindows() 
        
    else:
        print("no")

    

lb=Label(win,text="Enter name")
lb.place(x=100,y=100,width=100,height=30)
name=StringVar()
ent=Entry(win,textvariable=name)
ent.place(x=250,y=100,width=200,height=30)
btn=Button(win,text="Capture",command=lambda:TakeImages(name.get()))
btn.place(x=200,y=150,width=100,height=30)

win.mainloop()
