import cv2
import numpy as np

haar_data = cv2.CascadeClassifier(r'C:\Users\habis\PycharmProjects\pythonProject2\data.xml')
capture =cv2.VideoCapture(0)
data=[]
while True:
 flag,img= capture.read()
 if flag:
     faces=haar_data.detectMultiScale(img)
     for x,y,w,h in faces:
         cv2.rectangle(img,(x,y),(x+w , y+h),(225,0,0),4)
         face=img[y:y+h,x:x+w,:]
         face=cv2.resize(face,(50,50))
         print(len(data))
         if len(data) <= 400:
            data.append(face)
     cv2.imshow('result',img)
     if cv2.waitKey(2) == 27 or len(data)>=200 :
         break
capture.release()
cv2.destroyAllWindows()
np.save('without_mask',data) ##while capture a picture without mask, After that you have to change it to With_mask and capture a face with mask
