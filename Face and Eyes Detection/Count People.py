import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import imutils

face_cascPath = "Model/haarcascade_frontalface_default.xml"
eye_cascPath = "Model/haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)
log.basicConfig(filename='CountPeople.log',level=log.INFO)

#By FIKO
#cv2.VideoCapture(0) user for camera in computer
#cv2.VideoCapture(1) user for webcam(USB) in computer
video_capture = cv2.VideoCapture(0)
video_capture.set(3,640) #set Width
video_capture.set(4,480) #set Height
anterior_face = 0
anterior_eye = 2
log.info("-------------System Start-------------  at "+str(dt.datetime.now()))
log.info(" #####   #####   #####   ####    #####")
log.info(" #         #     #   #   #   #     #  ")
log.info(" #####     #     #####   ####      #  ")
log.info("     #     #     #   #   #  #      #  ")
log.info(" #####     #     #   #   #   #     #  " + "\n")
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame q
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    # frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eyeCascade.detectMultiScale(frame[y:y+h, x:x+w])
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,10),2)

    # If webcam can detect face more than one. next to check eyes if have more two eyes wait info to file than you set
    if anterior_face != len(faces):
        anterior_face = len(faces)
        if anterior_face >= 1:
            if anterior_eye <= len(eyes):
                log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
            # else :
            #     log.info("-------No People---------  at "+str(dt.datetime.now()))
    
    # Display the resulting frame
    cv2.imshow('Count People', frame)

    # click 'p" for stop system
    if cv2.waitKey(1) & 0xFF == ord('q'):
        log.info("-------System Offline---------  at "+str(dt.datetime.now()))
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
