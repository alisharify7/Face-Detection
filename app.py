"""
Face detection Module

@auther : alisharify
2023/9/2 - 1402/6/8

"""


import time
import datetime
from threading import Thread

import cv2 as cv
from deepface import DeepFace


global faceMatch
global FaceData
global CheckFlag

CheckFlag = True
faceMatch = False
FaceData = None
sourceImage = cv.imread("./Media/source.png")
sourceName = "Will Smith"

cap = cv.VideoCapture(0)


def checkFace(frame):
    """ 
        This function takes an image frame and 
        check for the face and compare the face with the source image
    """
    global faceMatch
    global CheckFlag
    global FaceData


    try:
        FaceData = DeepFace.verify(frame, sourceImage)
        faceMatch = FaceData['verified'] 
        FaceData = FaceData["facial_areas"]
    except Exception as e:
        faceMatch = False

    CheckFlag = True


# def AnalizyeFace(frame):
#  Analizye's face motion and race and gender
#     global Counter
#     Counter += 1
#     if Counter  == 60:
#         Counter = 0
#         result = DeepFace.analyze(frame)[0]
#         print(json.dumps(result, indent=1))

        

timer = time.time()
imageCounter = 0
FPS = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    if CheckFlag:
        counter = 0
        print(f"Thread Started !{imageCounter}")
        Thread(target=checkFace, args=(frame.copy(), )).start()
        CheckFlag = False

    if faceMatch:
        frame = cv.putText(img=frame,
            org=(0, 690) ,text="Face Match",  color=(0, 0, 255),
            fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=3, fontScale=1.5)

        frame = cv.rectangle(frame, (0,698), (300,698), color=(0 , 0, 255), thickness=2)


        x = "img1"
        startPoint = (FaceData[x]['x'], FaceData[x]['y'],)
        endPoint = (FaceData[x]['x']+FaceData[x]['w'], FaceData[x]['y']+FaceData[x]['h'],)
        
        # draw a rectangle around the face in the frame
        frame = cv.rectangle(frame, startPoint, endPoint, color=(0 , 255, 0), thickness=3)

        # add person name to frame
        frame = cv.putText(img=frame,
            org=(endPoint[0]-100, endPoint[1]+20), text=sourceName,
            color=(0, 0, 255), fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=2, fontScale=0.6)

        frame = cv.putText(img=frame,
            org=(endPoint[0]-100, endPoint[1]+40), text=f"x:{startPoint[0]},y:{startPoint[1]}",
            color=(0, 0, 255), fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=1, fontScale=0.6)


    else:
        frame = cv.putText(img=frame,
            org =(0, 690) , text="No Match", color=(0, 255, 0), 
            fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=3, fontScale=1.5)
        frame = cv.rectangle(frame, (0,698), (225, 698), color=(0 , 255, 0), thickness=2)

    frame = cv.putText(img=frame,
        org=(0, 715), text=f"{datetime.datetime.utcnow()}", color=(0, 0, 255),
        fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=1, fontScale=0.6)


    # FPS
    imageCounter += 1
    now = time.time()
    if now - timer > 1:
        FPS = imageCounter / 1
        imageCounter = 0
        timer = time.time()
    
    frame = cv.putText(img=frame,
        org=(0, 20), text=f"FPS:{FPS}",
        color=(0, 0, 255), fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=1, fontScale=0.6)

    frame = cv.resize(frame, (920, 640))
    cv.imshow("video", frame)
    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()

