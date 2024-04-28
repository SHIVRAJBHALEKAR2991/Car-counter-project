import cv2
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *

model =YOLO('yolov8n.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask =cv2.imread('mask2.png')
# cap=cv.VideoCapture(0)
cap =cv2.VideoCapture('Videos/carproject.mp4')
cap.set(3,640)
cap.set(4,480)


# tracking

tracker = Sort(max_age=20,min_hits=3 ,iou_threshold=0.3)
limits =[150,297,550,297]
total=[]


while True:
    success,img=cap.read()
    # cv.imshow('image',img)

    imgregion=cv.bitwise_and(img,mask)
    result=model(imgregion,stream=True)

    detections = np.empty((0,5))
    for r in result:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            # cv.rectangle(img,(x1,y1),(x2,y2),(0,0,255),thickness=2)

            # bounding box
            w,h=x2-x1,y2-y1

            # confidence
            conf=math.ceil((box.conf[0]*100))/100
            # print(conf)

            # class
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if ((currentclass=='car' or currentclass=='motorbike' or currentclass=='bicycle' or currentclass=='bus' or currentclass=='truck') and conf>0.3):
                # cvzone.putTextRect(img, f'{conf},{classNames[int(cls)]}', (max(0, x1), max(35, y1)), offset=3,
                #                    scale=0.6, thickness=1)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt =5)
                currentarray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentarray))
    resultstravker=tracker.update(detections)
    cv.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),thickness=1)



    for res in resultstravker:

        x1,y1,x2,y2,id =res
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w,h=x2-x1 ,y2-y1
        cx,cy=x1+w//2,y1+h//2
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2,colorR =(255,0,0))
        cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),max(35,y1)),scale=0.8,thickness=1,offset=5)
        cv.circle(img,(cx,cy),radius=2,color=(0,255,0), thickness=cv.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-20<cy<limits[3]+20 :
            if total.count(id)==0:
                total.append(id)

    cvzone.putTextRect(img,f'count {len(total)}',(50,50))
    cv.imshow('image',img)
    cv.waitKey(1)

