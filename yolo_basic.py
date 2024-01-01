from ultralytics import YOLO 
import torch
from tracker import *
import cv2
import torch
import numpy as np

# Model loading
model = torch.hub.load("ultralytics/yolov5", "custom",path='best.pt')  # Can be 'yolov5n' - 'yolov5x6', or 'custom'

# Inference on images
img = "./image_test/test1.jpg"  # Can be a file, Path, PIL, OpenCV, numpy, or list of images
count=0
# Run inference
while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(1020,600))
    results = model(img)
    results=model(frame)
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        if 'car' in d:
           cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
           cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

    cv2.imshow("FRAME",frame)
    cv2.setMouseCallback("FRAME",POINTS)
   

    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()


# Display results
results.print()  # Other options: .show(), .save(), .crop(), .pandas(), etc.
results.show()

