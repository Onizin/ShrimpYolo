import numpy as np
import torch
import cv2
import math


cap = cv2.VideoCapture('./image_test/13.mp4')

model = torch.hub.load("ultralytics/yolov5", "custom",path='D:\\project\\yolov5model1\\best.pt', force_reload=True)
# b = model.names[2] = 'car'

size = 416

count=0
counter=0
color = (0,0,255)
cyl =250
offset=6


while True:
    ret,img=cap.read()

    count+=1
    if count%1 != 0:
        continue
   
    if ret ==True:
         img = cv2.resize(img,(800,1000),fx=0,fy=0, interpolation=cv2.INTER_CUBIC)
    else:
        break
    
    results=model(img,size)
    count = (results.pandas().xyxy[0]['name'] == 'Shrimp').sum()
    
    for index,row in results.pandas().xyxy[0].iterrows():
        x1=int(row['xmin'])
        y1=int(row['ymin'])
        x2=int(row['xmax'])
        y2=int(row['ymax'])
        d=(row['class'])
        if d==0:
            ss = str(round(row['confidence'],3))
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(img,ss,(x1,y1),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
            rectx1,recty1=((x1+x2)/2,(y1+y2)/2)
            rectcenter=int(rectx1),int(recty1)
            cx=rectcenter[0]
            cy=rectcenter[1]
            length=math.sqrt(((x2-x1)**2))/10
            width =math.sqrt(((rectx1-recty1)**2))/10
            
            cv2.circle(img,(cx,cy),3,(0,255,0),-1)
            cv2.putText(img,"Length : "+str(length)+" CM",(cx,cy+10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
            cv2.putText(img,"Width : "+str(width)+" CM",(cx,cy+30),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
            cv2.putText(img,"Total Shrimp : "+str(count),(28,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
            cv2.imshow("IMG",img)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()