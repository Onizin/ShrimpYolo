import cv2
import torch
import numpy as np

model = torch.hub.load("ultralytics/yolov5", "custom",path='best.pt', force_reload=True)  # Can be 'yolov5n' - 'yolov5x6', or 'custom'

cap=cv2.VideoCapture('./image_test/13.mp4')
b= model.names[0] = 'Shrimp'

size = 416


color=(0,0,255)

# count=0


while True:
    ret,frame=cap.read()
    # scale_percent = 60 # percent of original size
    # width = int(cap.shape[1] * scale_percent / 100)
    # height = int(cap.shape[0] * scale_percent / 100)
    # dim = (width, height)
    resized = cv2.resize(frame, (500,500))
    results = model(resized)
    count = (results.pandas().xyxy[0]['name'] == 'Shrimp').sum()
    
    gogo = cv2.resize(np.squeeze(results.render()),(500,500), interpolation = cv2.INTER_AREA)
    frame = cv2.imshow('yolo',gogo)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()