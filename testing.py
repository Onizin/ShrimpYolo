import cv2
import torch
import numpy as np

model = torch.hub.load("ultralytics/yolov5", "custom",path="best.pt")

cap = cv2.VideoCapture('./image_test/13.mp4')
while True:
    ret,frame=cap.read()
    frame = cv2.resize(frame,(640,640))
    results=model(frame)
    frame=np.squeeze(results.render())
    cv2.imshow("frame",frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()