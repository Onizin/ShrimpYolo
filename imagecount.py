from ultralytics import YOLO 
import torch
import cv2
import numpy as np


# Model loading
model = torch.hub.load("ultralytics/yolov5", "custom",path='best.pt')  # Can be 'yolov5n' - 'yolov5x6', or 'custom'

cap=cv2.imread('D:\\project\\yolov5model1\\image_test\\test2.jpg')

count=0
scale_percent = 60 # percent of original size
width = int(cap.shape[1] * scale_percent / 100)
height = int(cap.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(cap, dim, interpolation = cv2.INTER_AREA)

results = model(resized)

# results.show()
count = (results.pandas().xyxy[0]['name'] == 'Shrimp').sum()
# print(results)      # save results 
# print('total : ',count)
# results = model(resized)
cv2.putText(resized,"Total Shrimp : "+str(count),(28,50),0,2,(0,0,0),10)
gogo = cv2.resize(np.squeeze(results.render()),(900,900), interpolation = cv2.INTER_AREA)
cv2.imshow('yolo',gogo)
results.show()
print(results.pandas().xyxy[0])
results.pandas().xyxy[0]
cv2.waitKey(0)
if k == 30:
    cv2.destroyAllWindows()
elif k == ord('q'):
    cv2.destroyAllWindows()
#  # Other options: .show(), .save(), .crop(), .pandas(), etc.

print(results.pandas().xyxy[0])
results.pandas().xyxy[0]
# keke = []
# keke = results.pandas().xyxy[0]['xmin']
# keke1 = []
# keke1 = results.pandas().xyxy[0]['ymin']
# print(keke)
# print(keke1)