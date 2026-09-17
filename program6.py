# PROGRAM 6: Traffic sign detection using HSV color segmentation
import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
image = cv2.imread(str(BASE/"inputs"/"traffic_signs.jpeg"))
if image is None: raise FileNotFoundError("traffic_signs.jpeg not found")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
red1 = cv2.inRange(hsv,(0,80,80),(10,255,255))
red2 = cv2.inRange(hsv,(170,80,80),(180,255,255))
blue = cv2.inRange(hsv,(90,80,70),(130,255,255))
yellow = cv2.inRange(hsv,(18,80,80),(40,255,255))
combined = red1 | red2 | blue | yellow

kernel = np.ones((5,5),np.uint8)
combined = cv2.morphologyEx(combined,cv2.MORPH_OPEN,kernel)
combined = cv2.morphologyEx(combined,cv2.MORPH_CLOSE,kernel)

contours,_ = cv2.findContours(combined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 150: continue
    perimeter = cv2.arcLength(cnt,True)
    if perimeter == 0: continue
    circularity = (4*np.pi*area)/(perimeter*perimeter)
    if circularity > 0.35:
        x,y,w,h = cv2.boundingRect(cnt)
        count += 1
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(image,"Traffic Sign",(x,max(y-8,20)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

out = BASE/"outputs"/"program6_traffic_signs.jpg"
cv2.imwrite(str(out),image)
print("Traffic signs detected:",count)
print("Output saved:",out)
cv2.imshow("Traffic Sign Detection",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
