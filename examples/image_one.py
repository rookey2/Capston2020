import sys
import os
import cv2
import math
import numpy as np
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints

def draw_numbers(image, keypoints, alpha = 1.0 ,thickness = 2):
    black = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5 
    center_list = [0, 1]
    left_list   = [2, 3, 4, 8, 9, 10]
    right_list  = [5, 6, 7, 11, 12, 13]    
    large_list  = [10, 11, 12, 13 ]

    n = 0
    overlay = image.copy()
    
    for kp in keypoints:
        for x,y,z in kp:  
            dx = 0
            dy = 0           
            if n in center_list:      
                dy = 15  
            if n in left_list:
                dx = -15
            if n in right_list:
                dx = 5            
            if n in large_list:               
                dy = 2*dy

            location = (x+dx, y+dy)                   
            cv2.putText(overlay, str(n), location , font , fontScale , black , thickness)
            n = n+1;
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

def get_angle(keypoints):
    x1,x2,x3 = keypoints[0][8][0], keypoints[0][11][0], keypoints[0][1][0] 
    y1,y2,y3 = keypoints[0][8][1], keypoints[0][11][1], keypoints[0][1][1] 
    
    x4 = (x1+x2)/2
    y4 = (y1+y2)/2
    x = x4 - x3 
    y = y4 - y3
    angle = math.degrees(math.atan2(y,x))
    if angle >= 90:
        angle = 180-angle
    elif angle >= -90:
        if angle < 0 : 
            angle = -angle
    else:
        angle = 180 + angle

    return angle

def get_rate(keypoints):
    x = [keypoints[0][2][0], keypoints[0][5][0], keypoints[0][8][0], keypoints[0][11][0]]
    y = [keypoints[0][2][1], keypoints[0][5][1], keypoints[0][8][1], keypoints[0][11][1]]    
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    w = xmax - xmin
    h = ymax - ymin
    rate = w/h
    return rate

img_path = "examples/Capture/video10_lb/video (10).avi_000006.641.jpg"   
img_path = "examples/media//example.jpg"   
estimator = BodyPoseEstimator(pretrained=True)               
image_src = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR )
keypoints = estimator(image_src)
print(keypoints)
image = image_src.copy();
w, h, v = image.shape
print(image.shape)
cv2.rectangle(image, (0,0) , (h, w), (255,255,255), -1 )

image_dst = draw_body_connections(image, keypoints, thickness=2, alpha=0.7)
image_dst = draw_keypoints(image_dst, keypoints, radius=3, alpha=0.8)
#image_dst = draw_numbers(image_dst, keypoints[:, 0:14, :], alpha=0.9, thickness=1)
angle = get_angle(keypoints)
rate  = get_rate(keypoints)
mykey = keypoints[0, 0:14, :]    # (14, 3)  =>   x, y, v    
print(f"rate={rate:7.2f} , angle={angle:7.2f} ")

while True:
    cv2.imshow('Image Demo', image_dst)
    if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
        break
cv2.destroyAllWindows()

    
