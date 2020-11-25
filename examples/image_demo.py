import sys
import os
import cv2
import math
import numpy as np
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints

def draw_numbers(image, keypoints, alpha = 1.0 ,thickness = 2):
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5 
    center_list = [0, 1]
    left_list   = [2, 3, 4, 8, 9, 10, 14, 16]
    right_list  = [5, 6, 7, 11, 12, 13, 15, 17]
    top_list    = [14, 15, 16, 17]
    large_list  = [10, 11, 12, 13, 14, 15, 16]

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
            if n in top_list:
                dy = - 5
            if n in large_list:               
                dy = 2*dy

            location = (x+dx, y+dy)                   
            cv2.putText(overlay, str(n), location , font , fontScale , white , thickness)
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


fname_list = []    
image_path = 'examples/media'
for root, dirs, files in os.walk(image_path):        
    rootpath = os.path.abspath(root)
    for fname in files:
        if fname.endswith('.jpg') or fname.endswith('.png') :
            fname_list.append(os.path.join(rootpath, fname)) 
count = 0
for fname_path in fname_list:    
    try: 
        estimator = BodyPoseEstimator(pretrained=True)
        img_path = os.path.dirname(fname_path)
        img_name = os.path.basename(fname_path)                
        image_src = cv2.imread(fname_path, cv2.IMREAD_ANYCOLOR )
        keypoints = estimator(image_src)
        image_dst = draw_body_connections(image_src, keypoints, thickness=2, alpha=0.7)
        image_dst = draw_keypoints(image_dst, keypoints, radius=3, alpha=0.8)
        #image_dst = draw_numbers(image_dst, keypoints, alpha=0.9, thickness=1)
        count = count + 1
        angle = get_angle(keypoints)
        rate  = get_rate(keypoints)
        mykey = keypoints[0, 0:14, :]    # (14, 3)  =>   x, y, v    
        print(f"No={count:4d} rate={rate:7.2f} , angle={angle:7.2f} ")
        mykey = mykey.reshape(-1)        #(42,)
        out_filename = img_path.split(os.path.sep)[-1]+".csv"       
        #print(out_filename)
        with open(out_filename, 'at')  as f:
            data = f"{out_filename} , {count}"
            for value in mykey:
                data = f"{data} , {value}" 
            data = f"{data} \n" 
            f.write(data)        
            
        cv2.imshow('Image Demo', image_dst)
        if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
            break

    except IndexError:
       print(f"No={count:4d} rate=   0.00 , angle=   0.00 ")
       with open(out_filename, 'at')  as f:
            data = f"{out_filename} , {count}"
            for i in range(42):
                data = f"{data} , 0" 
            data = f"{data} \n" 
            f.write(data) 

cv2.destroyAllWindows()

    
