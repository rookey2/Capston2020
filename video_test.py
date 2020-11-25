import sys
import glob
import os
import cv2
import math
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints
import numpy as np


def get_keydata(name, frameNo, mykey, rate, angle, direct): 
    key_list = f"{name}, {frameNo}"  
    if len(mykey) == 0:
        for value in range(15):
            key_list = f"{key_list}, 0, 0, 0, 0, 0" 
        key_list = f"{key_list}, 0, 0, 0 \n"   
    else:                
        thiskey = mykey[0, 0:15, 0:5].reshape(-1)          
        for value in thiskey:
            key_list = f"{key_list}, {value}" 
        key_list = f"{key_list}, {rate:5.2f}, {angle:5.2f}, {direct:4d} \n" 
    return key_list

def get_angle(keypoints):    
    angle = -1
    if len(keypoints) != 0:
        x1,x2,x3 = keypoints[0][8][0], keypoints[0][11][0], keypoints[0][1][0] 
        y1,y2,y3 = keypoints[0][8][1], keypoints[0][11][1], keypoints[0][1][1] 
        
        x4 = (x1+x2)/2
        y4 = (y1+y2)/2
        x = x4 - x3 
        y = y4 - y3
        angle = math.degrees(math.atan2(y,x))
        if angle >= 90:
            angle = 180-angle
        elif 0 > angle >= -90:
            angle = -angle
        elif -90 > angle:
            angle = 180 + angle
    return angle

def get_rate(keypoints):
    rate = -1
    if len(keypoints) != 0:
        key_num = [2,5,8,11]
        x = []
        y = []
        for num in key_num:
            if keypoints[0][2][2] is not 0:
                x.append(keypoints[0][num][0])
                y.append(keypoints[0][num][1])            
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        w = xmax - xmin
        h = ymax - ymin
        if(h != 0 ) : 
            rate = w/h        
    return rate

def dir_check(_dir): #해당 디렉토리 내의 모든 avi파일 주소 목록
    file_list = glob.glob(_dir)
    for file_check in file_list:
        if os.path.isdir(file_check):
            dir_check("{}/*".format(file_check))
        elif file_check.endswith(".avi"):
            avi_list.append(file_check)

def get_direction(keypoints):
    direct = 0
    if len(keypoints) != 0:
        y1,y2 = keypoints[0][8][1], keypoints[0][1][1]   
        if y2 >= y1:
            direct = -1
        else:
            direct = 1
    return direct

def get_PredictKey(npArr, key, i):
    new_key = np.zeros([1, 18, 5]) 
    dx, dx2, dy, dy2 = 0, 0, 0, 0 
    j = (i-1)%1800;        
    if len(key) != 0: 
        for k in range(18):
            if key[0, k, 2] != 0:    
                npArr[i, k, 0:3] = key[0, k, :]
                if npArr[j, k, 0] != 0:
                    dx  = npArr[i, k, 0]  - npArr[j, k, 0] 
                    if npArr[j, k, 3] != 0 :
                        dx2 = npArr[j, k, 3] 
                    else :
                        dx2 = dx
                else :
                    dx , dx2 = 0, 0
                    
                if npArr[j, k, 1] != 0 :
                    dy  = npArr[i, k, 1]  - npArr[j, k, 1]                    
                    if npArr[j, k, 4] != 0 :
                        dy2 = npArr[j, k, 4] 
                    else :
                        dy2 = dy                    
                else :
                    dy, dy2 = 0, 0
                                
                if (dx-dx2)**2 > 50000 :
                    npArr[i, k, 0]   = npArr[j, k, 0] + dx2
                if (dy-dy2)**2 > 50000 :    
                    npArr[i, k, 1]   = npArr[j, k, 1] + dy2                
                npArr[i, k, 3]   = dx
                npArr[i, k, 4]   = dy        
            else:
                npArr[i, k, 0:2]   = npArr[j, k, 0:2]
                npArr[i, k, 2]   = 0
                npArr[i, k, 3]   = 0
                npArr[i, k, 4]   = 0        
    else : 
        npArr[i, :, 0:2]  = npArr[j, :, 0:2]
        npArr[i, :, 2]   = 0
        npArr[i, :, 3]   = 0
        npArr[i, :, 4]   = 0      
   
    new_key[0, :, :] = npArr[i, :, :]
    return new_key


video_path = "./examples/fall_test/*" #avi파일 찾을 위치
#video_path = "./examples/media/*" #avi파일 찾을 위치
avi_list = []
dir_check(video_path)

with open('./fallen_data.csv', 'w') as f:
    data = 'fName, frameNo, x0, y0, v0, dx0, dy0, x1, y1, v1, dx1, dy1, x2, y2, v2, dx2, dy2, x3, y3, v3, dx3, dy3, x4, y4, v4, dx4, dy4, ' 
    data += 'x5, y5, v5, dx5, dy5, x6, y6, v6, dx6, dy6, x7, y7, v7, x7, y7, x8, y8, v8, dx8, dy8, x9, y9, v9, dx9, dy9, '
    data += 'x10, y10, v10, dx10, dy10, x11, y11, v11, dx11, dy11, x12, y12, v12, dx12, dy12, x13, y13, v13, dx13, dy13, x14, y14, v14, dx14, dy14, ' 
    data += ' rate, angle, direct\n'
    f.write(data)
    estimator = BodyPoseEstimator(pretrained=True)       
    npArray = np.zeros([1800, 18, 5])    
    newKeypont = np.zeros((1, 18, 3))
    
    for video_fullpath in avi_list:
        output_name = f"{os.path.dirname(video_fullpath).split(os.path.sep)[-1]}_{os.path.basename(video_fullpath)}"            
        videoclip = cv2.VideoCapture(video_fullpath)
        count = 0
        angle, rate, direct = -1, -1, 0        
        while videoclip.isOpened():
            flag, frame = videoclip.read()            
            if not flag:
                break            
            try :
                keypoints = estimator(frame)                
   
                key = get_PredictKey(npArray, keypoints, count%1800)              
                newKeypont[0, :, :] = key[0, :, 0:3]
                frame = draw_body_connections(frame, newKeypont, thickness=2, alpha=0.7)
                frame = draw_keypoints(frame, newKeypont, radius=3, alpha=0.8)                                               
                angle = get_angle(newKeypont)
                rate  = get_rate(newKeypont)
                direct  = get_direction(newKeypont)
                data  = get_keydata(output_name, count, key, angle, rate, direct)                
                f.write(data)
                #print(f"{output_name} No={count:4d}   rate={rate:7.2f}   angle={angle:7.2f}   direct={direct:3d}")   
                cv2.imshow('Video Demo', frame)
                if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
                    break  
                              
            except KeyboardInterrupt:
                videoclip.release() 
                cv2.destroyAllWindows()
                f.close()
                exit(1)
            except Exception as e:
                print("에러 :", e)
            count +=1
                     
        videoclip.release()        
    cv2.destroyAllWindows()
f.close()
