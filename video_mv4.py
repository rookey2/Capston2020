import sys
import glob
import os
import cv2
import math
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints
import numpy as np

def _draw_conn(image, point1, point2, color, thickness=1):
    x1, y1, v1 = point1
    x2, y2, v2 = point2
    if v1 and v2:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    return image

def draw_body_conn(image, keypoints, thickness=1, alpha=1.0):
    overlay = image.copy()
    b_conn = [(1, 2), (1, 5), (2, 8), (5, 11), (8, 11)]        
    for kp in keypoints:
        for i, j in b_conn:
            overlay = _draw_conn(overlay, kp[i], kp[j], (0, 0, 255), thickness)       
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
   


def get_keydata(name, frameNo, newkey, rate, angle, direct): 
    key_list = f"{name}, {frameNo}"              
    thiskey = newkey.reshape(-1)          
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


def get_Velocity(npArr, key, keynum, idx):  # 10 frame 이상의 속도 계산 
    retdata = np.zeros([4, 6])
    key_num = [2, 5, 8, 11]     
    for k, num in enumerate(key_num) :
        npArr[idx, k, 0:3] = key[0, num, :]               
        npArr[idx, k, 5] = idx 
        t2 = npArr[idx, k, 5]        
        for m in range(1, 11) :             # 현재 frame부터 10 frame 이상 떨어진 유효한 데이터 획득 (중간에 skip된 frame 고려)
            j = (idx-m)%1800               
            t1 =  npArr[j, k, 5]
            if(t2 - t1 >= 10) :
                break
        x2, y2 = npArr[idx, k, 0], npArr[idx, k, 1]
        if npArr[j, k, 2] == 1:
            x1, y1 = npArr[j, k, 0], npArr[j, k, 1]
        else :
            x1, y1 = x2, y2
        x = x2 - x1
        y = y2 - y1
        t = t2 - t1
        
        v = np.sqrt( x**2 + y**2) / t    
        angle = math.degrees(math.atan2(y, x))        
        npArr[idx, k, 3] = np.round_(v)
        npArr[idx, k, 4] = np.round_(angle)
    retdata = npArr[idx, :, :]
    print(retdata)
    return retdata

#video_path = "./examples/fall_test/*" #avi파일 찾을 위치
video_path = "./examples/media/*" #avi파일 찾을 위치
avi_list = []
dir_check(video_path)

with open('./fallen_data.csv', 'w') as f:
    data = 'fName, frameNo, x2, y2, val2, vec2, deg2, no, x5, y5, val5, vec5, deg5, no, ' 
    data += 'x8, y8, val8, vec8, deg8, no, x11, y11, val11, vec11, deg11, no, '
    data += 'rate, angle, direct\n'
    f.write(data)
    estimator = BodyPoseEstimator(pretrained=True)  
      
    newKeypont = np.zeros([1, 18, 3])
    npArray    = np.zeros([1800, 4, 6])   #  x, y, val, vec, deg, no
    
    for video_fullpath in avi_list:
        output_name = f"{os.path.dirname(video_fullpath).split(os.path.sep)[-1]}_{os.path.basename(video_fullpath)}"            
        videoclip = cv2.VideoCapture(video_fullpath)        
        booleanStart = False
        count  = 0
        angle, rate, direct = -1, -1, 0        
        key_num = [2, 5, 8, 11]
        
        while videoclip.isOpened():
            flag, frame = videoclip.read()            
            if not flag:
                break             
            keypoints = estimator(frame)
            if len(keypoints) == 0:             # keypoint를 못찾은  frame skip
                cv2.imshow('Video Demo', frame)
                if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
                    break  
                count += 1
                continue
            
            sum = 0
            for num in key_num:                  # key_num가 모두 유효하지 않은 frame skip
                sum += keypoints[0, num, 2]                        
            if sum != len(key_num) :                
                cv2.imshow('Video Demo', frame)
                if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
                    break  
                count += 1            
                continue

            if booleanStart == False :
                booleanStart = True
                count = 1

            newKey = get_Velocity(npArray, keypoints, key_num, (count%1800)) 
            sum = 0
            for num in range(4):                  # key_num가 움직이지 않는 frame skip
                sum += newKey[num, 3]  # 속도값
            if sum == 0 :                
                cv2.imshow('Video Demo', frame)
                if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
                    break  
                count += 1            
                continue

            try :                                              
                frame = draw_body_conn(frame, keypoints, thickness=2, alpha=0.7)
                frame = draw_keypoints(frame, keypoints, radius=1, alpha=0.6)                                               
                angle = get_angle(keypoints)
                rate  = get_rate(keypoints)
                direct  = get_direction(keypoints)
                data  = get_keydata(output_name, count, newKey, angle, rate, direct)                
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
