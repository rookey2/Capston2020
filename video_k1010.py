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
    h_conn = [(0, 1)]
    l_conn = [(5, 6), (6, 7), (11, 12), (12, 13)]
    r_conn = [(2, 3), (3, 4), (8, 9), (9, 10)]
    for kp in keypoints:
        for i, j in b_conn:
            overlay = _draw_conn(overlay, kp[i], kp[j], (0, 0, 0), thickness)
        '''    
        for i, j in h_conn:
            overlay = _draw_conn(overlay, kp[i], kp[j], (0, 0, 255), thickness)            
        for i, j in l_conn:
            overlay = _draw_conn(overlay, kp[i], kp[j], (0, 0, 255), thickness)
        for i, j in r_conn:
            overlay = _draw_conn(overlay, kp[i], kp[j], (0, 0, 255), thickness)
        '''    
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
       
   

def get_keydata(name, frameNo, newkey, velocity, deg, deg_std, rate, angle, direct): 
    key_list = f"{name}, {frameNo}"              
    thiskey = newkey.reshape(-1)          
    for value in thiskey:
        key_list = f"{key_list}, {value}"         
    key_list = f"{key_list},  {newkey[0, 3]:5.2f}, {velocity:5.2f}, {deg:5.2f}, {deg_std:5.2f}, {rate:5.2f}, {angle:5.2f}, {direct:5d} \n"     
    return key_list

def get_speed(keynum):    
    velocity = np.max(keynum[1:5, 3], axis=0)
    deg = np.mean(keynum[1:5, 4], axis=0) # (5,6)
    #deg_std = np.round_(np.std(keynum[1:5, 4], axis=0)) # (5,6)   
    deg_std = 1 if -15 >= deg >= -165 else 0 
    return velocity, deg, deg_std

def get_motion(npArr, key, idx):  # 5 frame 이상의 속도 계산 
    retdata = np.zeros([5, 6])
    bgetdata = False
    key_num = [2, 5, 8, 11]     
    npArr[idx, 0, 0:3] = key[0, 1, :]
    for k, num in enumerate(key_num) :
        npArr[idx, k+1, 0:3] = key[0, num, :]               
        npArr[idx, k+1, 5] = idx 
        t2 = npArr[idx, k+1, 5]        
        m = 1;
        while True:
            j = (idx-m)%1800
            t1 = npArr[j, k+1, 5]            
            if (m > 150) :
                break
            if( t1 != 0 and t2-t1 >=10) :
                bgetdata = True
                break            
            m += 1         

        if(bgetdata == True)  :         
            x2, y2 = npArr[idx, k+1, 0], npArr[idx, k+1, 1]
            if npArr[j, k+1, 2] == 1:
                x1, y1 = npArr[j, k+1, 0], npArr[j, k+1, 1]
            else :
                x1, y1 = x2, y2
            x = x2 - x1
            y = -(y2 - y1)
            t = t2 - t1               
            npArr[idx, 0, 3] = t     
            v = np.sqrt( x**2 + y**2) / t    
            angle = math.degrees(math.atan2(y, x))        
            npArr[idx, k+1, 3] = v
            npArr[idx, k+1, 4] = angle
        else :
            npArr[idx, 0, 3]   = 0
            npArr[idx, k+1, 3] = 0.0
            npArr[idx, k+1, 4] = 0.0
    retdata = npArr[idx, :, :]    
    return retdata
    
def get_angle(keypoints):    
    angle = -1
    if len(keypoints) != 0:
        x2, x5, x8, x11 = keypoints[0][2][0], keypoints[0][5][0], keypoints[0][8][0], keypoints[0][11][0]
        y2, y5, y8, y11 = keypoints[0][2][1], keypoints[0][5][1], keypoints[0][8][1], keypoints[0][11][1]        
        x25  = (x2 + x5 ) / 2
        x811 = (x8 + x11) / 2
        y25  = (y2 + y5 ) / 2
        y811 = (y8 + y11) / 2
        x = x25 - x811
        y = -(y25 - y811)   # y축은 아래로 갈수록 증가(반대방향)
        angle = math.degrees(math.atan2(y,x))   # 직교좌표계에서의 각도      
    return angle

def is_bandyPosture(angle):
    pos = 0 if 135 > angle > 45 else 1
    return pos

def get_rate(keypoints):
    rate = -1    
    if len(keypoints) != 0:
        key_num = [2,5,8,11]       
        diff_key = np.max(keypoints[0, key_num, :], axis=0) - np.min(keypoints[0, key_num, :], axis=0)        
        retRate = diff_key[0]/diff_key[1]
    return retRate

def dir_check(_dir): #해당 디렉토리 내의 모든 avi파일 주소 목록
    file_list = glob.glob(_dir)
    for file_check in file_list:
        if os.path.isdir(file_check):
            dir_check("{}/*".format(file_check))
        elif file_check.endswith(".avi"):
            avi_list.append(file_check)




#video_path = "./examples/fall_test/*" #avi파일 찾을 위치
video_path = "./examples/media/*" #avi파일 찾을 위치
avi_list = []
dir_check(video_path)

with open('./fallen_data.csv', 'w') as f:
    data = 'fName, frameNo, x1, y1, 0, 0, 0, 0, x2, y2, val2, vec2, deg2, no, x5, y5, val5, vec5, deg5, no, ' 
    data += 'x8, y8, val8, vec8, deg8, no, x11, y11, val11, vec11, deg11, no, '
    data += 'f, speed, speed_deg, isDown, rate, angle, isBandy\n'
    f.write(data)
    estimator = BodyPoseEstimator(pretrained=True)  
   
    npArray    = np.zeros([1800, 5, 6])   #  x, y, val, vec, deg, no
    
    for video_fullpath in avi_list:
        output_name = f"{os.path.dirname(video_fullpath).split(os.path.sep)[-1]}_{os.path.basename(video_fullpath)}"            
        videoclip = cv2.VideoCapture(video_fullpath)        
        booleanStart = False
        booleanSuceed = False
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
                count += 1
                if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
                    break                  
                continue
            
            sum = 0
            for num in key_num:                  # key_num가 모두 유효하지 않은 frame skip
                sum += keypoints[0, num, 2]                        
            if sum != len(key_num) :                
                cv2.imshow('Video Demo', frame)
                count += 1            
                if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
                    break                  
                continue

            if booleanStart == False :
                booleanStart = True
                '''
                image = frame.copy();
                w, h, v = image.shape
                cv2.rectangle(image, (0,0) , (h, w), (255,255,255), -1 )
                '''
                count = 1

            newKey = get_motion(npArray, keypoints, (count%1800)) 
            sum = 0
            for num in range(4):                  # key_num가 움직이지 않는 frame skip
                sum += newKey[num, 3]  # 속도값
            if sum == 0 :                
                cv2.imshow('Video Demo', frame)
                count += 1 
                if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
                    break              
                continue

            try :       
                                                       
                frame = draw_body_conn(frame, keypoints, thickness=2, alpha=1)
                frame = draw_keypoints(frame, keypoints, radius=2, alpha=1)
                velocity, degree, deg_std = get_speed(newKey)
                rate    = get_rate(keypoints)                
                angle   = get_angle(keypoints)
                posture  = is_bandyPosture(angle)                 
                data    = get_keydata(output_name, count, newKey, velocity, degree, deg_std, rate, angle, direct)                
                f.write(data)
                booleanSuceed = 1 if ((velocity > 5 and deg_std > 0 and  rate > 0.7) or (velocity > 4 and rate >= 0.9) or (velocity > 4 and posture > 0)) else 0
                print(f"{output_name} No={count:4d} , {newKey[0, 3]:5.2f} , {velocity:5.2f} , {degree:5.2f} , {deg_std:5.2f} , {rate:5.2f} , {angle:5.2f}, {posture:5d} ==> {booleanSuceed} ")   
                cv2.imshow('Video Demo', frame)
                count +=1
                if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
                    break  
                              
            except KeyboardInterrupt:
                videoclip.release() 
                cv2.destroyAllWindows()
                f.close()
                exit(1)
            except Exception as e:
                print("에러 :", e)
                     
        videoclip.release()        
    cv2.destroyAllWindows()
f.close()
