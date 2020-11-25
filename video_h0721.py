import sys
import glob
import os
import math
sys.path.append('../')

import cv2
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


def cv_print(text,frame,alpha = 1.0,thickness = 2):
    overlay = frame.copy()
    cv2.putText(overlay, text, (30 , 30) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , (200,100,100) , thickness)
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

def fall_check(angle,rate):
    global num

    if rate >= 1 and 45 > angle:
        num , f_count = check_list(num,1)
        if f_count == "f":
            return "fall"
        else:
            return "nor"
    else :
        num , f_count = check_list(num,0)
        if f_count == "n":
            return "nor"
        else:
            return "fall"

    print(angle)
    print(rate)

def check_list(num,check = 0):
    ch_list[num] = check
    fcount = ch_list.count(1)  #기존에 넘어진 횟수
    num_c = num
    
    if num == 9:
        num_c = 0
    else:
        num_c += 1

    if fcount > 5:
        return num_c,"f"
    elif fcount <= 5:
        return num_c,"n"
    


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
    elif 0 > angle >= -90:
        angle = -angle
    elif -90 > angle:
        angle = 180 + angle

    return angle

def get_rate(keypoints):
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

def save_csv(keypoints):
    for i in range(15):
            max_list[i] += 1
    if len(keypoints) == 0:
        return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    else:
        key_list = []
        for idx ,check in enumerate(keypoints[0]):
            num = check[2]
            key_list.append(num)
            point_list[idx] += num

            if idx == 14:
                break
            
        return key_list


video_path = "E:/work/python/Fallen/openpose-pytorch-master/examples/fall_test/Coffee_room/*" #avi파일 찾을 위치

avi_list = []
dir_check(video_path)
max_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
point_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

ch_list = [0,0,0,0,0,0,0,0,0,0]
num = 0

estimator = BodyPoseEstimator(pretrained=True)

# f = open('d:/file2.csv', 'w') #csv파일 저장위치
# f.write("파일명,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14")


for video_path in avi_list:
    videoclip = cv2.VideoCapture(video_path)
    while videoclip.isOpened():
        flag, frame = videoclip.read()
        if not flag:
            break

        keypoints = estimator(frame)

        frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
        frame = draw_keypoints(frame, keypoints, radius=3, alpha=0.8)
        
        # if keypoints is not None:
        #     f.write("\n{},{}".format(os.path.basename(video_path),save_csv(keypoints)).replace("[",'').replace("]",''))
        #     print(keypoints)

        if keypoints.any():
            angle = get_angle(keypoints)
            rate  = get_rate(keypoints)
            f_result = fall_check(angle,rate)
            if f_result == "fall":
                frame = cv_print("fall",frame)
            elif f_result == "nor":
                frame = cv_print("nor",frame)
        else:
            num,f_count = check_list(num)
            if f_count == "f":
                frame = cv_print("fall",frame)
            elif f_count =="n":
                frame = cv_print("nor",frame)
        
        
        
        cv2.imshow('Video Demo', frame)
        if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
            break
    videoclip.release()

# f.write("\n총횟수,{}".format(max_list).replace("[",'').replace("]",''))
# f.write("\n찾은횟수,{}".format(point_list).replace("[",'').replace("]",''))
# f.close()
cv2.destroyAllWindows()

