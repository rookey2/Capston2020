import sys
import glob
import os
import cv2
import math
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints

sys.path.append('../')
import numpy as np


def get_mrate(keypoints):
    x = []
    y = []
    
    for i in range(15):
        if keypoints[0][i][2] != 0:
            x.append(keypoints[0][i][0])
            y.append(keypoints[0][i][1])

    max_x = max(x)
    min_x = min(x)
    max_y = max(y)
    min_y = min(y)

    return [min_x,min_y], [max_x,max_y]


def cv_print(text,img,pt1,pt2):
    overlay = img.copy()
    color = (255,0,0)
    line = 2
    text_size = 0.4
    tu1 = tuple(pt1)
    tu2 = tuple(pt2)
    cv2.rectangle(overlay,tu1,tu2,color,line) #프레임

    tu = (0,0)
    if pt1[1] < 40:
        pt1[1] = pt2[1] + 10
    else:
        pt1[1] -= 10
    tu = tuple(pt1)
    cv2.putText(overlay, text, tu , cv2.FONT_HERSHEY_SIMPLEX , text_size , color , thickness = 1) #텍스트
    return overlay

def fall_check(angle,rate):
    global num

    if rate >= 1 and 45 > angle:
        num , f_count = check_list(num,1)
        if f_count == "f":
            return "fall"
        else:
            return "stand"
    else :
        num , f_count = check_list(num,0)
        if f_count == "s":
            return "stand"
        else:
            return "fall"


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
        return num_c,"s"
    


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

def save_csv(keypoints,result):
    for i in range(15):
            max_list[i] += 1
    if keypoints.any() == False:
        return [0 for i in range(15)]
    else:
        key_list = []
        for idx ,check in enumerate(keypoints[0]):
            num = str(check[0]) + "," + str(check[1])
            key_list.append(num)
            # point_list[idx] += num

            if idx == 14:
                break
        if result:
            key_list.append(result)
            
        return key_list


video_path = "./examples/media/*" #avi파일 찾을 위치

avi_list = []
dir_check(video_path)
max_list   = [0 for _ in range(15)]
point_list = [0 for _ in range(15)]
ch_list    = [0 for _ in range(10)]
num = 0

save_list = [] #키포인트 값
amount_list = [] #키포인트 변화량

estimator = BodyPoseEstimator(pretrained=True)
f = open('d:/file.csv', 'w') #csv파일 저장위치
f.write("파일명,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14")
for video_path in avi_list:
    videoclip = cv2.VideoCapture(video_path)
    while videoclip.isOpened():
        result = False
        flag, frame = videoclip.read()
        if not flag:
            break

        keypoints = estimator(frame)
        

        pt1,pt2 = [0,0],[0,0]
        # print("기본" ,keypoints)
        
        if keypoints.any():
            pt1,pt2 = get_mrate(keypoints)
            angle   = get_angle(keypoints)
            rate    = get_rate(keypoints)
            
            p_list = []
            if len(save_list) > 10: #리스트 최대값
                save_list.pop(0)
            for i in range(15):
                key_list = []
                for j in range(3):
                    key_list.append(keypoints[0][i][j])
                p_list.append(key_list.copy())
            save_list.append(p_list.copy())

            f_result = fall_check(angle,rate)
            if f_result == "fall":
                frame = cv_print("Fallen person",frame,pt1,pt2)
            elif f_result == "stand":
                frame = cv_print("Standing person",frame,pt1,pt2)
            
        else:                   #keypoints 못찾았을때
            if save_list:
                result = [[[0 for i in range(3)]for j in range(18)]]
                s_len = len(save_list)

                for i in range(15):      #최근의 값에 이동한 거리의 평균을 더한값
                    x_sum = 0
                    y_sum = 0
                    x_avg = 0
                    y_avg = 0
                    x_result = 0
                    y_result = 0
                    count = 0
                    for j in range(s_len-1):
                        if save_list[j][i][2] == 1 and save_list[j+1][i][2] == 1:
                            x_sum += save_list[j+1][i][0] - save_list[j][i][0]
                            y_sum += save_list[j+1][i][1] - save_list[j][i][1]
                            count += 1

                    if count > 0:
                        x_avg = int(x_sum / count)
                        y_avg = int(y_sum / count)
                    
                    if save_list[-1][i][0] and save_list[-1][i][1] != 0:
                        x_result = save_list[-1][i][0] + x_avg
                        y_result = save_list[-1][i][1] + y_avg
                    result[0][i][0] = x_result
                    result[0][i][1] = y_result
                    result[0][i][2] = save_list[-1][i][2]

                keypoints = np.array(result)
                save_list.pop(0)

                result = True #csv 키포인트 추측값 확인

        if save_list and len(save_list) >= 2:
            if len(amount_list) > 10:
                amount_list.pop(0)

            avgall = []
            for i in range(15):
                amo = []
                
                if save_list[-1][i][2] != 0 and save_list[-2][i][2] != 0:
                    x_res = save_list[-1][i][0] - save_list[-2][i][0]
                    y_res = save_list[-1][i][1] - save_list[-2][i][1]
                else:
                    x_res = 0
                    y_res = 0
                amo.append(x_res)
                amo.append(y_res)
                avgall.append(amo.copy())
            amount_list.append(avgall.copy())

        print(amount_list)

                
                     
                    
        f.write("\n{},{}".format(os.path.basename(video_path),save_csv(keypoints,result)).replace("[",'').replace("]",''))

        frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
        frame = draw_keypoints(frame, keypoints, radius=3, alpha=0.8)
        
        cv2.imshow('Video Demo', frame)
        if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
            break
    videoclip.release()

# f.write("\n총횟수,{}".format(max_list).replace("[",'').replace("]",''))
# f.write("\n찾은횟수,{}".format(point_list).replace("[",'').replace("]",''))
f.close()
cv2.destroyAllWindows()

