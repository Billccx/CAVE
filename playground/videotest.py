import cv2
import os
from cameras import Cameras
from calculators.HandsYolo.handsDectYolo import HandsDectYolo
from calculators.visualizer import Visualizer
import numpy as np
import time


def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)


cap=cv2.VideoCapture('basicvideo.mp4')
hands=HandsDectYolo()
vis=Visualizer()
cnt=0
f_count=0
t1=time.time()
while(cap.isOpened()):
    if f_count > 30:
        t1 = time.time()
        f_count = 0
    f_count += 1
    cnt += 1
    success, img = cap.read()
    result0 = hands.Process(img)
    frame0 = vis.Process([result0], img)

    fps = f_count / (time.time() - t1)
    cv2.putText(img, "FPS: %.2f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)

    #img_=gamma_trans(img,0.6)

    # (b, g, r) = cv2.split(img)
    # bH = cv2.equalizeHist(b)
    # gH = cv2.equalizeHist(g)
    # rH = cv2.equalizeHist(r)
    # img_ = cv2.merge((bH, gH, rH))



    # result=hands.Process(img_)
    # if len(result.result)==0:
    #     print("frame {} detect failed".format(cnt))
    # frame=vis.Process([result],img_)

    # frame2=cv2.hconcat([frame0,frame])
    # if (len(result0.result) == 0 and len(result.result) == 1):
    #     cv2.imwrite('compare/{}.jpg'.format(cnt), frame2)

    cv2.imshow('img', img)
    cv2.waitKey(5)
