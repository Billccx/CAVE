import pyrealsense2 as rs
import os
import cv2
import numpy as np
import mediapipe as mp
from cameras import Cameras,CameraIntrinsics
from graphs.pipeline1 import Pipeline1
from calculators.HandsYolo.handsDectYolo import HandsDectYolo




if __name__=='__main__':
    mp_drawing = mp.solutions.drawing_utils
    pipe = Pipeline1()
    cams = Cameras()
    cams.captureRGB(0)
    while (True):
        img = cams.getRGBFrame(0)
        img = pipe.forward(img)

        cv2.imshow('img', img)
        if cv2.waitKey(5) & 0xFF == 27:
            break



    # mp_drawing = mp.solutions.drawing_utils
    # cams=Cameras()
    # cams.captureRGB(0)
    # #pipe=Pipeline1()
    #
    # while (True):
    #     img = cams.getRGBFrame(0)
    #     #img=pipe.forward(img)
    #
    #     cv2.imshow(img)
    #     cv2.waitKey(0)
