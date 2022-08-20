import cv2
import numpy as np
import pyrealsense2 as rs2
from calculators.baseCalculator import BaseCalculator

class Line(BaseCalculator):
    def __init__(self):
        f=open('/home/cuichenxi/code/Python/extrinsics2/extrinsics.txt','r',encoding='utf-8')
        R_ = []
        t_ = []
        knt = 0
        for line in f:
            row = []
            line = line.strip()
            line = line.split(' ')
            row.append(float(line[0]))
            row.append(float(line[1]))
            row.append(float(line[2]))
            if (knt < 3):
                R_.append(row)
            else:
                t_.append(row)
            knt += 1
        self.R = np.array(R_).T
        self.t = -np.matmul(self.R,np.array(t_).reshape(3, 1))+np.array([-310,-360,58]).reshape(3, 1)


    def Process(self,img,**kwargs):
        '''
        :param img: colorframe
        :param kwargs: depthframe, intrinsics, faceresult, handsresult
        :return:
        '''
        depth=kwargs['depthframe']
        intrinsics=kwargs['intrinsics']
        faceresult=kwargs['faceresult']
        handsresult=kwargs['handsresult']

        headpoints3d = [0, 0, 0]
        if len(faceresult.result):
            headlandmarks=faceresult.result[0]['landmark']
            #两眼中心
            headpoint2d = (
                (headlandmarks[0] + headlandmarks[2]) / 2,
                (headlandmarks[1] + headlandmarks[3]) / 2
            )
            headpoint_depth = depth.get_distance(int(headpoint2d[0]), int(headpoint2d[1]))

            headpoint3d=rs2.rs2_deproject_pixel_to_point(intrin=intrinsics,pixel=headpoint2d,depth=headpoint_depth)
            print("head: {}".format(headpoint3d))

        handpoint3d = [0, 0, 0]
        if len(handsresult.result):
            handslandmark=handsresult.result[0]
            id, name, confidence, x, y, w, h = handslandmark
            handpoint2d=(x+w/2,y+h/2)
            handpoint_depth = depth.get_distance(int(handpoint2d[0]),int(handpoint2d[1]))
            handpoint3d=rs2.rs2_deproject_pixel_to_point(intrin=intrinsics,pixel=handpoint2d,depth=handpoint_depth)
            print("hand: {}".format(handpoint3d))

        headpoint3d = np.array(headpoint3d).reshape(3, 1)*1000
        handpoint3d = np.array(handpoint3d).reshape(3, 1)*1000
        headpoint3d_trans=np.matmul(self.R,headpoint3d)+self.t
        handpoint3d_trans=np.matmul(self.R,handpoint3d)+self.t
        print(headpoint3d_trans,'\n\n',handpoint3d_trans)





    def Draw(self,img,**kwargs):
        pass


