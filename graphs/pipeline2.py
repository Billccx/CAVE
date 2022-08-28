from calculators.faceDect import FaceDect
from calculators.handsDect import HandsDect
from calculators.poseDect import PoseDect
from calculators.visualizer import Visualizer
from calculators.HandsYolo.handsDectYolo import HandsDectYolo
from calculators.FaceYolo.faceDectYolo import FaceDectYolo
from calculators.line2 import Line2
import numpy as np
import cv2

class Pipeline2:
    def __init__(self):
        self.face0 = FaceDectYolo()
        self.hands0 = HandsDectYolo()

        self.face1 = FaceDectYolo()
        self.hands1 = HandsDectYolo()

        self.line0 = Line2('extrinsics0.txt')
        self.line1 = Line2('extrinsics1.txt')

        self.calculators={
            'face0':self.face0,
            'face1':self.face1,
            'hands0':self.hands0,
            'hands1':self.hands1,
            'line0':self.line0,
            'line1':self.line1
        }
        self.results={}

    def forward(self,img0,img1,**kwargs):

        screen = np.zeros(shape=(500, 500, 3))

        for key,calculator in self.calculators.items():
            if(key=='face0' or key=='hands0'):
                result=calculator.Process(img0)
                self.results[key]=result
            elif (key == 'face1' or key == 'hands1'):
                result = calculator.Process(img1)
                self.results[key] = result
            elif(key=='line0'):
                result = calculator.Process(img0,
                                            depthframe=kwargs['depthframe0'],
                                            intrinsics=kwargs['intrinsics0'],
                                            faceresult=self.results['face0'],
                                            handsresult=self.results['hands0'])
                self.results[key] = result
            elif (key == 'line1'):
                result = calculator.Process(img1,
                                            depthframe=kwargs['depthframe1'],
                                            intrinsics=kwargs['intrinsics1'],
                                            faceresult=self.results['face1'],
                                            handsresult=self.results['hands1'])
                self.results[key] = result


        for key,calculator in self.calculators.items():
            if (key=='line0'):
                calculator.Draw(screen,result=self.results[key],index=0)
            elif (key == 'line1'):
                calculator.Draw(screen, result=self.results[key],index=1)
            elif(key[-1]=='0'):
                calculator.Draw(img0,result=self.results[key])
            elif (key[-1] == '1'):
                calculator.Draw(img1, result=self.results[key])

        img=cv2.hconcat([img0,img1])

        self.results.clear()
        return img,screen
