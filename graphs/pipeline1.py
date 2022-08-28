from calculators.faceDect import FaceDect
from calculators.handsDect import HandsDect
from calculators.poseDect import PoseDect
from calculators.visualizer import Visualizer
from calculators.HandsYolo.handsDectYolo import HandsDectYolo
from calculators.FaceYolo.faceDectYolo import FaceDectYolo
from calculators.line import Line
import numpy as np

class Pipeline1:
    def __init__(self):
        self.face = FaceDectYolo()
        self.face2 = FaceDect()
        self.hands = HandsDectYolo()
        self.hands2 = HandsDect()
        self.pose = PoseDect()
        self.visual = Visualizer()
        self.line = Line()
        self.calculators={'face':self.face,'hands':self.hands,'line':self.line}
        self.results={}

    def forward(self,img,**kwargs):

        screen = np.zeros(shape=(720, 720, 3))

        for key,calculator in self.calculators.items():
            if(key=='face' or key=='hands'):
                result=calculator.Process(img)
                self.results[key]=result
            elif(key=='line'):
                result = calculator.Process(img,
                                            depthframe=kwargs['depthframe'],
                                            intrinsics=kwargs['intrinsics'],
                                            faceresult=self.results['face'],
                                            handsresult=self.results['hands'])
                self.results[key] = result


        for key,calculator in self.calculators.items():
            if(key=='line'):
                calculator.Draw(screen,result=self.results[key])
            else:
                calculator.Draw(img,result=self.results[key])



        #self.results.append(self.face.Process(img))
        #self.results.append(self.face2.Process(img))
        #self.results.append(self.hands.Process(img))
        #self.results.append(self.hands2.Process(img))
        #self.results.append(self.pose.Process(img))
        #self.visual.Process(self.results,img)
        self.results.clear()
        return img,screen
