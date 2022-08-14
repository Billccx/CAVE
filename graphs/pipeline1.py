from calculators.faceDect import FaceDect
from calculators.handsDect import HandsDect
from calculators.poseDect import PoseDect
from calculators.visualizer import Visualizer
from calculators.HandsYolo.handsDectYolo import HandsDectYolo
from calculators.FaceYolo.faceDectYolo import FaceDectYolo

class Pipeline1:
    def __init__(self):
        self.face = FaceDectYolo()
        self.face2 = FaceDect()
        self.hands = HandsDectYolo()
        self.hands2 = HandsDect()
        self.pose = PoseDect()
        self.visual = Visualizer()
        self.calculators={'face':self.face,'hands':self.hands}
        self.results={}

    def forward(self,img):

        for key,calculator in self.calculators.items():
            result=calculator.Process(img)
            self.results[key]=result

        for key,calculator in self.calculators.items():
            calculator.Draw(img,result=self.results[key])



        #self.results.append(self.face.Process(img))
        #self.results.append(self.face2.Process(img))
        #self.results.append(self.hands.Process(img))
        #self.results.append(self.hands2.Process(img))
        #self.results.append(self.pose.Process(img))
        #self.visual.Process(self.results,img)
        self.results.clear()
        return img
