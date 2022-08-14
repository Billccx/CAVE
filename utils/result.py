import torch

class Result:
    def __init__(self,info):
        self.info=info

    def setResult(self,result):
        self.result=result

    def getInfo(self):
        return self.info
