import numpy as np

from model.cbam import *
from model.resnet50 import *
import torch
import torch.nn as nn

class DModule(nn.Module):
    def __init__(self,in_ch):
        super(DModule,self).__init__()
        self.cbam=cbam_Module(in_ch)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x,c_pre=None,s_pre=None):
        z=self.cbam(x)
        c=self.sigmoid(z)*x+c_pre
        s=self.sigmoid(-z)*x+s_pre
        return c,s
    
class MModule(nn.Module):
    def __init__(self,in_ch):
        super(MModule,self).__init__()
        self.mergeConv2d=nn.Conv2d(in_ch,in_ch,1,stride=1)
        
    def forward(self, c,s):
        return self.mergeConv2d(c)+self.mergeConv2d(s)
    
class Caam_Module(nn.Module):
    def __init__(self,in_ch):
        super(Caam_Module,self).__init__()
        self.DModule=DModule(in_ch)
        self.MModule=MModule(in_ch)
        
    def forward(self, c,s):
        x=self.MModule(c,s)
        c_now,s_now=self.DModule(x,c,s)
        c_now=c_now+c
        s_now=s_now+s
        return c_now,s_now
        
        
class Resnet50_Caam(nn.Module):
    def __init__(self,numclass,M_circle=4):
        super(Resnet50_Caam,self).__init__()
        self.cnn=ResNet50_Layers()
        self.DModule=DModule(2048)

        self.M=M_circle
        # self.caam_module=Caam_Module(2048)
        self.CaamModule=nn.ModuleList(
            self.makeCaamList()
        )
        self.average=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(2048,numclass)
        
    def makeCaamList(self):
        caamlist=[]
        for _ in range(self.M-1):
            caamlist.append(Caam_Module(2048))
        return caamlist
        
    def forward(self, x):
        x=self.cnn(x)
        c_pre=torch.tensor(np.ones(x.shape),dtype=torch.float32).to(x.device)
        s_pre=torch.tensor(np.ones(x.shape),dtype=torch.float32).to(x.device)

        c,s=self.DModule(x,c_pre,s_pre)
        # for i in range(self.M-1):
        #     c,s=self.CaamModule[i](c,s)
        for i,cm in enumerate(self.CaamModule):
            c,s=self.CaamModule[i](c,s)
        # c,s=self.CaamModule(x)
        output=self.average(c)
        output=self.fc(output.reshape(output.shape[:2]))
        return output
        
        
        
        
        