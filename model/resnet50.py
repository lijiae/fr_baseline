import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cbam import *
class Conv(nn.Module):
    def __init__(self,in_ch,out_ch,k_size=1,stride=1,padding=None,relu=True):
        super(Conv,self).__init__()
        padding=k_size//2 if padding is None else padding
        self.conv=nn.Conv2d(in_ch,out_ch,k_size,stride,padding,bias=False)
        self.bn=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        output=self.conv(x)
        output=self.bn(output)
        output=self.relu(output)
        return output

class BottleBlock(nn.Module):
    def __init__(self,in_ch,out_ch,down_sample=False):
        super(BottleBlock,self).__init__()
        stride=2 if down_sample else 1
        mid_ch=out_ch//4
        self.shortcut=Conv(in_ch,out_ch,stride=stride,relu=False) if in_ch!=out_ch else nn.Identity()
        self.conv=nn.Sequential(*[
            Conv(in_ch,mid_ch),
            Conv(mid_ch,mid_ch,k_size=3,stride=stride),
            Conv(mid_ch,out_ch,relu=False)
        ])

    def forward(self, x):
        output=self.conv(x)+self.shortcut(x)
        return F.relu(output,inplace=True)

class ResNet50_Layers(nn.Module):
    def __init__(self):
        super(ResNet50_Layers,self).__init__()
        self.stem=nn.Sequential(*[
            Conv(3,64,k_size=7,stride=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        ])
        self.stages=nn.Sequential(*[
            self.make_stages(64,256,down_samples=False,num_blocks=3),
            self.make_stages(256,512,True,4),
            self.make_stages(512,1024,True,6),
            self.make_stages(1024,2048,True,3)
        ])
    def make_stages(self,in_ch,out_ch,down_samples,num_blocks):
        layers=[BottleBlock(in_ch,out_ch,down_samples)]
        for _ in range(num_blocks-1):
            layers.append(BottleBlock(out_ch,out_ch,down_sample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        output=self.stem(x)
        output=self.stages(output)
        # output=self.average(output)
        # output=self.fc(output.reshape(output.shape[:2]))
        return output

class ResNet50(nn.Module):
    def __init__(self,numclass):
        super(ResNet50,self).__init__()
        self.stem=nn.Sequential(*[
            Conv(3,64,k_size=7,stride=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        ])
        self.stages=nn.Sequential(*[
            self.make_stages(64,256,down_samples=False,num_blocks=3),
            self.make_stages(256,512,True,4),
            self.make_stages(512,1024,True,6),
            self.make_stages(1024,2048,True,3)
        ])
        self.average=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(2048,numclass)

    def make_stages(self,in_ch,out_ch,down_samples,num_blocks):
        layers=[BottleBlock(in_ch,out_ch,down_samples)]
        for _ in range(num_blocks-1):
            layers.append(BottleBlock(out_ch,out_ch,down_sample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        output=self.stem(x)
        output=self.stages(output)
        output=self.average(output)
        output=self.fc(output.reshape(output.shape[:2]))
        return output
    
class ResNet50_cbam(nn.Module):
    def __init__(self,numclass):
        super(ResNet50_cbam,self).__init__()
        self.stem=nn.Sequential(*[
            Conv(3,64,k_size=7,stride=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        ])
        self.stages=nn.Sequential(*[
            self.make_stages(64,256,down_samples=False,num_blocks=3),
            self.make_stages(256,512,True,4),
            self.make_stages(512,1024,True,6),
            self.make_stages(1024,2048,True,3)
        ])
        self.average=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(2048,numclass)
        self.cam=ChannelAttentionModule(2048)
        self.sam=SpatialAttentionModule()

    def make_stages(self,in_ch,out_ch,down_samples,num_blocks):
        layers=[BottleBlock(in_ch,out_ch,down_samples)]
        for _ in range(num_blocks-1):
            layers.append(BottleBlock(out_ch,out_ch,down_sample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        output=self.stem(x)
        output=self.stages(output)
        cam=self.cam(output)
        output=cam*output
        sam=self.sam(output)
        output=sam*output
        output=self.average(output)
        output=self.fc(output.reshape(output.shape[:2]))
        return output