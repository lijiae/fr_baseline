import torch
from torch import nn

class ChannelAttentionModule(nn.Module):
    def __init__(self,ch,ratio=16):
        super(ChannelAttentionModule,self).__init__()
        self.AvgPool=nn.AdaptiveAvgPool2d(1)
        self.MaxPool=nn.AdaptiveMaxPool2d(1)
        self.share_MLP=nn.Sequential(
            nn.Conv2d(ch,ch//ratio,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(ch//ratio,ch,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        avgout=self.share_MLP(self.AvgPool(x))
        maxout=self.share_MLP(self.MaxPool(x))
        return self.sigmoid(maxout+avgout)
    

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule,self).__init__()
        self.conv2d=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=7,stride=1,padding=3)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        avgout=torch.mean(x,dim=1,keepdim=True)
        maxout,_=torch.max(x,dim=1,keepdim=True)
        out=torch.cat([avgout,maxout],dim=1)
        out=self.sigmoid(self.conv2d(out))
        return out
    
class cbam_Module(nn.Module):
    def __init__(self,in_ch):
        super(cbam_Module,self).__init__()
        self.cam=ChannelAttentionModule(in_ch)
        self.sam=SpatialAttentionModule()
        
    def forward(self, x):
        ca=self.cam(x)
        out=x*ca
        sa=self.sam(out)
        return sa*out