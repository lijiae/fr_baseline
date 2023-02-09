import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

from model.resnet50 import ResNet50
from model.caam import Resnet50_Caam
import tensorboardX as tx

import torch
from torch import nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from data.imagedata import imagedataset,imagedataset_name



def makeargs():
    parse=argparse.ArgumentParser()
    parse.add_argument('--image_dir',type=str,default="/media/lijia/系统/data/train_align")
    parse.add_argument('--maad_path',type=str,default='/media/lijia/系统/data/vggface2/MAAD_Face.csv')
    parse.add_argument('--save_path',type=str,default='checkpoints/caam')
    parse.add_argument('--train_csv',type=str,default='/media/lijia/系统/data/vggface2/train_id_sample.csv')
    parse.add_argument('--test_csv',type=str,default='/media/lijia/系统/data/vggface2/test_id_sample.csv')
    parse.add_argument('--batch_size',type=int,default=64)
    parse.add_argument('-lr',type=float,default=0.001)
    parse.add_argument('--epoch',type=int,default=5)
    parse.add_argument('--idclass',type=int,default=8631)
    args=parse.parse_args()
    return args

def loadimage(args):
    train_csv=pd.read_csv(args.train_csv)
    train_dataset=imagedataset(args.image_dir,train_csv)
    train_dl=DataLoader(train_dataset,args.batch_size,True)
    test_csv=pd.read_csv(args.test_csv)
    test_dataset=imagedataset(args.image_dir,test_csv)
    test_dl=DataLoader(test_dataset,args.batch_size)
    return train_dl,test_dl

def main():
    # 加载
    args=makeargs()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    writer=tx.SummaryWriter('./log')

    # 读取数据
    train_dl,test_dl=loadimage(args)

    #读取模型
    print("load model...")
    model=Resnet50_Caam(args.idclass)
    # model.load_state_dict(torch.load("checkpoints/cbam_save/3_cbam.pth.tar")['state_dict'])
    print("load checkpoints...")
    model.cnn.load_state_dict(torch.load("/home/lijia/codes/202212/caam_face/checkpoints/0_classifier.pth.tar")['state_dict'],False)
    
    optimizer=torch.optim.SGD(model.parameters(),args.lr,momentum=0.9)
    schedule=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1,last_epoch=-1)
    cel=nn.CrossEntropyLoss()
    model.to(device)

    bs=0
    print("start training...")
    for e in range(args.epoch):
        print("训练到第{}个epoch".format(str(e)))
        losses=0
        timelosses=0
        model.train()
        for d in tqdm(train_dl):
            bs += 1
            y=model(d[0].to(device))
            loss=cel(y,d[1].to(device))
            optimizer.zero_grad()
            losses=losses+loss
            timelosses+=loss
            loss.backward()
            optimizer.step()
            if bs%2000==0:
                schedule.step()
                writer.add_scalar('cbam_loss/train',timelosses,int(bs/2000))
                timelosses=0
        writer.add_scalar('cbam_loss/traine',losses,e)
        torch.save({'epoch': e, 'state_dict': model.state_dict()},
                   os.path.join(args.save_path, str(e) + '_cbam.pth.tar'))
    total=0
    corr=0
    model.eval()
    for d in tqdm(test_dl):
        y=model(d[0].to(device))
        _,label=torch.max(y,1)
        total=total+label.size()[0]
        corr+=(d[1].to(device)==label).sum()
    print(float(corr)/float(total))
    writer.add_scalar('acc/test',float(corr)/float(total),e)

main()