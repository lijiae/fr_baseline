import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from model.resnet50 import ResNet50
from model.cbam import ResNet50_cbam
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
    # parse.add_argument('--maad_path',type=str,default='/media/lijia/系统/data/vggface2/MAAD_Face.csv')
    parse.add_argument('--save_path',type=str,default='checkpoints')
    parse.add_argument('--test_csv',type=str,default='/media/lijia/系统/data/vggface2/test_id_sample.csv')
    parse.add_argument('--batch_size',type=int,default=32)
    parse.add_argument('--epoch',type=int,default=20)
    parse.add_argument('--idclass',type=int,default=8631)
    args=parse.parse_args()
    return args


def load_image(args):
    train_csv=pd.read_csv(args.test_csv)
    dataset=imagedataset_name(args.image_dir,train_csv)
    dl=DataLoader(dataset,batch_size=args.batch_size)
    return dl

def main():
    # 加载
    args=makeargs()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # writer=tx.SummaryWriter('./log')

    # 读取数据
    dl=load_image(args)

    #读取模型
    model=ResNet50_cbam(args.idclass)
    model.load_state_dict(torch.load("checkpoints/cbam_save/4_cbam.pth.tar")['state_dict'])
    model.to(device)

    model.eval()
    corr=0
    result=[]
    names=[]
    total=0
    for d in tqdm(dl):
        y=model(d[0].to(device))
        _,label=torch.max(y,1)
        total=total+label.size()[0]
        corr+=(d[1].to(device)==label).sum()
        print(total)
        print(corr)
        print(corr/total)
        names=names+list(d[2])
        result += label.detach().cpu().numpy().tolist()

    ndf=pd.DataFrame({
        'Filename':names,
    })
    rdf=pd.DataFrame(result)
    df=ndf.join(rdf)
    df.to_csv('CBAM_testresult.csv',index=None)
    print("total:",total)
    print("corr:",corr)
    print(corr/total)


main()