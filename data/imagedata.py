from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision
from PIL import Image
import os

class imagedataset(Dataset):
    def __init__(self,imagepath,idfile):
        self.mean_bgr=np.array([91.4953, 103.8827, 131.0912])
        self.idfile=idfile
        self.dir=imagepath

    def __len__(self):
        return len(self.idfile)

    def __getitem__(self, index):
        # Sample
        sample=self.idfile.iloc[index]

        # data and label information
        imgname=sample['Filename']
        id=sample["id"]
        label=torch.tensor(int(id)).long()
        data = torchvision.transforms.Resize(224)(Image.open(os.path.join(self.dir,imgname)))
        data = np.array(data, dtype=np.uint8)
        data = self.transform(data)

        return data.float(), label

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

class imagedataset_name(Dataset):
    def __init__(self,imagepath,idfile):
        self.mean_bgr=np.array([91.4953, 103.8827, 131.0912])
        self.idfile=idfile
        self.dir=imagepath

    def __len__(self):
        return len(self.idfile)

    def __getitem__(self, index):
        # Sample
        sample=self.idfile.iloc[index]

        # data and label information
        imgname=sample['Filename']
        id=sample["id"]
        label=torch.tensor(int(id)).long()
        data = torchvision.transforms.Resize(224)(Image.open(os.path.join(self.dir,imgname)))
        data = np.array(data, dtype=np.uint8)
        data = self.transform(data)

        return data.float(), label,imgname

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img