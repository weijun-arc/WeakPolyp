import os
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class TrainData(Dataset):
    def __init__(self, cfg):
        self.samples   = []
        for folder in os.listdir(cfg.train_image):
            for name in os.listdir(cfg.train_image+'/'+folder):
                image = cfg.train_image+'/'+folder+'/'+name
                mask  = cfg.train_mask +'/'+folder+'/'+name.replace('.jpg', '.png')
                self.samples.append((image, mask))
        
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]
        image, mask           = cv2.imread(image_name), cv2.imread(mask_name)
        image, mask           = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), np.float32(mask>128)
        pair                  = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'].permute(2,0,1)

    def __len__(self):
        return len(self.samples)


class TestData(Dataset):
    def __init__(self, cfg):
        self.samples  = []
        self.cfg      = cfg
        for test_image, test_mask in zip(cfg.test_image, cfg.test_mask):
            for folder in os.listdir(test_image):
                for name in os.listdir(test_image+'/'+folder):
                    image = test_image+'/'+folder+'/'+name
                    mask  = test_mask+'/'+folder+'/'+name.replace('.jpg', '.png')
                    self.samples.append((image, mask))
        print('Test Data: %s,   Test Samples: %s'%(cfg.test_image, len(self.samples)))

        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(320, 320),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]
        image, mask           = cv2.imread(image_name), cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        image, mask           = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), np.float32(mask>128)
        pair                  = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'], mask_name

    def __len__(self):
        return len(self.samples)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.PReLU)):
            pass
        else:
            m.initialize()


def preprocess(path_src):
    print('process', path_src)
    path_dst = path_src.replace('/SUN-SEG/', '/SUN-SEG-Processed/')
    for folder in os.listdir(path_src+'/Frame'):
        print(folder)
        for name in os.listdir(path_src+'/Frame/'+folder):
            image    = cv2.imread(path_src+'/Frame/'+folder+'/'+name)
            image    = cv2.resize(image, (352,352), interpolation=cv2.INTER_LINEAR)
            mask     = cv2.imread(path_src+'/GT/'+folder+'/'+name.replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE)
            mask     = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            box      = np.zeros_like(mask)
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                box[y:y+h, x:x+w] = 255
            
            os.makedirs(path_dst+'/Frame/'+folder, exist_ok=True)
            cv2.imwrite(path_dst+'/Frame/'+folder+'/'+name, image)
            os.makedirs(path_dst+'/GT/'   +folder, exist_ok=True)
            cv2.imwrite(path_dst+'/GT/'   +folder+'/'+name.replace('.jpg', '.png'), mask)
            os.makedirs(path_dst+'/Box/'  +folder, exist_ok=True)
            cv2.imwrite(path_dst+'/Box/'  +folder+'/'+name.replace('.jpg', '.png'), box)