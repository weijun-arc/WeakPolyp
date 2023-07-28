import os
import sys
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import WeakPolyp
from utils import TrainData, TestData, clip_gradient, preprocess

class Train:
    def __init__(self, cfg):
        ## parameter
        self.cfg        = cfg
        self.logger     = SummaryWriter(cfg.log_path)
        logging.basicConfig(level=logging.INFO, filename=cfg.save_path+'/train.log', filemode='a', format='[%(asctime)s | %(message)s]', datefmt='%I:%M:%S')
        ## model
        self.model      = WeakPolyp(cfg).cuda()
        self.model.train()
        ## data
        self.data       = TrainData(cfg)
        self.loader     = DataLoader(dataset=self.data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        ## optimizer
        base, head      = [], []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                base.append(param)
            else:
                head.append(param)
        self.optimizer = torch.optim.SGD([{'params':base, 'lr':0.1*cfg.lr}, {'params':head, 'lr':cfg.lr}], momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
        self.max_dice   = 0


    def forward(self):
        global_step    = 0
        scaler         = torch.cuda.amp.GradScaler()
        for epoch in range(self.cfg.epoch):
            if epoch in [3, 6, 9, 12]:
                self.optimizer.param_groups[0]['lr'] *= 0.5
            for i, (image, mask) in enumerate(self.loader):
                with torch.cuda.amp.autocast():
                    ## pred 1
                    image, mask    = image.cuda(), mask.cuda()
                    size1          = np.random.choice([256, 288, 320, 352, 384, 416, 448])
                    image1         = F.interpolate(image, size=size1, mode='bilinear')
                    pred1          = self.model(image1)
                    pred1          = F.interpolate(pred1, size=352, mode='bilinear')
                    ## pred 2
                    size2          = np.random.choice([256, 288, 320, 352, 384, 416, 448])
                    image2         = F.interpolate(image, size=size2, mode='bilinear')
                    pred2          = self.model(image2)
                    pred2          = F.interpolate(pred2, size=352, mode='bilinear')
                    ## loss_sc
                    loss_sc        = (torch.sigmoid(pred1)-torch.sigmoid(pred2)).abs()
                    loss_sc        = loss_sc[mask[:,0:1]==1].mean()
                    ## M2B transformation
                    pred           = torch.cat([pred1, pred2], dim=0)
                    mask           = torch.cat([mask, mask], dim=0)
                    predW, predH   = pred.max(dim=2, keepdim=True)[0], pred.max(dim=3, keepdim=True)[0]
                    pred           = torch.minimum(predW, predH)
                    pred, mask     = pred[:,0], mask[:,0]
                    ## loss_ce + loss_dice 
                    loss_ce        = F.binary_cross_entropy_with_logits(pred, mask)
                    pred           = torch.sigmoid(pred)
                    inter          = (pred*mask).sum(dim=(1,2))
                    union          = (pred+mask).sum(dim=(1,2))
                    loss_dice      = 1-(2*inter/(union+1)).mean()
                    loss           = loss_ce + loss_dice + loss_sc

                ## backward
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                clip_gradient(self.optimizer, self.cfg.clip)
                scaler.step(self.optimizer)
                scaler.update()

                global_step += 1
                self.logger.add_scalar('lr'  , self.optimizer.param_groups[0]['lr'], global_step=global_step)
                self.logger.add_scalars('loss', {'ce':loss_ce.item(), 'dice':loss_dice.item(), 'sc':loss_sc.item()}, global_step=global_step)
                ## print loss
                if global_step % 20 == 0:
                    print('{} epoch={:03d}/{:03d}, step={:04d}/{:04d}, loss_ce={:0.4f}, loss_dice={:0.4f}, loss_sc={:0.4f}'.format(datetime.now(), epoch, self.cfg.epoch, i, len(self.loader), loss_ce.item(), loss_dice.item(), loss_sc.item()))
            self.evaluate(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            data                = TestData(self.cfg)
            loader              = DataLoader(dataset=data, batch_size=64, shuffle=False, num_workers=self.cfg.num_workers)
            dice, iou, cnt      = 0, 0, 0
            for image, mask, name in tqdm(loader):
                image, mask     = image.cuda().float(), mask.cuda().float()
                B, H, W         = mask.shape
                pred            = self.model(image)
                pred            = F.interpolate(pred, size=(H, W), mode='bilinear')
                pred            = (pred.squeeze()>0)
                inter, union    = (pred*mask).sum(dim=(1,2)), (pred+mask).sum(dim=(1,2))
                dice           += ((2*inter+1)/(union+1)).sum().cpu().numpy()
                iou            += ((inter+1)/(union-inter+1)).sum().cpu().numpy()
                cnt            += B
            logging.info('epoch=%-8d | dice=%.4f | iou=%.4f | path=%s'%(epoch, dice/cnt, iou/cnt, self.cfg.test_image))

        if dice/cnt>self.max_dice:
            self.max_dice = dice/cnt
            torch.save(self.model.state_dict(), self.cfg.backbone+'/model.pth')
        self.model.train()



if __name__=='__main__':
    ## preprocess the dataset
    if not os.path.exists('../dataset/SUN-SEG-Processed'):
        preprocess('../dataset/SUN-SEG/TrainDataset')
        preprocess('../dataset/SUN-SEG/TestEasyDataset/Seen')
        preprocess('../dataset/SUN-SEG/TestEasyDataset/Unseen')
        preprocess('../dataset/SUN-SEG/TestHardDataset/Seen')
        preprocess('../dataset/SUN-SEG/TestHardDataset/Unseen')

    ## hyperparameter config
    class Config:
        def __init__(self, backbone):
            ## set the backbone type
            self.backbone       =  backbone
            ## set the path of training dataset
            self.train_image    = '../dataset/SUN-SEG-Processed/TrainDataset/Frame'
            self.train_mask     = '../dataset/SUN-SEG-Processed/TrainDataset/Box'
            ## set the path of testing dataset
            self.test_image     = ['../dataset/SUN-SEG-Processed/TestHardDataset/Seen/Frame', '../dataset/SUN-SEG-Processed/TestHardDataset/Unseen/Frame']
            self.test_mask      = ['../dataset/SUN-SEG-Processed/TestHardDataset/Seen/GT'   , '../dataset/SUN-SEG-Processed/TestHardDataset/Unseen/GT'   ]
            ## set the path of logging
            self.log_path       = self.backbone+'/log'
            os.makedirs(self.log_path, exist_ok=True)
            ## keep unchanged
            if self.backbone=='res2net50':
                self.mode           = 'train'
                self.epoch          = 16
                self.batch_size     = 16
                self.lr             = 0.1
                self.num_workers    = 4
                self.weight_decay   = 1e-3
                self.clip           = 0.5
            if self.backbone=='pvt_v2_b2':
                self.mode           = 'train'
                self.epoch          = 16
                self.batch_size     = 16
                self.lr             = 0.1
                self.num_workers    = 4
                self.weight_decay   = 1e-4
                self.clip           = 1000

    ## training
    os.environ ["CUDA_VISIBLE_DEVICES"] = '0'
    Train(Config('res2net50')).forward()
    # os.environ ["CUDA_VISIBLE_DEVICES"] = '1' 
    # Train(Config('pvt_v2_b2')).forward()
