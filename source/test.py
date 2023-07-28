import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import TestData
from model import WeakPolyp

class Test(object):
    def __init__(self, cfg):
        ## dataset
        self.cfg       = cfg 
        self.data      = TestData(cfg)
        self.loader    = DataLoader(self.data, batch_size=cfg.batch_size, pin_memory=False, shuffle=True, num_workers=cfg.num_workers)
        ## model
        self.model     = WeakPolyp(cfg).cuda()
        self.model.eval()

    def test_prediction(self):
        with torch.no_grad():
            mae, iou, dice, cnt = 0, 0, 0, 0
            for image, mask, name in self.loader:
                B, H, W         = mask.shape
                pred            = self.model(image.cuda().float())
                pred            = F.interpolate(pred, size=(H, W), mode='bilinear')
                pred            = (pred.squeeze()>0).cpu().float()
                cnt            += B
                mae            += np.abs(pred-mask).mean()
                inter, union    = (pred*mask).sum(dim=(1,2)), (pred+mask).sum(dim=(1,2))
                iou            += ((inter+1)/(union-inter+1)).sum()
                dice           += ((2*inter+1)/(union+1)).sum()
                print('cnt=%10d | mae=%.4f | dice=%.4f | iou=%.4f'%(cnt, mae/cnt, dice/cnt, iou/cnt))
            print('cnt=%10d | mae=%.4f | dice=%.4f | iou=%.4f'%(cnt, mae/cnt, dice/cnt, iou/cnt))


if __name__=='__main__':
    class Config:
        def __init__(self, backbone, testset):
            ## set the backbone type
            self.backbone       = backbone
            ## set the path of snapshot model
            self.snapshot       = self.backbone+'/model.pth'
            ## set the path of testing dataset
            self.test_image     = ['../dataset/SUN-SEG-Processed/'+testset+'/Seen/Frame', '../dataset/SUN-SEG-Processed/'+testset+'/Unseen/Frame']
            self.test_mask      = ['../dataset/SUN-SEG-Processed/'+testset+'/Seen/GT'   , '../dataset/SUN-SEG-Processed/'+testset+'/Unseen/GT'   ]
            ## other settings
            self.mode           = 'test'
            self.batch_size     = 64
            self.num_workers    = 4

    os.environ ["CUDA_VISIBLE_DEVICES"] = '0' 
    Test(Config('res2net50', 'TestEasyDataset')).test_prediction()
    Test(Config('res2net50', 'TestHardDataset')).test_prediction()
    # Test(Config('pvt_v2_b2', 'TestEasyDataset')).test_prediction()
    # Test(Config('pvt_v2_b2', 'TestHardDataset')).test_prediction()