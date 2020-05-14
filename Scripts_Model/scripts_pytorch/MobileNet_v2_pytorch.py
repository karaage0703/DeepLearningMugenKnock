import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from easydict import EasyDict
from _main_base import main
import os

#---
# config
#---
cfg = EasyDict()

# class
cfg.CLASS_LABEL = ['akahara', 'madara']
cfg.CLASS_NUM = len(cfg.CLASS_LABEL)

# model
cfg.INPUT_HEIGHT = 64
cfg.INPUT_WIDTH = 64
cfg.INPUT_CHANNEL = 3

cfg.GPU = False
cfg.DEVICE = torch.device("cuda" if cfg.GPU and torch.cuda.is_available() else "cpu")

cfg.MODEL_SAVE_PATH = 'models/MobileNet_v2_{}.pt'
cfg.MODEL_SAVE_INTERVAL = 200
cfg.ITERATION = 1000
cfg.MINIBATCH = 8
cfg.OPTIMIZER = torch.optim.SGD
cfg.LEARNING_RATE = 0.01
cfg.MOMENTUM = 0.9
cfg.LOSS_FUNCTION = loss_fn = torch.nn.NLLLoss()

cfg.TRAIN = EasyDict()
cfg.TRAIN.DISPAY_ITERATION_INTERVAL = 50

cfg.TRAIN.DATA_PATH = '../Dataset/train/images/'
cfg.TRAIN.DATA_HORIZONTAL_FLIP = True
cfg.TRAIN.DATA_VERTICAL_FLIP = True
cfg.TRAIN.DATA_ROTATION = False

cfg.TEST = EasyDict()
cfg.TEST.MODEL_PATH = cfg.MODEL_SAVE_PATH.format('final')
cfg.TEST.DATA_PATH = '../Dataset/test/images/'
cfg.TEST.MINIBATCH = 2

# random seed
torch.manual_seed(0)


class MobileNet_v2(torch.nn.Module): 
    def __init__(self):
        super(MobileNet_v2, self).__init__()
        
        # define block
        class MobileNetBlock(torch.nn.Module):
            def __init__(self, in_dim, out_dim, stride=1, expansion_t=6, split_division_by=8):
                super(MobileNetBlock, self).__init__()
                
                self.module = torch.nn.Sequential(
                    torch.nn.Conv2d(in_dim, in_dim * expansion_t, kernel_size=1, padding=0, stride=1, groups=in_dim),
                    torch.nn.BatchNorm2d(in_dim * expansion_t),
                    torch.nn.ReLU6(),
                    torch.nn.Conv2d(in_dim * expansion_t, in_dim * expansion_t, kernel_size=3, padding=1, stride=stride, groups=split_division_by),
                    torch.nn.BatchNorm2d(in_dim * expansion_t),
                    torch.nn.ReLU6(),
                    torch.nn.Conv2d(in_dim * expansion_t, out_dim, kernel_size=1, padding=0, stride=1),
                    torch.nn.BatchNorm2d(out_dim),
                )
                    
            def forward(self, _input):
                x = self.module(_input)
                
                # if shape matches, add skip connection
                if x.size() == _input.size():
                    x = x + _input
                
                return x
            
        # define feature dimension flattening layer
        class Flatten(torch.nn.Module):
            def forward(self, x):
                x = x.view(x.size()[0], -1)
                return x
        
        self.module = torch.nn.Sequential(
            # input
            # 224 x 224 x 3
            torch.nn.Conv2d(cfg.INPUT_CHANNEL, 32, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU6(),
            # 112 x 112 x 32
            MobileNetBlock(32, 16, expansion_t=1),
            # 112 x 112 x 16
            MobileNetBlock(16, 24, stride=2),
            MobileNetBlock(24, 24),
            # 56 x 56 x 24
            MobileNetBlock(24, 32, stride=2),
            MobileNetBlock(32, 32),
            MobileNetBlock(32, 32),
            # 28 x 28 x 32
            MobileNetBlock(32, 64, stride=2),
            MobileNetBlock(64, 64),
            MobileNetBlock(64, 64),
            MobileNetBlock(64, 64),
            # 14 x 14 x 64
            MobileNetBlock(64, 96),
            MobileNetBlock(96, 96),
            MobileNetBlock(96, 96),
            # 14 x 14 x 96
            MobileNetBlock(96, 160, stride=2),
            MobileNetBlock(160, 160),
            MobileNetBlock(160, 160),
            # 7 x 7 x 160
            MobileNetBlock(160, 320),
            # 7 x 7 x 320
            torch.nn.Conv2d(320, 1280, kernel_size=1, padding=0, stride=1),
            torch.nn.BatchNorm2d(1280),
            torch.nn.ReLU6(),
            # 7 x 7 x 1280
            torch.nn.AdaptiveAvgPool2d([1, 1]),
            Flatten(),
            # 1 x 1 x 1280
            torch.nn.Linear(1280, cfg.CLASS_NUM),
            torch.nn.Softmax(dim=1)
        )

        
    def forward(self, x):
        x = self.module(x)
        return x

# main
if __name__ == '__main__':

    model_save_dir = '/'.join(cfg.MODEL_SAVE_PATH.split('/')[:-1])
    os.makedirs(model_save_dir, exist_ok=True)

    main(cfg, MobileNet_v2())