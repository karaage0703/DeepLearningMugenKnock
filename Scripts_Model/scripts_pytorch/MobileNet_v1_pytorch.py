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

cfg.MODEL_SAVE_PATH = 'models/MobileNet_v1_{}.pt'
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


class MobileNet_v1(torch.nn.Module): 
    def __init__(self):
        super(MobileNet_v1, self).__init__()

        class MobileNetBlock(torch.nn.Module):
            def __init__(self, in_dim, out_dim, repeat=1, stride=1):
                super(MobileNetBlock, self).__init__()
                _module = []
                for _ in range(repeat):
                    _module += [
                        torch.nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=stride, groups=in_dim),
                        torch.nn.BatchNorm2d(in_dim),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, stride=1),
                        torch.nn.BatchNorm2d(out_dim),
                        torch.nn.ReLU(),
                    ]
                    
                self.module = torch.nn.Sequential(*_module)
                    
            def forward(self, x):
                x = self.module(x)
                return x
            
        class Flatten(torch.nn.Module):
            def forward(self, x):
                x = x.view(x.size()[0], -1)
                return x
        
        self.module = torch.nn.Sequential(
            #-----
            # 1/1 x 1/1 x 3
            #-----
            torch.nn.Conv2d(cfg.INPUT_CHANNEL, 32, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            #-----
            # 1/2 x 1/2 x 32
            #-----
            MobileNetBlock(32, 64),

            #-----
            # 1/4 x 1/4 x 64
            #-----
            MobileNetBlock(64, 128, stride=2),
            MobileNetBlock(128, 128),

            #-----
            # 1/8 x 1/8 x 128
            #-----
            MobileNetBlock(128, 256, stride=2),
            MobileNetBlock(256, 256),

            #-----
            # 1/16 x 1/16 x 256
            #-----
            MobileNetBlock(256, 512, stride=2),
            MobileNetBlock(512, 512, repeat=5),
            
            #-----
            # 1/32 x 1/32 x 1024
            #-----
            MobileNetBlock(512, 1024, stride=2),
            MobileNetBlock(1024, 1024),
            #torch.nn.AvgPool2d([img_height // 32, img_width // 32], stride=1, padding=0),
            torch.nn.AdaptiveAvgPool2d([1, 1]),
            Flatten(),
            torch.nn.Linear(1024, cfg.CLASS_NUM),
            torch.nn.Softmax(dim=1)
        )

        
    def forward(self, x):
        x = self.module(x)
        return x

# main
if __name__ == '__main__':

    model_save_dir = '/'.join(cfg.MODEL_SAVE_PATH.split('/')[:-1])
    os.makedirs(model_save_dir, exist_ok=True)

    main(cfg, MobileNet_v1())