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

cfg.MODEL_SAVE_PATH = 'models/Xception_{}.pt'
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


class Xception(torch.nn.Module):
    def __init__(self):
        super(Xception, self).__init__()

        class Block(torch.nn.Module):
            def __init__(self, dim=728, cardinality=1):
                super(Block, self).__init__()

                self.block = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=cardinality),
                    torch.nn.BatchNorm2d(dim),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=cardinality),
                    torch.nn.BatchNorm2d(dim),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=cardinality),
                    torch.nn.BatchNorm2d(dim),
                )
                
            def forward(self, x):
                res_x = self.block(x)            
                x = torch.add(res_x, x)
                return x

        # Entry flow
        self.conv1 = torch.nn.Conv2d(cfg.INPUT_CHANNEL, 32, kernel_size=3, padding=1, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(3, stride=2, padding=1))
        self.conv3_sc = torch.nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=2)
        self.bn3_sc = torch.nn.BatchNorm2d(128)
        
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(3, stride=2, padding=1))
        self.conv4_sc = torch.nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=2)
        self.bn4_sc = torch.nn.BatchNorm2d(256)
        
        self.conv5 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 728, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(728),
            torch.nn.ReLU(),
            torch.nn.Conv2d(728, 728, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(728),
            torch.nn.MaxPool2d(3, stride=2, padding=1))
        self.conv5_sc = torch.nn.Conv2d(256, 728, kernel_size=1, padding=0, stride=2)
        self.bn5_sc = torch.nn.BatchNorm2d(728)
        
        # Middle flow
        self.middle_flow = torch.nn.Sequential(
            *[Block() for _ in range(8)]
        )
        
        # Exit flow
        self.conv_exit1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(728, 728, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(728),
            torch.nn.ReLU(),
            torch.nn.Conv2d(728, 1024, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.MaxPool2d(3, stride=2, padding=1))
        self.conv_exit1_sc = torch.nn.Conv2d(728, 1024, kernel_size=1, padding=0, stride=2)
        self.bn_exit1_sc = torch.nn.BatchNorm2d(1024)
        
        self.conv_exit2 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 1536, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(1536),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1536, 2048, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(2048),)
        
        self.linear = torch.nn.Linear(2048, cfg.CLASS_NUM)
        
        
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x_sc = self.conv3_sc(x)
        x_sc = self.bn3_sc(x_sc)
        x = self.conv3(x)
        x = torch.add(x_sc, x)
        
        x_sc = self.conv4_sc(x_sc)
        x_sc = self.bn4_sc(x_sc)
        x = self.conv4(x)
        x = torch.add(x_sc, x)
        
        x_sc = self.conv5_sc(x_sc)
        x_sc = self.bn5_sc(x_sc)
        x = self.conv5(x)
        x = torch.add(x_sc, x)
        
        # Middle flow
        x = self.middle_flow(x)
        
        # Exit flow
        x_sc = self.conv_exit1_sc(x)
        x_sc = self.bn_exit1_sc(x_sc)
        x = self.conv_exit1(x)
        x = torch.add(x_sc, x)
        
        x = self.conv_exit2(x)

        x = F.avg_pool2d(x, [cfg.INPUT_HEIGHT // 32, cfg.INPUT_WIDTH // 32], padding=0, stride=1)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        
        return x

# main
if __name__ == '__main__':

    model_save_dir = '/'.join(cfg.MODEL_SAVE_PATH.split('/')[:-1])
    os.makedirs(model_save_dir, exist_ok=True)

    main(cfg, Xception())