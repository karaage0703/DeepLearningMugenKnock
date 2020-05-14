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

cfg.MODEL_SAVE_PATH = 'models/ResNet18_{}.pt'
cfg.MODEL_SAVE_INTERVAL = 200
cfg.ITERATION = 1000
cfg.MINIBATCH = 8
cfg.OPTIMIZER = torch.optim.SGD
cfg.LEARNING_RATE = 0.001
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


class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        class ResBlock(torch.nn.Module):
            def __init__(self, in_f, out_f, stride=1):
                super(ResBlock, self).__init__()

                self.stride = stride
                self.fit_dim = False

                self.block = torch.nn.Sequential(
                    torch.nn.Conv2d(in_f, out_f, kernel_size=3, padding=1, stride=stride),
                    torch.nn.BatchNorm2d(out_f),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(out_f, out_f, kernel_size=3, padding=1, stride=1),
                    torch.nn.BatchNorm2d(out_f),
                    torch.nn.ReLU()
                )

                if in_f != out_f:
                    self.fit_conv = torch.nn.Conv2d(in_f, out_f, kernel_size=1, padding=0, stride=1)
                    self.fit_dim = True
                    
            def forward(self, x):
                res_x = self.block(x)
                
                if self.fit_dim:
                    x = self.fit_conv(x)
                
                if self.stride == 2:
                    x = F.max_pool2d(x, 2, stride=2)
                    
                x = torch.add(res_x, x)
                x = F.relu(x)
                return x


        self.conv1 = torch.nn.Conv2d(cfg.INPUT_CHANNEL, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(64)
        
        self.resblock2_1 = ResBlock(64, 64)
        self.resblock2_2 = ResBlock(64, 64)

        self.resblock3_1 = ResBlock(64, 128, stride=2)
        self.resblock3_2 = ResBlock(128, 128)

        self.resblock4_1 = ResBlock(128, 256, stride=2)
        self.resblock4_2 = ResBlock(256, 256)

        self.resblock5_1 = ResBlock(256, 512, stride=2)
        self.resblock5_2 = ResBlock(512, 512)
        
        self.linear = torch.nn.Linear(512, cfg.CLASS_NUM)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, padding=1, stride=2)

        x = self.resblock2_1(x)
        x = self.resblock2_2(x)

        x = self.resblock3_1(x)
        x = self.resblock3_2(x)

        x = self.resblock4_1(x)
        x = self.resblock4_2(x)

        x = self.resblock5_1(x)
        x = self.resblock5_2(x)

        x = F.avg_pool2d(x, [cfg.INPUT_HEIGHT // 32, cfg.INPUT_WIDTH // 32], padding=0, stride=1)
        x = x.view(list(x.size())[0], -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        
        return x

# main
if __name__ == '__main__':

    model_save_dir = '/'.join(cfg.MODEL_SAVE_PATH.split('/')[:-1])
    os.makedirs(model_save_dir, exist_ok=True)

    main(cfg, ResNet18())