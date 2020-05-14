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

cfg.MODEL_SAVE_PATH = 'models/ResNeXt50_{}.pt'
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
cfg.TRAIN.DATA_ROTATION = 1

cfg.TEST = EasyDict()
cfg.TEST.MODEL_PATH = cfg.MODEL_SAVE_PATH.format('final')
cfg.TEST.DATA_PATH = '../Dataset/test/images/'
cfg.TEST.MINIBATCH = 2

# random seed
torch.manual_seed(0)

class ResNeXt50(torch.nn.Module):
    def __init__(self):
        super(ResNeXt50, self).__init__()

        class ResNeXtBlock(torch.nn.Module):
            def __init__(self, in_f, f_1, out_f, stride=1, cardinality=32):
                super(ResNeXtBlock, self).__init__()

                self.stride = stride
                self.fit_dim = False
                
                self.block = torch.nn.Sequential(
                    torch.nn.Conv2d(in_f, f_1, kernel_size=1, padding=0, stride=stride),
                    torch.nn.BatchNorm2d(f_1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(f_1, f_1, kernel_size=3, padding=1, stride=1, groups=cardinality),
                    torch.nn.BatchNorm2d(f_1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(f_1, out_f, kernel_size=1, padding=0, stride=1),
                    torch.nn.BatchNorm2d(out_f),
                    torch.nn.ReLU(),
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
        
        
        self.block2_1 = ResNeXtBlock(64, 64, 256)
        self.block2_2 = ResNeXtBlock(256, 64, 256)
        self.block2_3 = ResNeXtBlock(256, 64, 256)

        self.block3_1 = ResNeXtBlock(256, 128, 512, stride=2)
        self.block3_2 = ResNeXtBlock(512, 128, 512)
        self.block3_3 = ResNeXtBlock(512, 128, 512)
        self.block3_4 = ResNeXtBlock(512, 128, 512)

        self.block4_1 = ResNeXtBlock(512, 256, 1024, stride=2)
        self.block4_2 = ResNeXtBlock(1024, 256, 1024)
        self.block4_3 = ResNeXtBlock(1024, 256, 1024)
        self.block4_4 = ResNeXtBlock(1024, 256, 1024)
        self.block4_5 = ResNeXtBlock(1024, 256, 1024)
        self.block4_6 = ResNeXtBlock(1024, 256, 1024)

        self.block5_1 = ResNeXtBlock(1024, 512, 2048, stride=2)
        self.block5_2 = ResNeXtBlock(2048, 512, 2048)
        self.block5_3 = ResNeXtBlock(2048, 512, 2048)
        
        self.linear = torch.nn.Linear(2048, cfg.CLASS_NUM)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, padding=1, stride=2)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)

        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)
        x = self.block4_4(x)
        x = self.block4_5(x)
        x = self.block4_6(x)

        x = self.block5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)

        x = F.avg_pool2d(x, [cfg.INPUT_HEIGHT // 32, cfg.INPUT_WIDTH // 32], padding=0, stride=1)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        
        return x


# main
if __name__ == '__main__':

    model_save_dir = '/'.join(cfg.MODEL_SAVE_PATH.split('/')[:-1])
    os.makedirs(model_save_dir, exist_ok=True)

    main(cfg, ResNeXt50())