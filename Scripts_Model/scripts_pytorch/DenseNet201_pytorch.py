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

cfg.MODEL_SAVE_PATH = 'models/DenseNet201_{}.pt'
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


class DenseNet201(torch.nn.Module):
    def __init__(self):
        super(DenseNet201, self).__init__()

        class Block(torch.nn.Module):
            def __init__(self, first_dim, k=32, L=6):
                super(Block, self).__init__()
                self.L = L
                self.blocks = torch.nn.ModuleList()
                self.blocks.append(torch.nn.Sequential(
                        torch.nn.BatchNorm2d(first_dim),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(first_dim, k, kernel_size=1, padding=0, stride=1),
                        torch.nn.BatchNorm2d(k),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(k, k, kernel_size=3, padding=1, stride=1),
                    ))
                
                for i in range(1, L):
                    self.blocks.append(torch.nn.Sequential(
                        torch.nn.BatchNorm2d(k * i + first_dim),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(k * i + first_dim, k, kernel_size=1, padding=0, stride=1),
                        torch.nn.BatchNorm2d(k),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(k, k, kernel_size=3, padding=1, stride=1),
                    ))
                
            def forward(self, x):
                xs = [None for _ in range(self.L + 1)]
                xs[0] = x
                xs[1] = self.blocks[0](x)
                
                for i in range(1, self.L):
                    x_in = xs[i]
                    for j in range(i):
                        x_in = torch.cat([x_in, xs[j]], dim=1)
                    x = self.blocks[i](x_in)
                    xs[i + 1] = x
                        
                x = xs[0]
                for i in range(1, (self.L + 1)):
                    x = torch.cat([x, xs[i]], dim=1)

                return x

        k = 32
        theta = 0.5
        self.bn1 = torch.nn.BatchNorm2d(cfg.INPUT_CHANNEL)
        self.conv1 = torch.nn.Conv2d(cfg.INPUT_CHANNEL, k * 2, kernel_size=7, padding=3, stride=2)
        
        # Dense block1
        block1_L = 6
        block1_dim = int(k * block1_L * theta)
        
        self.block1 = Block(first_dim = k * 2, L = block1_L)
        
        # Transition layer1
        self.transition1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(k * block1_L + k * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(k * block1_L + k * 2, block1_dim, kernel_size=1, padding=0, stride=1),
            torch.nn.AvgPool2d(2, stride=2, padding=0)
        )
    
        # Dense block2
        block2_L = 12
        block2_dim = int(k * block2_L * theta)
        
        self.block2 = Block(first_dim = block1_dim, L = block2_L)

        # Transition layer2        
        self.transition2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(k * block2_L + block1_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(k * block2_L + block1_dim, block2_dim, kernel_size=1, padding=0, stride=1),
            torch.nn.AvgPool2d(2, stride=2, padding=0)
        )
        
        # Dense block3
        block3_L = 48
        block3_dim = int(k * block3_L * theta)
        
        self.block3 = Block(first_dim = block2_dim, L = block3_L)
        
        # Transition layer3
        self.transition3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(k * block3_L + block2_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(k * block3_L + block2_dim, block3_dim, kernel_size=1, padding=0, stride=1),
            torch.nn.AvgPool2d(2, stride=2, padding=0)
        )
        
        # Dense block4
        block4_L = 32
        self.block4 = Block(first_dim = block3_dim, L = block4_L)
        
        self.linear = torch.nn.Linear(k * block4_L + block3_dim, cfg.CLASS_NUM)
        
        
    def forward(self, x):
        # Entry flow
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        
        x = F.max_pool2d(x, 3, padding=1, stride=2)
        
        x = self.block1(x)
        
        x = self.transition1(x)
        
        x = self.block2(x)
        
        x = self.transition2(x)
        
        x = self.block3(x)
        
        x = self.transition3(x)
        
        x = self.block4(x)

        x = F.avg_pool2d(x, [cfg.INPUT_HEIGHT // 32, cfg.INPUT_WIDTH // 32], padding=0, stride=1)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        
        return x

# main
if __name__ == '__main__':

    model_save_dir = '/'.join(cfg.MODEL_SAVE_PATH.split('/')[:-1])
    os.makedirs(model_save_dir, exist_ok=True)

    main(cfg, DenseNet201())