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

cfg.MODEL_SAVE_PATH = 'models/VGG16_{}.pt'
cfg.MODEL_SAVE_INTERVAL = 200
cfg.ITERATION = 1000
cfg.MINIBATCH = 8
cfg.OPTIMIZER = torch.optim.SGD
cfg.LEARNING_RATE = 0.1
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


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.conv1 = torch.nn.Sequential(OrderedDict({
            'conv1_1' : torch.nn.Conv2d(cfg.INPUT_CHANNEL, 64, kernel_size=3, padding=1, stride=1),
            'conv1_1_relu' : torch.nn.ReLU(),
            'conv1_1_bn' : torch.nn.BatchNorm2d(64),
            'conv1_2' : torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            'conv1_2_relu' : torch.nn.ReLU(),
            'conv1_2_bn' : torch.nn.BatchNorm2d(64),
        }))

        self.conv2 = torch.nn.Sequential(OrderedDict({
            'conv2_1' : torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            'conv2_1_relu' : torch.nn.ReLU(),
            'conv2_1_bn' : torch.nn.BatchNorm2d(128),
            'conv2_2' : torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            'conv2_2_relu' : torch.nn.ReLU(),
            'conv2_2_bn' : torch.nn.BatchNorm2d(128),
        }))

        self.conv3 = torch.nn.Sequential(OrderedDict({
            'conv3_1' : torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            'conv3_1_relu' : torch.nn.ReLU(),
            'conv3_1_bn' : torch.nn.BatchNorm2d(256),
            'conv3_2' : torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            'conv3_2_relu' : torch.nn.ReLU(),
            'conv3_2_bn' : torch.nn.BatchNorm2d(256),
            'conv3_3' : torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            'conv3_3_relu' : torch.nn.ReLU(),
            'conv3_3_bn' : torch.nn.BatchNorm2d(256),
            'conv3_4' : torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            'conv3_4_relu' : torch.nn.ReLU(),
            'conv3_4_bn' : torch.nn.BatchNorm2d(256),
        }))

        self.conv4 = torch.nn.Sequential(OrderedDict({
            'conv4_1' : torch.nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            'conv4_1_relu' : torch.nn.ReLU(),
            'conv4_1_bn' : torch.nn.BatchNorm2d(512),
            'conv4_2' : torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            'conv4_2_relu' : torch.nn.ReLU(),
            'conv4_2_bn' : torch.nn.BatchNorm2d(512),
            'conv4_3' : torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            'conv4_3_relu' : torch.nn.ReLU(),
            'conv4_3_bn' : torch.nn.BatchNorm2d(512),
            'conv4_4' : torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            'conv4_4_relu' : torch.nn.ReLU(),
            'conv4_4_bn' : torch.nn.BatchNorm2d(512),
        }))

        self.conv5 = torch.nn.Sequential(OrderedDict({
            'conv5_1' : torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            'conv5_1_relu' : torch.nn.ReLU(),
            'conv5_1_bn' : torch.nn.BatchNorm2d(512),
            'conv5_2' : torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            'conv5_2_relu' : torch.nn.ReLU(),
            'conv5_2_bn' : torch.nn.BatchNorm2d(512),
            'conv5_3' : torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            'conv5_3_relu' : torch.nn.ReLU(),
            'conv5_3_bn' : torch.nn.BatchNorm2d(512),
            'conv5_3' : torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            'conv5_3_relu' : torch.nn.ReLU(),
            'conv5_3_bn' : torch.nn.BatchNorm2d(512),
        }))
        
        self.top = torch.nn.Sequential(OrderedDict({
            'Dense1' : torch.nn.Linear(512 * (cfg.INPUT_HEIGHT // 32) * (cfg.INPUT_WIDTH // 32), 256),
            'Dense1_relu' : torch.nn.ReLU(),
            'Dense1_dropout' : torch.nn.Dropout(p=0.5),
            'Dense2' : torch.nn.Linear(256, 256),
            'Dense2_relu' : torch.nn.ReLU(),
            'Dense2_dropout' : torch.nn.Dropout(p=0.5),
        }))

        self.fc_out = torch.nn.Linear(256, cfg.CLASS_NUM)
        

    def forward(self, x):
        # block conv1
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        # block conv2
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        # block conv3
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        # block conv4
        x = self.conv4(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        # block conv5
        x = self.conv5(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        
        x = x.view(x.shape[0], -1)
        x = self.top(x)
        x = self.fc_out(x)
        x = F.softmax(x, dim=1)
        return x

# main
if __name__ == '__main__':

    model_save_dir = '/'.join(cfg.MODEL_SAVE_PATH.split('/')[:-1])
    os.makedirs(model_save_dir, exist_ok=True)

    main(cfg, VGG19())