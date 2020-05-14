import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from easydict import EasyDict
from _main_base import main

#---
# config
#---
cfg = EasyDict()

# class
cfg.CLASS_LABEL = ['akahara', 'madara']
cfg.CLASS_NUM = len(cfg.CLASS_LABEL)

# model
cfg.INPUT_HEIGHT = 32
cfg.INPUT_WIDTH = 32
cfg.INPUT_CHANNEL = 3

cfg.GPU = False
cfg.DEVICE = torch.device("cuda" if cfg.GPU and torch.cuda.is_available() else "cpu")

cfg.MODEL_SAVE_PATH = 'EfficientNetB6_{}.pt'
cfg.MODEL_SAVE_INTERVAL = 200
cfg.ITERATION = 1000
cfg.MINIBATCH = 2
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


class EfficientNet(torch.nn.Module):
    def __init__(self, block_type='B0'):
        super(EfficientNet, self).__init__()

        print('--')
        print('You call EfficientNet')
        print(' - Input dim (h, w, c) >> ({}, {}, {})'.format(cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, cfg.INPUT_CHANNEL))
        print('--')

        EFFICIENTNET_CONFIG = {
            'B0' : {'WIDTH_COEFFICIENT' : 1, 'DEPTH_COEFFICIENT' : 1, 'DROPOUT_RATIO' : 0.2, 'DEPTH_DIVISOR' : 8, 'DROP_CONNECT_RATIO' : 0.2},
            'B1' : {'WIDTH_COEFFICIENT' : 1, 'DEPTH_COEFFICIENT' : 1.1, 'DROPOUT_RATIO' : 0.2, 'DEPTH_DIVISOR' : 8, 'DROP_CONNECT_RATIO' : 0.2},
            'B2' : {'WIDTH_COEFFICIENT' : 1.1, 'DEPTH_COEFFICIENT' : 1.2, 'DROPOUT_RATIO' : 0.3, 'DEPTH_DIVISOR' : 8, 'DROP_CONNECT_RATIO' : 0.2},
            'B3' : {'WIDTH_COEFFICIENT' : 1.2, 'DEPTH_COEFFICIENT' : 1.4, 'DROPOUT_RATIO' : 0.3, 'DEPTH_DIVISOR' : 8, 'DROP_CONNECT_RATIO' : 0.2},
            'B4' : {'WIDTH_COEFFICIENT' : 1.2, 'DEPTH_COEFFICIENT' : 1.4, 'DROPOUT_RATIO' : 0.3, 'DEPTH_DIVISOR' : 8, 'DROP_CONNECT_RATIO' : 0.2},
            'B5' : {'WIDTH_COEFFICIENT' : 1.6, 'DEPTH_COEFFICIENT' : 2.2, 'DROPOUT_RATIO' : 0.4, 'DEPTH_DIVISOR' : 8, 'DROP_CONNECT_RATIO' : 0.2},
            'B6' : {'WIDTH_COEFFICIENT' : 1.8, 'DEPTH_COEFFICIENT' : 2.6, 'DROPOUT_RATIO' : 0.5, 'DEPTH_DIVISOR' : 8, 'DROP_CONNECT_RATIO' : 0.2},
            'B7' : {'WIDTH_COEFFICIENT' : 2.0, 'DEPTH_COEFFICIENT' : 3.1, 'DROPOUT_RATIO' : 0.5, 'DEPTH_DIVISOR' : 8, 'DROP_CONNECT_RATIO' : 0.2}
        }

        width_coefficient = EFFICIENTNET_CONFIG[block_type]['WIDTH_COEFFICIENT']
        depth_coefficient = EFFICIENTNET_CONFIG[block_type]['DEPTH_COEFFICIENT']
        dropout_ratio = EFFICIENTNET_CONFIG[block_type]['DROPOUT_RATIO']
        depth_divisor = EFFICIENTNET_CONFIG[block_type]['DEPTH_DIVISOR']
        drop_connect_rate = EFFICIENTNET_CONFIG[block_type]['DROP_CONNECT_RATIO']

        DEFAULT_BLOCKS_ARGS = [
            # block 1
            {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
            'expand_ratio': 1, 'id_skip': True, 'stride': 1, 'se_ratio': 0.25},
            # block 2
            {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
            'expand_ratio': 6, 'id_skip': True, 'stride': 2, 'se_ratio': 0.25},
            # block 3
            {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
            'expand_ratio': 6, 'id_skip': True, 'stride': 2, 'se_ratio': 0.25},
            # block 4
            {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
            'expand_ratio': 6, 'id_skip': True, 'stride': 2, 'se_ratio': 0.25},
            # block 5
            {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
            'expand_ratio': 6, 'id_skip': True, 'stride': 1, 'se_ratio': 0.25},
            # block 6
            {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
            'expand_ratio': 6, 'id_skip': True, 'stride': 2, 'se_ratio': 0.25},
            # block 7
            {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
            'expand_ratio': 6, 'id_skip': True, 'stride': 1, 'se_ratio': 0.25}
        ]

        def round_filters(filters, divisor=depth_divisor):
            """Round number of filters based on depth multiplier."""
            filters *= width_coefficient
            new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(np.ceil(depth_coefficient * repeats))

            
        class Reshape(torch.nn.Module):
            def __init__(self, c, h, w):
                super(Reshape, self).__init__()
                self.c = c
                self.h = h
                self.w = w
            
            def forward(self, x):
                x = x.view(x.size()[0], self.c, self.h, self.w)
                return x

        class Flatten(torch.nn.Module):
            def __init__(self):
                super(Flatten, self).__init__()

            def forward(self, x):
                x = x.view(x.size()[0], -1)
                return x

        class Swish(torch.nn.Module):
            def __init__(self):
                super(Swish, self).__init__()

            def forward(self, x):
                return x * torch.sigmoid(x)
                    

        class Block(torch.nn.Module):
            def __init__(self, activation_fn=Swish(), drop_rate=0., name='',
                filters_in=32, filters_out=16, kernel_size=3, stride=1,
                expand_ratio=1, se_ratio=0., id_skip=True):
                super(Block, self).__init__()

                # Expansion phase
                filters = filters_in * expand_ratio

                if expand_ratio != 1:
                    _modules = OrderedDict()
                    _modules[name + 'expand_conv'] = torch.nn.Conv2d(filters_in, filters, kernel_size=1, padding=0, bias=False)
                    _modules[name + 'expand_bn'] = torch.nn.BatchNorm2d(filters)
                    _modules[name + 'expand_activation'] = activation_fn
                    self.expansion = torch.nn.Sequential(_modules)

                # Depthwise Convolution
                _modules = OrderedDict()

                conv_pad = kernel_size // 2
                
                _modules[name + 'dw_conv'] = torch.nn.Conv2d(filters, filters, kernel_size, stride=stride, padding=conv_pad, bias=False, groups=1)
                _modules[name + 'dw_bn'] = torch.nn.BatchNorm2d(filters)
                _modules[name + 'dw_activation'] = activation_fn
                self.DW_conv = torch.nn.Sequential(_modules)


                # Squeeze and Excitation phase
                if 0 < se_ratio <= 1:
                    filters_se = max(1, int(filters_in * se_ratio))

                    _modules = OrderedDict()
                    _modules[name + 'se_sqeeze'] = torch.nn.AdaptiveMaxPool2d((1, 1))
                    _modules[name + 'se_reshape'] = Reshape(c=filters, h=1, w=1)
                    _modules[name + 'se_reduce_conv'] = torch.nn.Conv2d(filters, filters_se, kernel_size=1, padding=0)
                    _modules[name + 'se_reduce_activation'] = activation_fn
                    _modules[name + 'se_expand_conv'] = torch.nn.Conv2d(filters_se, filters, kernel_size=1, padding=0)
                    _modules[name + 'se_expand_activation'] = torch.nn.Sigmoid()
                    self.SE_phase = torch.nn.Sequential(_modules)
                    

                # Output phase
                _modules = OrderedDict()
                _modules[name + 'project_conv'] = torch.nn.Conv2d(filters, filters_out, kernel_size=1, padding=0, bias=False)
                _modules[name + 'project_bn'] = torch.nn.BatchNorm2d(filters_out)
                self.output_phase = torch.nn.Sequential(_modules)


                # 
                self.last_add = False
                if (id_skip is True and stride == 1 and filters_in == filters_out):
                    if drop_rate > 0:
                        self.output_phase_Dropout = torch.nn.Dropout2d(p=drop_rate)

                    self.last_add = True

                
            def forward(self, input_x):
                # expansion phase
                if hasattr(self, 'expansion'):
                    x = self.expansion(input_x)
                else:
                    x = input_x

                x = self.DW_conv(x)

                # Squeeze and Excitation phase
                if hasattr(self, 'SE_phase'):
                    x_SE_phase = self.SE_phase(x)
                    x = x * x_SE_phase

                # Output phase
                x = self.output_phase(x)

                if hasattr(self, 'output_phase_Dropout'):
                    x = self.output_phase_Dropout(x)

                if self.last_add:
                    x = x + input_x

                return x

        # stem
        _modules = OrderedDict()
        _modules['stem_conv'] = torch.nn.Conv2d(cfg.INPUT_CHANNEL, round_filters(32), kernel_size=3, padding=1, stride=2, bias=False)
        _modules['stem_bn'] = torch.nn.BatchNorm2d(round_filters(32))
        _modules['stem_activation'] = Swish()
        self.stem = torch.nn.Sequential(_modules)
        
        # block
        _modules = []

        b = 0
        block_Num = float(sum(args['repeats'] for args in DEFAULT_BLOCKS_ARGS))

        for (i, args) in enumerate(DEFAULT_BLOCKS_ARGS):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = round_filters(args['filters_in'])
            args['filters_out'] = round_filters(args['filters_out'])

            for j in range(round_repeats(args.pop('repeats'))):
                # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args['stride'] = 1
                    args['filters_in'] = args['filters_out']

                _modules.append(
                    Block(activation_fn=Swish(), drop_rate=drop_connect_rate * b / block_Num, name='block{}{}_'.format(i + 1, chr(j + 97)), **args))
                b += 1

        self.block = torch.nn.Sequential(*_modules)


        # top
        _modules = OrderedDict()
        _modules['top_conv'] = torch.nn.Conv2d(DEFAULT_BLOCKS_ARGS[-1]['filters_out'], round_filters(1280), kernel_size=1, padding=0, bias=False)
        _modules['top_bn'] = torch.nn.BatchNorm2d(round_filters(1280))
        _modules['top_activation'] = Swish()
        self.top = torch.nn.Sequential(_modules)

        _modules = OrderedDict()
        _modules['top_class_GAP'] = torch.nn.AdaptiveMaxPool2d((1, 1))
        if dropout_ratio > 0:
            _modules['top_class_dropout'] = torch.nn.Dropout2d(p=dropout_ratio)
        _modules['top_class_flatten'] = Flatten()
        _modules['top_class_linear'] = torch.nn.Linear(round_filters(1280), cfg.CLASS_NUM)
        self.top_class = torch.nn.Sequential(_modules)
        
        
    def forward(self, x):
        # stem
        x = self.stem(x)

        # blocks
        x = self.block(x)

        # top
        x = self.top(x)
        x = self.top_class(x)

        x = F.softmax(x, dim=1)        
        return x


# main
if __name__ == '__main__':
    main(cfg, EfficientNet('B6'))