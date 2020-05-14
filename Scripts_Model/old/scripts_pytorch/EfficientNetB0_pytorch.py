import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import copy
from collections import OrderedDict
from tqdm import tqdm

# class config
class_label = ['akahara', 'madara']
class_N = len(class_label)

# config
img_height, img_width = 96, 96
channel = 3

# GPU
GPU = False
device = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")

# other
model_path = 'EfficientNetB0.pt'

# random seed
torch.manual_seed(0)


class EfficientNetB0(torch.nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()

        width_coefficient=1
        depth_coefficient=1
        dropout_ratio=0.2
        depth_divisor=8
        drop_connect_rate=0.2

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
        _modules['stem_conv'] = torch.nn.Conv2d(channel, round_filters(32), kernel_size=3, padding=1, stride=2, bias=False)
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
        _modules['top_class_linear'] = torch.nn.Linear(round_filters(1280), class_N)
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


# get train data
def data_load(path, hf=False, vf=False, rot=False):
    if (rot == 0) and (rot != False):
        raise Exception('invalid rot >> ', rot, 'should be [1, 359] or False')

    paths = []
    ts = []
    
    data_num = 0
    for dir_path in glob(path + '/*'):
        data_num += len(glob(dir_path + "/*"))
            
    pbar = tqdm(total = data_num)
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            for i, cls in enumerate(class_label):
                if cls in path:
                    t = i

            paths.append({'path': path, 'hf': False, 'vf': False, 'rot': 0})
            ts.append(t)

            # horizontal flip
            if hf:
                paths.append({'path': path, 'hf': True, 'vf': False, 'rot': 0})
                ts.append(t)
            # vertical flip
            if vf:
                paths.append({'path': path, 'hf': False, 'vf': True, 'rot': 0})
                ts.append(t)
            # horizontal and vertical flip
            if hf and vf:
                paths.append({'path': path, 'hf': True, 'vf': True, 'rot': 0})
                ts.append(t)
            # rotation
            if rot is not False:
                angle = rot
                while angle < 360:
                    paths.append({'path': path, 'hf': False, 'vf': False, 'rot': rot})
                    angle += rot
                    ts.append(t)
                
            pbar.update(1)
                    
    pbar.close()
    
    return np.array(paths), np.array(ts)

def get_image(infos):
    xs = []
    
    for info in infos:
        path = info['path']
        hf = info['hf']
        vf = info['vf']
        rot = info['rot']
        x = cv2.imread(path)

        # resize
        x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
        
        # channel BGR -> Gray
        if channel == 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = np.expand_dims(x, axis=-1)

        # channel BGR -> RGB
        if channel == 3:
            x = x[..., ::-1]

        # normalization [0, 255] -> [-1, 1]
        x = x / 127.5 - 1

        # horizontal flip
        if hf:
            x = x[:, ::-1]

        # vertical flip
        if vf:
            x = x[::-1]

        # rotation
        scale = 1
        _h, _w, _c = x.shape
        max_side = max(_h, _w)
        tmp = np.zeros((max_side, max_side, _c))
        tx = int((max_side - _w) / 2)
        ty = int((max_side - _h) / 2)
        tmp[ty: ty+_h, tx: tx+_w] = x.copy()
        M = cv2.getRotationMatrix2D((max_side / 2, max_side / 2), rot, scale)
        _x = cv2.warpAffine(tmp, M, (max_side, max_side))
        _x = _x[tx:tx+_w, ty:ty+_h]

        xs.append(x)
                
    xs = np.array(xs, dtype=np.float32)
    xs = np.transpose(xs, (0,3,1,2))
    
    return xs



# train
def train():
    # model
    model = EfficientNetB0().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    paths, ts = data_load('../Dataset/train/images/', hf=True, vf=True, rot=1)

    # training
    mb = 32
    mbi = 0
    data_N = len(paths)
    train_ind = np.arange(data_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)

    loss_fn = torch.nn.NLLLoss()
    
    for i in range(500):
        if mbi + mb > data_N:
            mb_ind = copy.copy(train_ind)[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (data_N - mbi))]))
        else:
            mb_ind = train_ind[mbi: mbi + mb]
            mbi += mb

        x = torch.tensor(get_image(paths[mb_ind]), dtype=torch.float).to(device)
        t = torch.tensor(ts[mb_ind], dtype=torch.long).to(device)

        opt.zero_grad()
        y = model(x)
        #y = F.log_softmax(y, dim=1)
        loss = loss_fn(torch.log(y), t)
        
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item() / mb

        if (i + 1) % 50 == 0:
            print("iter >>", i+1, ', loss >>', loss.item(), ', accuracy >>', acc)

    torch.save(model.state_dict(), model_path)

# test
def test():
    model = EfficientNetB0().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    paths, ts = data_load('../Dataset/test/images/', hf=False, vf=False, rot=False)

    with torch.no_grad():
        for i in range(len(paths)):
            path = paths[i]
            x = get_image(path)
            t = ts[i]
            
            x = torch.tensor(x, dtype=torch.float).to(device)
            
            pred = model(x)
            pred = pred.detach().cpu().numpy()[0]
        
            print("in {}, predicted probabilities >> {}".format(path, pred))
    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
