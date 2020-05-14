import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from copy import copy
from tqdm import tqdm

# class config
class_label = {'akahara' : [0, 0, 128], 'madara' : [0, 128, 0]}
class_N = len(class_label)

# config
img_height, img_width = 64, 64 #572, 572
out_height, out_width = 64, 64 #388, 388
channel = 3

# GPU
GPU = True
device = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")

# other
model_path = 'UNet.pt'

# random seed
torch.manual_seed(0)


class SegNet(torch.nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        
        # VGG block
        class VGG_block(torch.nn.Module):
            def __init__(self, dim1, dim2, layer_N):
                super(VGG_block, self).__init__()

                _module = []

                for i in range(layer_N):
                    dim = dim1 if i == 0 else dim2
                    _module.append(torch.nn.Conv2d(dim, dim2, kernel_size=3, padding=1, stride=1))
                    _module.append(torch.nn.BatchNorm2d(dim2))
                    _module.append(torch.nn.ReLU())

                self.module = torch.nn.Sequential(*_module)

            def forward(self, x):
                x = self.module(x)
                return x
            
        # VGG Decoder block
        class VGG_block_decoder(torch.nn.Module):
            def __init__(self, dim1, dim2, layer_N):
                super(VGG_block_decoder, self).__init__()

                _module = []

                for i in range(layer_N):
                    dim = dim1 if i < (layer_N-1) else dim2
                    _module.append(torch.nn.Conv2d(dim1, dim, kernel_size=3, padding=1, stride=1))
                    _module.append(torch.nn.BatchNorm2d(dim2))
                    _module.append(torch.nn.ReLU())

                self.module = torch.nn.Sequential(*_module)

            def forward(self, x):
                x = self.module(x)
                return x

        
        self.enc1 = VGG_block(3, 64, 2)
        self.enc2 = VGG_block(64, 128, 2)
        self.enc3 = VGG_block(128, 256, 3)
        self.enc4 = VGG_block(256, 512, 3)
        self.enc5 = VGG_block(512, 512, 3)

        self.dec5 = VGG_block(512, 512, 3)
        self.dec4 = VGG_block(512, 256, 3)
        self.dec3 = VGG_block(256, 128, 3)
        self.dec2 = VGG_block(128, 64, 2)
        self.dec1 = VGG_block(64, 64, 2)

        self.out = torch.nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1)
        
        
    def forward(self, x):
        # Encoder block 1
        x_enc1 = self.enc1(x)
        x, pool1_ind = F.max_pool2d(x_enc1, 2, stride=2, padding=0, return_indices=True)
        
        # Encoder block 2
        x_enc2 = self.enc2(x)
        x, pool2_ind = F.max_pool2d(x_enc2, 2, stride=2, padding=0, return_indices=True)
        
        # Encoder block 3
        x_enc3 = self.enc3(x)
        x, pool3_ind = F.max_pool2d(x_enc3, 2, stride=2, padding=0, return_indices=True)
        
        # Encoder block 4
        x_enc4 = self.enc4(x)
        x, pool4_ind = F.max_pool2d(x_enc4, 2, stride=2, padding=0, return_indices=True)
        
        # Encoder block 5
        x_enc5 = self.enc5(x)
        x, pool5_ind = F.max_pool2d(x_enc5, 2, stride=2, padding=0, return_indices=True)

        # Decoder block 5
        x = F.max_unpool2d(x, pool5_ind, 2, stride=2, padding=0)
        x = self.dec5(x)
        
        # Decoder block 4
        x = F.max_unpool2d(x, pool4_ind, 2, stride=2, padding=0)
        x = self.dec4(x)
        
        # Decoder block 3
        x = F.max_unpool2d(x, pool3_ind, 2, stride=2, padding=0)
        x = self.dec3(x)
        
        # Decoder block 2
        x = F.max_unpool2d(x, pool2_ind, 2, stride=2, padding=0)
        x = self.dec2(x)
        
        # Decoder block 1
        x = F.max_unpool2d(x, pool1_ind, 2, stride=2, padding=0)
        x = self.dec1(x)

        # output
        x = self.out(x)
        x = torch.sigmoid(x)
        
        return x

# get train data
def data_load(path, hf=False, vf=False, rot=False):
    if (rot == 0) and (rot != False):
        raise Exception('invalid rot >> ', rot, 'should be [1, 359] or False')

    paths = []
    paths_gt = []
    
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
            
            gt_path = path.replace("images", "seg_images").replace(".jpg", ".png")
            paths_gt.append({'path': gt_path, 'hf': False, 'vf': False, 'rot': 0})

            # horizontal flip
            if hf:
                paths.append({'path': path, 'hf': True, 'vf': False, 'rot': 0})
                paths_gt.append({'path': gt_path, 'hf': True, 'vf': False, 'rot': 0})
            # vertical flip
            if vf:
                paths.append({'path': path, 'hf': False, 'vf': True, 'rot': 0})
                paths_gt.append({'path': gt_path, 'hf': False, 'vf': True, 'rot': 0})
            # horizontal and vertical flip
            if hf and vf:
                paths.append({'path': path, 'hf': True, 'vf': True, 'rot': 0})
                paths_gt.append({'path': gt_path, 'hf': True, 'vf': True, 'rot': 0})
            # rotation
            if rot is not False:
                angle = rot
                while angle < 360:
                    paths.append({'path': path, 'hf': False, 'vf': False, 'rot': rot})
                    paths_gt.append({'path': gt_path, 'hf': False, 'vf': False, 'rot': rot})
                    angle += rot
                
            pbar.update(1)
                    
    pbar.close()
    
    return np.array(paths), np.array(paths_gt)

def get_image(infos, gt=False):
    xs = []
    
    for info in infos:
        path = info['path']
        hf = info['hf']
        vf = info['vf']
        rot = info['rot']
        x = cv2.imread(path)

        # resize
        if gt:
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
        else:
            x = cv2.resize(x, (out_width, out_height)).astype(np.float32)
        
        # channel BGR -> Gray
        if channel == 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = np.expand_dims(x, axis=-1)

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
        x = _x[tx:tx+_w, ty:ty+_h]

        if gt:
            _x = x
            x = np.zeros((out_height, out_width), dtype=np.int)

            for i, (_, vs) in enumerate(class_label.items()):
                ind = (_x[..., 0] == vs[0]) * (_x[..., 1] == vs[1]) * (_x[..., 2] == vs[2])
                x[ind] = 1
        else:
            # normalization [0, 255] -> [-1, 1]
            x = x / 127.5 - 1

            # channel BGR -> RGB
            if channel == 3:
                x = x[..., ::-1]

        xs.append(x)
                
    xs = np.array(xs, dtype=np.float32)

    if not gt:
        xs = np.transpose(xs, (0,3,1,2))
    
    return xs


# train
def train():
    # model
    model = SegNet().to(device)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    paths, paths_gt = data_load('../Dataset/train/images/', hf=True, vf=True, rot=False)

    # training
    mb = 16
    mbi = 0
    train_N = len(paths)
    train_ind = np.arange(train_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)

    loss_fn = torch.nn.BCELoss()
    
    for i in range(1000):
        if mbi + mb > train_N:
            mb_ind = copy(train_ind[mbi:])
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[ : (mb - (train_N - mbi))]))
            mbi = mb - (train_N - mbi)
        else:
            mb_ind = train_ind[mbi : mbi + mb]
            mbi += mb

        # data load
        x = torch.tensor(get_image(paths[mb_ind]), dtype=torch.float).to(device)
        t = torch.tensor(np.expand_dims(get_image(paths_gt[mb_ind], gt=True), axis=1), dtype=torch.float).to(device)

        opt.zero_grad()
        y = model(x)
        
        loss = loss_fn(y, t)
        loss.backward()
        opt.step()
    
        MAE = np.abs((y - t).detach().cpu().numpy()).mean()
        
        if (i + 1) % 50 == 0:
            print('Iter : {} , Loss : {} , MAE : {}'.format(i + 1, loss.item(), MAE))

    torch.save(model.state_dict(), model_path)

# test
def test():
    model = SegNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    paths, path_gt = data_load('../Dataset/test/images/')

    with torch.no_grad():
        for i in range(len(paths)):
            path = paths[[i]]
            x = get_image(path)

            x = torch.tensor(get_image(paths[[i]]), dtype=torch.float).to(device)

            pred = model(x)

            #pred = pred.permute(0,2,3,1).reshape(-1, class_num+1)
            pred = pred.detach().cpu().numpy()[0, 0]


            print(">> {}".format(path[0]['path']))

            plt.subplot(1, 2, 1)
            plt.imshow((x.detach().cpu().numpy()[0].transpose(1,2,0) * 127.5 + 127.5).astype(np.uint8))
            plt.subplot(1, 2, 2)
            plt.imshow(pred, cmap='gray')
            plt.show()
    

def arg_parse():
    parser = argparse.ArgumentParser(description='')
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