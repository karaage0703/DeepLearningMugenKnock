from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from copy import copy

# class config
class_label = {'akahara': [0,0,128],
       'madara': [0,128,0]}

class_N = len(class_label)

# config
img_height, img_width = 64, 64 #572, 572
out_height, out_width = 64, 64 #388, 388
channel = 3

# GPU
GPU = False
device = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")

# other
model_path = 'SegNet.pt'

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

        self.out = torch.nn.Conv2d(64, class_N, kernel_size=1, padding=0, stride=1)
        
        
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
        x = F.softmax(x, dim=1)
        
        return x

    
# get train data
def data_load(path, hf=False, vf=False):
    xs = []
    ts = []
    paths = []
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            x = x[..., ::-1]
            xs.append(x)

            gt_path = path.replace("images", "seg_images").replace(".jpg", ".png")
            gt = cv2.imread(gt_path)
            gt = cv2.resize(gt, (out_width, out_height), interpolation=cv2.INTER_NEAREST)

            t = np.zeros((out_height, out_width), dtype=np.int)

            for i, (_, vs) in enumerate(class_label.items()):
                ind = (gt[...,0] == vs[0]) * (gt[...,1] == vs[1]) * (gt[...,2] == vs[2])
                t[ind] = i + 1
            #print(gt_path)
            #import matplotlib.pyplot as plt
            #plt.imshow(t)
            #plt.show()

            ts.append(t)
            
            paths.append(path)

            if hf:
                xs.append(x[:, ::-1])
                ts.append(t[:, ::-1])
                paths.append(path)

            if vf:
                xs.append(x[::-1])
                ts.append(t[::-1])
                paths.append(path)

            if hf and vf:
                xs.append(x[::-1, ::-1])
                ts.append(t[::-1, ::-1])
                paths.append(path)

    xs = np.array(xs)
    ts = np.array(ts)

    xs = xs.transpose(0,3,1,2)
    
    return xs, ts, paths


# train
def train():
    # model
    model = SegNet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True)

    # training
    mb = 16
    mbi = 0
    train_N = len(xs)
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)

    loss_fn = torch.nn.NLLLoss()
    
    for i in range(1000):
        if mbi + mb > train_N:
            mb_ind = copy(train_ind[mbi:])
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (train_N - mbi))]))
            mbi = mb - (train_N - mbi)
        else:
            mb_ind = train_ind[mbi : mbi + mb]
            mbi += mb

        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(ts[mb_ind], dtype=torch.long).to(device)

        opt.zero_grad()
        y = model(x)

        y = y.permute(0,2,3,1).contiguous()
        y = y.view(-1, class_N)
        t = t.view(-1)
        
        loss = loss_fn(torch.log(y), t)
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item() / mb / img_height / img_width
        
        print("iter :", i+1, ', loss :', loss.item(), ', accuracy :', acc)

    torch.save(model.state_dict(), model_path)

# test
def test():
    model = SegNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    xs, ts, paths = data_load('../Dataset/test/images/')

    with torch.no_grad():
        for i in range(len(paths)):
            x = xs[i]
            t = ts[i]
            path = paths[i]

            x = np.expand_dims(x, axis=0)
            x = torch.tensor(x, dtype=torch.float).to(device)

            pred = model(x)

            #pred = pred.permute(0,2,3,1).reshape(-1, class_N+1)
            pred = pred.detach().cpu().numpy()[0]
            pred = pred.argmax(axis=0)

            # visualize
            out = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            for i, (_, vs) in enumerate(class_label.items()):
                out[pred == (i+1)] = vs

            print("in {}".format(path))

            plt.subplot(1,2,1)
            plt.imshow(x.detach().cpu().numpy()[0].transpose(1,2,0))
            plt.subplot(1,2,2)
            plt.imshow(out[..., ::-1])
            plt.show()
    

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
