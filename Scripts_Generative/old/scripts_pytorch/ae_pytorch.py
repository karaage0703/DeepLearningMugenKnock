import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# class config
class_label = {'akahara': [0,0,128],
       'madara': [0,128,0]}

class_N = len(class_label)

# config
img_height, img_width = 64, 64 #572, 572
channel = 3

# GPU
GPU = False
device = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")

# other
model_path = 'AE.pt'

# random seed
torch.manual_seed(0)

    
class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.enc = torch.nn.Linear(img_height * img_width * channel, 64)
        self.dec = torch.nn.Linear(64, img_height * img_width * channel)
        
    def forward(self, x):
        mb, c, h, w = x.size()
        x = x.view(mb, -1)
        x = self.enc(x)
        x = self.dec(x)
        return x

    
CLS = {'akahara': [0,0,128],
       'madara': [0,128,0]}
    
# get train data
def data_load(path, hf=False, vf=False, rot=False):
    xs = []
    paths = []
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            if channel == 1:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x = x / 127.5 - 1
            if channel == 3:
                x = x[..., ::-1]
            xs.append(x)

            paths.append(path)

            if hf:
                xs.append(x[:, ::-1])
                paths.append(path)

            if vf:
                xs.append(x[::-1])
                paths.append(path)

            if hf and vf:
                xs.append(x[::-1, ::-1])
                paths.append(path)

            if rot != False:
                angle = 0
                scale = 1
                while angle < 360:
                    angle += rot
                    if channel == 1:
                        _h, _w = x.shape
                        max_side = max(_h, _w)
                        tmp = np.zeros((max_side, max_side))
                    else:
                        _h, _w, _c = x.shape
                        max_side = max(_h, _w)
                        tmp = np.zeros((max_side, max_side, _c))
                    
                    tx = int((max_side - _w) / 2)
                    ty = int((max_side - _h) / 2)
                    tmp[ty: ty+_h, tx: tx+_w] = x.copy()
                    M = cv2.getRotationMatrix2D((max_side/2, max_side/2), angle, scale)
                    _x = cv2.warpAffine(tmp, M, (max_side, max_side))
                    _x = _x[tx:tx+_w, ty:ty+_h]
                    xs.append(_x)
                    paths.append(path)
                    
    xs = np.array(xs, dtype=np.float32)

    if channel == 1:
        xs = np.expand_dims(xs, axis=-1)
    xs = np.transpose(xs, [0,3,1,2])
    
    return xs, paths


# train
def train():
    # model
    model = AE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    xs, paths = data_load('../Dataset/train/images/', hf=True, vf=True, rot=1)

    # training
    mb = 256
    mbi = 0
    train_N = len(xs)
    train_ind = np.arange(train_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(1000):
        if mbi + mb > train_N:
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (train_N - mbi))]))
            mbi = mb - (train_N - mbi)
        else:
            mb_ind = train_ind[mbi : mbi + mb]
            mbi += mb

        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)

        opt.zero_grad()

        y = model(x)
        
        loss = torch.nn.MSELoss()(y, t.view(mb, -1))
        loss.backward()
        opt.step()
    
        #pred = y.argmax(dim=1, keepdim=True)
        acc = y.eq(t.view_as(y)).sum().item() / mb
        
        print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', acc)

    torch.save(model.state_dict(), model_path)

# test
def test():
    model = Mynet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    xs, paths = data_load('../Dataset/test/images/')

    with torch.no_grad():
        for i in range(len(paths)):
            x = xs[i]
            path = paths[i]
            
            x = np.expand_dims(x, axis=0)
            x = torch.tensor(x, dtype=torch.float).to(device)
            
            pred = model(x)

            pred = pred.view(channel, img_height, img_width)
            pred = pred.detach().cpu().numpy()
            pred -= pred.min()
            pred /= pred.max()
            pred = pred.transpose(1,2,0)
            
            _x = x.detach().cpu().numpy()[0]
            _x = (_x + 1) / 2
            if channel == 1:
                pred = pred[..., 0]
                _x = _x[0]
                cmap = 'gray'
            else:
                _x = _x.transpose(1,2,0)
                cmap = None

            print("in {}".format(path))
                
            plt.subplot(1,2,1)
            plt.title("input")
            plt.imshow(_x, cmap=cmap)
            plt.subplot(1,2,2)
            plt.title("predicted")
            plt.imshow(pred, cmap=cmap)
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
