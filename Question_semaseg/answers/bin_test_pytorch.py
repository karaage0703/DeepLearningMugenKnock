import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

num_classes = 2
img_height, img_width = 64, 64#572, 572
out_height, out_width = 64, 64#388, 388
GPU = False
torch.manual_seed(0)
    
class Mynet(torch.nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()

        enc1 = []
        for i in range(6):
            f = 3 if i == 0 else 32
            enc1.append(torch.nn.Conv2d(f, 32, kernel_size=3, padding=1, stride=1))
            enc1.append(torch.nn.BatchNorm2d(32))
            enc1.append(torch.nn.ReLU())
        self.enc1 = torch.nn.Sequential(*enc1)

        self.out = torch.nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1)
        
    def forward(self, x):
        # block conv1
        x = self.enc1(x)
        x = self.out(x)
        return x

    
CLS = {'akahara': [0,0,128],
       'madara': [0,128,0]}
    
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

            t = np.zeros((out_height, out_width, 1), dtype=np.int)

            ind = (gt[...,0] > 0) + (gt[..., 1] > 0) + (gt[...,2] > 0)
            t[ind] = 1

            #print(gt_path)
            #import matplotlib.pyplot as plt
            #plt.imshow(t, cmap='gray')
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
    # GPU
    device = torch.device("cuda" if GPU else "cpu")

    # model
    model = Mynet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True)

    # training
    mb = 4
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(500):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            mbi = mb - (len(xs) - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(ts[mb_ind], dtype=torch.float).to(device)

        opt.zero_grad()
        y = model(x)

        y = y.permute(0,2,3,1).contiguous()
        
        y = torch.sigmoid(y)
        loss = torch.nn.BCELoss()(y, t)
        loss.backward()
        opt.step()
    
        #pred = y.argmax(dim=1, keepdim=True)
        acc = y.eq(t.view_as(y)).sum().item() / mb
        
        print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', acc)

    torch.save(model.state_dict(), 'cnn.pt')

# test
def test():
    device = torch.device("cuda" if GPU else "cpu")
    model = Mynet().to(device)
    model.eval()
    model.load_state_dict(torch.load('cnn.pt'))

    xs, ts, paths = data_load('../Dataset/test/images/')

    for i in range(len(paths)):
        x = xs[i]
        t = ts[i]
        path = paths[i]
        
        x = np.expand_dims(x, axis=0)
        x = torch.tensor(x, dtype=torch.float).to(device)
        
        pred = model(x)
    
        pred = torch.sigmoid(pred)
        pred = pred.detach().cpu().numpy()[0, 0]

        ## binalization
        bin_pred = pred.copy()
        th = 0.5
        bin_pred[bin_pred >= th] = 1
        bin_pred[bin_pred < th] = 0
   
        plt.subplot(1,3,1)
        plt.imshow(x.detach().cpu().numpy()[0].transpose(1,2,0))
        plt.title("input")
        plt.subplot(1,3,2)
        plt.imshow(pred, cmap='gray')
        plt.title("predicted")
        plt.subplot(1,3,3)
        plt.imshow(bin_pred, cmap='gray')
        plt.title("after binalization")
        plt.show()

        print("in {}".format(path))
    

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
