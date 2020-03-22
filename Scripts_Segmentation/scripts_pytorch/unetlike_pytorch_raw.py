import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

CLS = {'akahara': [0,0,128],
       'madara': [0,128,0]}
       
class_num = len(CLS)
img_height, img_width = 64, 64 #572, 572
out_height, out_width = 64, 64 #388, 388
GPU = False
torch.manual_seed(0)

    
class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        base = 16
        
        self.enc1 = torch.nn.Sequential()
        for i in range(2):
            f = 3 if i == 0 else base
            self.enc1.add_module("enc1_{}".format(i+1), torch.nn.Conv2d(f, base, kernel_size=3, padding=1, stride=1))
            self.enc1.add_module("enc1_relu_{}".format(i+1), torch.nn.ReLU())
            self.enc1.add_module("enc1_bn_{}".format(i+1), torch.nn.BatchNorm2d(base))

        self.enc2 = torch.nn.Sequential()
        for i in range(2):
            f = base if i == 0 else base * 2
            self.enc2.add_module("enc2_{}".format(i+1), torch.nn.Conv2d(f, base*2, kernel_size=3, padding=1, stride=1))
            self.enc2.add_module("enc2_relu_{}".format(i+1), torch.nn.ReLU())
            self.enc2.add_module("enc2_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*2))

        self.enc3 = torch.nn.Sequential()
        for i in range(2):
            f = base*2 if i == 0 else base*4
            self.enc3.add_module("enc3_{}".format(i+1), torch.nn.Conv2d(f, base*4, kernel_size=3, padding=1, stride=1))
            self.enc3.add_module("enc3_relu_{}".format(i+1), torch.nn.ReLU())
            self.enc3.add_module("enc3_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*4))

        self.enc4 = torch.nn.Sequential()
        for i in range(2):
            f = base*4 if i == 0 else base*8
            self.enc4.add_module("enc4_{}".format(i+1), torch.nn.Conv2d(f, base*8, kernel_size=3, padding=1, stride=1))
            self.enc4.add_module("enc4_relu_{}".format(i+1), torch.nn.ReLU())
            self.enc4.add_module("enc4_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*8))

        self.enc5 = torch.nn.Sequential()
        for i in range(2):
            f = base*8 if i == 0 else base*16
            self.enc5.add_module("enc5_{}".format(i+1), torch.nn.Conv2d(f, base*16, kernel_size=3, padding=1, stride=1))
            self.enc5.add_module("enc5_relu_{}".format(i+1), torch.nn.ReLU())
            self.enc5.add_module("enc5_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*16))

        self.tconv4 = torch.nn.ConvTranspose2d(base*16, base*8, kernel_size=2, stride=2)
        self.tconv4_bn = torch.nn.BatchNorm2d(base*8)

        self.dec4 = torch.nn.Sequential()
        for i in range(2):
            f = base*16 if i == 0 else base*8
            self.dec4.add_module("dec4_{}".format(i+1), torch.nn.Conv2d(f, base*8, kernel_size=3, padding=1, stride=1))
            self.dec4.add_module("dec4_relu_{}".format(i+1), torch.nn.ReLU())
            self.dec4.add_module("dec4_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*8))
        

        self.tconv3 = torch.nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.tconv3_bn = torch.nn.BatchNorm2d(base*4)

        self.dec3 = torch.nn.Sequential()
        for i in range(2):
            f = base*8 if i == 0 else base*4
            self.dec3.add_module("dec3_{}".format(i+1), torch.nn.Conv2d(f, base*4, kernel_size=3, padding=1, stride=1))
            self.dec3.add_module("dec3_relu_{}".format(i+1), torch.nn.ReLU())
            self.dec3.add_module("dec3_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*4))

        self.tconv2 = torch.nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.tconv2_bn = torch.nn.BatchNorm2d(base*2)

        self.dec2 = torch.nn.Sequential()
        for i in range(2):
            f = base*4 if i == 0 else base*2
            self.dec2.add_module("dec2_{}".format(i+1), torch.nn.Conv2d(f, base*2, kernel_size=3, padding=1, stride=1))
            self.dec2.add_module("dec2_relu_{}".format(i+1), torch.nn.ReLU())
            self.dec2.add_module("dec2_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*2))

        self.tconv1 = torch.nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)
        self.tconv1_bn = torch.nn.BatchNorm2d(base)

        self.dec1 = torch.nn.Sequential()
        for i in range(2):
            f = base*2 if i == 0 else base
            self.dec1.add_module("dec1_{}".format(i+1), torch.nn.Conv2d(f, base, kernel_size=3, padding=1, stride=1))
            self.dec1.add_module("dec1_relu_{}".format(i+1), torch.nn.ReLU())
            self.dec1.add_module("dec1_bn_{}".format(i+1), torch.nn.BatchNorm2d(base))

        self.out = torch.nn.Conv2d(base, class_num+1, kernel_size=1, padding=0, stride=1)
        
        
    def forward(self, x):
        # block conv1
        x_enc1 = self.enc1(x)
        x = F.max_pool2d(x_enc1, 2, stride=2, padding=0)
        
        # block conv2
        x_enc2 = self.enc2(x)
        x = F.max_pool2d(x_enc2, 2, stride=2, padding=0)
        
        # block conv31
        x_enc3 = self.enc3(x)
        x = F.max_pool2d(x_enc3, 2, stride=2, padding=0)
        
        # block conv4
        x_enc4 = self.enc4(x)
        x = F.max_pool2d(x_enc4, 2, stride=2, padding=0)
        
        # block conv5
        x = self.enc5(x)

        x = self.tconv4_bn(self.tconv4(x))

        x = torch.cat((x, x_enc4), dim=1)
        x = self.dec4(x)

        x = self.tconv3_bn(self.tconv3(x))

        x = torch.cat((x, x_enc3), dim=1)
        x = self.dec3(x)

        x = self.tconv2_bn(self.tconv2(x))
        x = torch.cat((x, x_enc2), dim=1)
        x = self.dec2(x)

        x = self.tconv1_bn(self.tconv1(x))
        x = torch.cat((x, x_enc1), dim=1)
        x = self.dec1(x)

        x = self.out(x)
        
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

            for i, (_, vs) in enumerate(CLS.items()):
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
    
    for i in range(1000):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            mbi = mb - (len(xs) - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(ts[mb_ind], dtype=torch.long).to(device)

        opt.zero_grad()
        y = model(x)

        y = y.permute(0,2,3,1).contiguous()
        y = y.view(-1, class_num+1)
        t = t.view(-1)
        
        y = F.log_softmax(y, dim=1)
        loss = torch.nn.CrossEntropyLoss()(y, t)
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item() / mb
        
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
    
        pred = pred.permute(0,2,3,1).reshape(-1, class_num+1)
        pred = F.softmax(pred, dim=1)
        pred = pred.reshape(-1, out_height, out_width, class_num+1)
        pred = pred.detach().cpu().numpy()[0]
        pred = pred.argmax(axis=-1)

        # visualize
        out = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        for i, (_, vs) in enumerate(CLS.items()):
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
