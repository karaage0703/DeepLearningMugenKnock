import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from copy import copy
from collections import OrderedDict


CLS = OrderedDict({
    "background": [0, 0, 0],
    "akahara": [0,0,128],
    "madara": [0,128,0]
      })

class_num = len(CLS)

img_height, img_width = 64, 64 #572, 572
out_height, out_width = 64, 64 #388, 388

# GPU
GPU = True
device = torch.device("cuda" if GPU else "cpu")

torch.manual_seed(0)

    
class Flatten(torch.nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x
    
class Interpolate(torch.nn.Module):
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x
    
    
class UNet_block(torch.nn.Module):
    def __init__(self, dim1, dim2, name):
        super(UNet_block, self).__init__()

        _module = OrderedDict()

        for i in range(2):
            f = dim1 if i == 0 else dim2
            _module["unet_{}_conv{}".format(name, i+1)] = torch.nn.Conv2d(f, dim2, kernel_size=3, padding=1, stride=1)
            _module["unet_{}_relu{}".format(name, i+1)] = torch.nn.ReLU()
            _module["unet_{}_bn{}".format(name, i+1)] = torch.nn.BatchNorm2d(dim2)
            
            

        self.module = torch.nn.Sequential(_module)

    def forward(self, x):
        x = self.module(x)
        return x

class UNet_deconv_block(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super(UNet_deconv_block, self).__init__()

        self.module = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(dim1, dim2, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(dim2)
        )

    def forward(self, x):
        x = self.module(x)
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        base = 32
        
        self.enc1 = UNet_block(3, base, name="enc1")
        self.enc2 = UNet_block(base, base * 2, name="enc2")
        self.enc3 = UNet_block(base * 2, base * 4, name="enc3")
        self.enc4 = UNet_block(base * 4, base * 8, name="enc4")
        self.enc5 = UNet_block(base * 8, base * 16, name="enc5")

        self.tconv4 = UNet_deconv_block(base * 16, base * 8)
        self.tconv3 = UNet_deconv_block(base * 8, base * 4)
        self.tconv2 = UNet_deconv_block(base * 4, base * 2)
        self.tconv1 = UNet_deconv_block(base * 2, base)

        self.dec4 = UNet_block(base * 24, base * 8, name="dec4")
        self.dec3 = UNet_block(base * 12, base * 4, name="dec3")
        self.dec2 = UNet_block(base * 6, base * 2, name="dec2")
        self.dec1 = UNet_block(base * 3, base, name="dec1")

        self.out = torch.nn.Conv2d(base, class_num, kernel_size=1, padding=0, stride=1)
        
        
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

        #x = self.tconv4(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat((x, x_enc4), dim=1)
        x = self.dec4(x)

        #x = self.tconv3(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat((x, x_enc3), dim=1)
        x = self.dec3(x)

        #x = self.tconv2(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat((x, x_enc2), dim=1)
        x = self.dec2(x)

        #x = self.tconv1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat((x, x_enc1), dim=1)
        x = self.dec1(x)

        x = self.out(x)
        x = torch.tanh(x)
        #x = F.softmax(x, dim=1)
        #x = x * 2 - 1
        
        return x

    
    
class UNet2(torch.nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()

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

        self.out = torch.nn.Conv2d(base, class_num, kernel_size=1, padding=0, stride=1)
        
        
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
        x = torch.tanh(x)
        #x = F.softmax(x, dim=1)
        #x = x * 2 - 1
        
        return x
    

class Discriminator(torch.nn.Module):
    def __init__(self):
        self.base = 32
        
        super(Discriminator, self).__init__()
        
        self.module = torch.nn.Sequential(OrderedDict({
            "conv1": torch.nn.Conv2d(class_num * 2, self.base, kernel_size=5, padding=2, stride=2),
            "relu1": torch.nn.LeakyReLU(0.2),
            "conv2": torch.nn.Conv2d(self.base, self.base * 2, kernel_size=5, padding=2, stride=2),
            "relu2": torch.nn.LeakyReLU(0.2),
            "conv3": torch.nn.Conv2d(self.base * 2, self.base * 4, kernel_size=5, padding=2, stride=2),
            "relu3": torch.nn.LeakyReLU(0.2),
            "conv4": torch.nn.Conv2d(self.base * 4, self.base * 8, kernel_size=5, padding=2, stride=2),
            "relu4": torch.nn.LeakyReLU(0.2),
            "flatten": Flatten(),
            "linear1": torch.nn.Linear((img_height // 16) * (img_width // 16) * self.base * 8, 1),
            "sigmoid": torch.nn.Sigmoid(),
        }))

    def forward(self, x):
        x = self.module(x)
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
            x /= 127.5 - 1
            x = x[..., ::-1]
            xs.append(x)

            gt_path = path.replace("images", "seg_images").replace(".jpg", ".png")
            gt = cv2.imread(gt_path)
            gt = cv2.resize(gt, (out_width, out_height), interpolation=cv2.INTER_NEAREST)

            t = np.zeros((class_num, out_height, out_width), dtype=np.int)

            for i, (_, vs) in enumerate(CLS.items()):
                ind = (gt[...,0] == vs[0]) * (gt[...,1] == vs[1]) * (gt[...,2] == vs[2])
                t[i][ind] = 1

            ts.append(t)
            
            paths.append(path)

            if hf:
                xs.append(x[:, ::-1])
                ts.append(t[:, :, ::-1])
                paths.append(path)

            if vf:
                xs.append(x[::-1])
                ts.append(t[:, ::-1])
                paths.append(path)

            if hf and vf:
                xs.append(x[::-1, ::-1])
                ts.append(t[:, ::-1, ::-1])
                paths.append(path)

    xs = np.array(xs)
    ts = np.array(ts)
    
    ts = ts * 2 - 1

    xs = xs.transpose(0,3,1,2)
    
    return xs, ts, paths


# train
def train():
    # model
    G = UNet().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    D = Discriminator().to(device)
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    G.train()
    D.train()

    imgs, gts, paths = data_load('../Dataset/train/images/', hf=True, vf=True)

    # training
    mb = 32
    mbi = 0
    train_num = len(imgs)
    train_ind = np.arange(train_num)
    np.random.seed(0)
    np.random.shuffle(train_ind)
                          
    loss_fn = torch.nn.BCELoss()
    loss_l1 = torch.nn.L1Loss()
    
    Lambda = 1.
    
    for i in range(5000):
        if mbi + mb > train_num:
            mb_ind = copy(train_ind[mbi:])
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(train_num-mbi))]))
            mbi = mb - (train_num - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb
            
             
        opt_D.zero_grad()
        opt_G.zero_grad()
            
        # Discriminator training
        x = torch.tensor(imgs[mb_ind], dtype=torch.float).to(device)
        y = torch.tensor(gts[mb_ind], dtype=torch.float).to(device)

        Gx= G(x)
                          
        fake_x = torch.cat([Gx, x], dim=1)                    
        loss_D_fake = loss_fn(D(fake_x), torch.ones(mb, dtype=torch.float).to(device))
        
        real_x = torch.cat([y, x], dim=1)      
        loss_D_real = loss_fn(D(real_x), torch.zeros(mb, dtype=torch.float).to(device))

        oss_D = loss_D_real + loss_D_fake
        loss_D.backward(retain_graph=True)
        
        opt_D.step()
                          
            
        # UNet training
        loss_G_fake = loss_fn(D(fake_x), torch.zeros(mb, dtype=torch.float).to(device))
        loss_G_l1 = Lambda * loss_l1(Gx, x)
        loss_G = loss_G_fake + loss_G_l1
        loss_G.backward()
        
        opt_G.step()
        
        if (i+1) % 10 == 0:
            print("iter : ", i+1, ", loss D : ", loss_D.item(), ", loss G :", loss_G.item())

    torch.save(G.state_dict(), 'cnn.pt')


# test
def test():
    model = UNet().to(device)
    model.eval()
    model.load_state_dict(torch.load('cnn.pt'))

    xs, ts, paths = data_load('../Dataset/test/images/')

    for i in range(len(paths)):
        img = xs[i]
        t = ts[i]
        path = paths[i]
        
        x = np.expand_dims(img, axis=0)
        x = torch.tensor(x, dtype=torch.float).to(device)
        
        pred = model(x)
  
        pred = F.softmax(pred, dim=1)

        pred = pred.detach().cpu().numpy()[0]
        pred = pred.argmax(axis=0)

        # visualize
        out = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        for i, (label_name, v) in enumerate(CLS.items()):
            out[pred == i] = v

        print("in {}".format(path))
        
        plt.subplot(1,2,1)
        plt.imshow(((img.transpose(1, 2, 0) + 1) / 2).astype(np.float32))
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
