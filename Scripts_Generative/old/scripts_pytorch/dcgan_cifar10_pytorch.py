import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

num_classes = 2
img_height, img_width = 32, 32
channel = 3
mb = 64

# GPU
GPU = False
device = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
    
class Generator(torch.nn.Module):

    def __init__(self):
        self.in_h = img_height // 16
        self.in_w = img_width // 16
        self.base = 128
        
        super(Generator, self).__init__()
        #self.lin = torch.nn.Linear(100, self.in_h * self.in_w * self.base * 8)
        self.lin = torch.nn.ConvTranspose2d(100, self.base * 8, kernel_size=self.in_h, stride=1, bias=False)
        self.bnin = torch.nn.BatchNorm2d(self.base * 8)
        self.l1 = torch.nn.ConvTranspose2d(self.base* 8, self.base * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.base * 4)
        self.l2 = torch.nn.ConvTranspose2d(self.base * 4, self.base * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(self.base * 2)
        self.l3 = torch.nn.ConvTranspose2d(self.base * 2, self.base, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(self.base)
        self.l4 = torch.nn.ConvTranspose2d(self.base, channel, kernel_size=4, stride=2, padding=1, bias=False)
        
        
    def forward(self, x):
        x = self.lin(x)
        x = self.bnin(x)
        #x = x.view([-1, self.base*8, self.in_h, self.in_w])
        x = torch.nn.functional.relu(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.l3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = self.l4(x)
        x = torch.nn.functional.tanh(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        self.base = 32
        
        super(Discriminator, self).__init__()
        self.l1 = torch.nn.Conv2d(channel, self.base, kernel_size=5, padding=2, stride=2)
        self.l2 = torch.nn.Conv2d(self.base, self.base * 2, kernel_size=5, padding=2, stride=2)
        #self.bn2 = torch.nn.BatchNorm2d(self.base * 2)
        self.l3 = torch.nn.Conv2d(self.base * 2, self.base * 4, kernel_size=5, padding=2, stride=2)
        #self.bn3 = torch.nn.BatchNorm2d(self.base * 4)
        self.l4 = torch.nn.Conv2d(self.base * 4, self.base * 8, kernel_size=5, padding=2, stride=2)
        #self.bn4 = torch.nn.BatchNorm2d(self.base * 8)
        self.l5 = torch.nn.Linear((img_height // 16) * (img_width // 16) * self.base * 8, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.l2(x)
        #x = self.bn2(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.l3(x)
        #x = self.bn3(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.l4(x)
        #x = self.bn4(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = x.view([-1, (img_height // 16) * (img_width // 16) * self.base * 8])
        x = self.l5(x)
        x = torch.nn.functional.sigmoid(x)
        return x


    
class GAN(torch.nn.Module):
    def __init__(self, g, d):
        super(GAN, self).__init__()
        self.g = g
        self.d = d
        
    def forward(self, x, y):
        x = self.g(x, y)
        x = self.d(x)
        return x
    

import pickle
import os
    
def load_cifar10():

    path = 'cifar-10-batches-py'

    if not os.path.exists(path):
        os.system("wget {}".format(path))
        os.system("tar xvf {}".format(path))

    # train data
    
    train_x = np.ndarray([0, 32, 32, 3], dtype=np.float32)
    train_y = np.ndarray([0, ], dtype=np.int)
    
    for i in range(1, 6):
        data_path = path + '/data_batch_{}'.format(i)
        with open(data_path, 'rb') as f:
            datas = pickle.load(f, encoding='bytes')
            print(data_path)
            x = datas[b'data']
            x = x.reshape(x.shape[0], 3, 32, 32)
            x = x.transpose(0, 2, 3, 1)
            train_x = np.vstack((train_x, x))
        
            y = np.array(datas[b'labels'], dtype=np.int)
            train_y = np.hstack((train_y, y))

    # test data
    
    data_path = path + '/test_batch'
    
    with open(data_path, 'rb') as f:
        datas = pickle.load(f, encoding='bytes')
        print(data_path)
        x = datas[b'data']
        x = x.reshape(x.shape[0], 3, 32, 32)
        test_x = x.transpose(0, 2, 3, 1)
    
        test_y = np.array(datas[b'labels'], dtype=np.int)

    return train_x, train_y, test_x, test_y


# train
def train():
    # model
    gen = Generator().to(device)
    dis = Discriminator().to(device)
    gan = Gan(gen, dis)
    #gan = torch.nn.Sequential(gen, dis)

    opt_d = torch.optim.Adam(dis.parameters(), lr=0.0002,  betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))

    train_x, train_y, test_x, test_y = load_cifar10()
    xs = train_x / 127.5 - 1
    xs = xs.transpose(0, 3, 1, 2)

    # training
    mbi = 0
    data_N = len(xs)
    train_ind = np.arange(data_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(20000):
        if mbi + mb > data_N:
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (data_N - mbi))]))
            mbi = mb - (data_N - mbi)
        else:
            mb_ind = train_ind[mbi: mbi + mb]
            mbi += mb

        opt_d.zero_grad()
        opt_g.zero_grad()
            
        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)

        #for param in dis.parameters():
        #    param.requires_grad = True
        #dis.train()
        input_noise = np.random.uniform(-1, 1, size=(mb, 100, 1, 1))
        input_noise = torch.tensor(input_noise, dtype=torch.float).to(device)
        g_output = gen(input_noise)

        X = torch.cat([x, g_output])
        t = [1] * mb + [0] * mb
        t = torch.tensor(t, dtype=torch.float).to(device)

        dy = dis(X)[..., 0]
        loss_d = torch.nn.BCELoss()(dy, t)

        loss_d.backward()
        opt_d.step()

        #for param in dis.parameters():
        #    param.requires_grad = False
        #dis.eval()
        #gen.train()
        input_noise = np.random.uniform(-1, 1, size=(mb, 100, 1, 1))
        input_noise = torch.tensor(input_noise, dtype=torch.float).to(device)
        y = gan(input_noise)[..., 0]
        t = torch.tensor([1] * mb, dtype=torch.float).to(device)
        loss_g = torch.nn.BCELoss()(y, t)

        loss_g.backward()
        opt_g.step()

        if (i+1) % 100 == 0:
            print("iter >>", i+1, ',G:loss >>', loss_g.item(), ',D:loss >>', loss_d.item())

    torch.save(gen.state_dict(), 'cnn.pt')

# test
def test():
    gen = Generator().to(device)
    gen.eval()
    gen.load_state_dict(torch.load('cnn.pt'))

    np.random.seed(100)
    
    for i in range(3):
        mb = 10
        input_noise = np.random.uniform(-1, 1, size=(mb, 100, 1, 1))
        input_noise = torch.tensor(input_noise, dtype=torch.float).to(device)

        g_output = gen(input_noise)

        if GPU:
            g_output = g_output.cpu()
            
        g_output = g_output.detach().numpy()
        g_output = (g_output + 1) / 2
        g_output = g_output.transpose(0,2,3,1)

        for i in range(mb):
            generated = g_output[i]
            plt.subplot(1,mb,i+1)
            plt.imshow(generated)
            plt.axis('off')

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
