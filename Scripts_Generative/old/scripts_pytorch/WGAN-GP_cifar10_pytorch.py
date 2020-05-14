import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from copy import copy
import os
from collections import OrderedDict
import pickle

CLS = {'akahara': [0,0,128],
       'madara': [0,128,0]}

class_num = len(CLS)
img_height, img_width = 32, 32 #572, 572
channel = 3
mb = 64

# GAN parameter
Z_dim = 128

# Gradient penalty parameter
Lambda = 10

model_path = 'WGAN_GP.pt'

save_dir = 'output_gan'
os.makedirs(save_dir, exist_ok=True)


# GPU
GPU = True
device = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")

torch.manual_seed(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class Flatten(torch.nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Reshape(torch.nn.Module):
    def __init__(self, c, h, w):
        super(Reshape, self).__init__()
        self.c = c
        self.h = h
        self.w = w
    
    def forward(self, x):
        x = x.view(x.size()[0], self.c, self.h, self.w)
        return x
    
    
# ResNet block
class ResBlock(torch.nn.Module):
    def __init__(self, dim_first=None, dim=128, activation_fn=torch.nn.ReLU(), batch_norm=False):
        super(ResBlock, self).__init__()

        if dim_first is None:
            dim_first = dim
        else:
            if batch_norm:
                self.skip_conv = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(dim_first),
                    activation_fn,
                    torch.nn.Conv2d(dim_first, dim, kernel_size=3, padding=1, stride=1)
                )
            else:
                self.skip_conv = torch.nn.Sequential(
                    activation_fn,
                    torch.nn.Conv2d(dim_first, dim, kernel_size=3, padding=1, stride=1)
                )
        
        if batch_norm:
            self.block = torch.nn.Sequential(
                torch.nn.BatchNorm2d(dim_first),
                activation_fn,
                torch.nn.Conv2d(dim_first, dim, kernel_size=3, padding=1, stride=1),
                torch.nn.BatchNorm2d(dim),
                activation_fn,
                torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
            )
        else:
            self.block = torch.nn.Sequential(
                activation_fn,
                torch.nn.Conv2d(dim_first, dim, kernel_size=3, padding=1, stride=1),
                activation_fn,
                torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
            )


    def forward(self, x):
        res_x = self.block(x)

        if hasattr(self, 'skip_conv'):
            x = self.skip_conv(x)
        
        x = torch.add(res_x, x)
        x = F.relu(x)
        return x
        
    
class Generator(torch.nn.Module):

    def __init__(self):
        in_h = img_height // 8
        in_w = img_width // 8
        dim = 128
        
        super(Generator, self).__init__()
        
        self.module = torch.nn.Sequential(
            torch.nn.Linear(Z_dim, dim * in_h * in_w),
            Reshape(dim, in_h, in_w),
            torch.nn.BatchNorm2d(dim),
            torch.nn.ReLU(),     
            
            ResBlock(dim=dim, activation_fn=torch.nn.ReLU(), batch_norm=True),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            
            ResBlock(dim=dim, activation_fn=torch.nn.ReLU(), batch_norm=True),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            
            ResBlock(dim=dim, activation_fn=torch.nn.ReLU(), batch_norm=True),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
        
            #ResBlock(dim=dim, activation_fn=torch.nn.ReLU(), batch_norm=True),

            torch.nn.Conv2d(dim, channel, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
        )
        
    def forward(self, x):
        x = self.module(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        dim = 128
        
        self.module = torch.nn.Sequential(
            ResBlock(dim_first=channel, dim=dim, activation_fn=torch.nn.LeakyReLU(0.2), batch_norm=False),
            #torch.nn.Conv2d(channel, dim, kernel_size=3, padding=1, stride=1),
            #torch.nn.LeakyReLU(0.2),
            torch.nn.AvgPool2d(2, stride=2),

            ResBlock(dim=dim, activation_fn=torch.nn.LeakyReLU(0.2), batch_norm=False),
            torch.nn.AvgPool2d(2, stride=2),

            ResBlock(dim=dim, activation_fn=torch.nn.LeakyReLU(0.2), batch_norm=False),

            ResBlock(dim=dim, activation_fn=torch.nn.LeakyReLU(0.2), batch_norm=False),
            torch.nn.ReLU(),

            torch.nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            #torch.nn.Linear(dim * (img_height // 8) * (img_width // 8), 1),
            torch.nn.Linear(dim, 1),
            #torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.module(x)
        return x

    
def load_cifar10():

    path = 'cifar-10-batches-py'

    if not os.path.exists(path):
        os.system("wget {}".format('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'))
        os.system("tar xvf {}".format('cifar-10-python.tar.gz'))

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

    print(train_x.shape)
    print(train_y.shape)

    # test data
    
    data_path =  path + '/test_batch'
    
    with open(data_path, 'rb') as f:
        datas = pickle.load(f, encoding='bytes')
        print(data_path)
        x = datas[b'data']
        x = x.reshape(x.shape[0], 3, 32, 32)
        test_x = x.transpose(0, 2, 3, 1)
    
        test_y = np.array(datas[b'labels'], dtype=np.int)

    print(test_x.shape)
    print(test_y.shape)

    return train_x, train_y, test_x, test_y


# train
def train():
    # model
    G = Generator().to(device)
    D = Discriminator().to(device)
    
    G.apply(weights_init)
    D.apply(weights_init)
    
    # wgan hyper-parameter
    n_critic = 5

    opt_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0, 0.9))
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0, 0.9))


    #xs, paths = data_load('drive/My Drive/Colab Notebooks/datasets/', hf=True, vf=True, rot=False)
    train_x, train_y, test_x, test_y = load_cifar10()
    xs = train_x / 127.5 - 1
    xs = xs.transpose(0, 3, 1, 2)

    # training
    mbi = 0
    data_N = len(xs)
    train_ind = np.arange(data_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    one = torch.FloatTensor([1])
    minus_one = one * -1
    if GPU:
        one = one.cuda()
        minus_one = mone.cuda()
    
    for ite in range(50000):
        if mbi + mb > data_N:
            mb_ind = copy(train_ind[mbi:])
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (data_N - mbi))]))
            mbi = mb - (data_N - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        # Discriminator training
        for _ in range(n_critic):
            opt_D.zero_grad()

            # sample x from dataset
            x = xs[mb_ind]
            x = torch.tensor(x, dtype=torch.float).to(device)

            # sample z from uniform distribution [-1, 1]
            z = np.random.uniform(-1, 1, size=(mb, Z_dim))
            z = torch.tensor(z, dtype=torch.float).to(device)

            Gz = G(z)

            # sample epsilon from [0, 1]
            epsilon = np.random.random() #np.random.uniform(0, 1, 1)

            # sample x_hat 
            x_hat = (epsilon * x + (1 - epsilon) * Gz).requires_grad_(True)

            # loss for fake
            loss_D_fake = D(Gz).mean(0).view(1)
            # loss for real
            loss_D_real = D(x).mean(0).view(1)

            # gradient penalty
            Dx_hat = D(x_hat)
            musk = torch.ones_like(Dx_hat)
            gradients = torch.autograd.grad(Dx_hat, x_hat, grad_outputs=musk,
                             retain_graph=True, create_graph=True,
                             allow_unused=True)[0]
            gradients = gradients.view(-1, 1)
            gradient_penalty = Lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            # loss backpropagation
            loss_D_real.backward(one, retain_graph=True)
            loss_D_fake.backward(minus_one, retain_graph=True)
            gradient_penalty.backward(retain_graph=True)

            # total loss
            loss_D = loss_D_fake - loss_D_real + gradient_penalty

            opt_D.step()

        # Generator training
        opt_G.zero_grad()

        # sample z from normal distribution [-1, 1]
        z = np.random.uniform(-1, 1, size=(mb, Z_dim))
        z = torch.tensor(z, dtype=torch.float).to(device)
        
        # loss for fake
        loss_G = D(G(z)).mean(0).view(1)

        # loss backpropagation
        loss_G.backward(one)
        opt_G.step()

        if (ite + 1) % 1000 == 0:
            print("iter :", ite + 1, ", G:loss :", loss_G.item(), ",D:loss :", loss_D.item())
            
            
            # save training process Generator output
            img_N = 16
            z = np.random.uniform(-1, 1, size=(img_N, Z_dim))
            z = torch.tensor(z, dtype=torch.float).to(device)

            Gz = G(z)

            if GPU:
                Gz = Gz.cpu()

            Gz = Gz.detach().numpy()
            Gz = (Gz + 1) / 2
            Gz = Gz.transpose(0,2,3,1)

            for i in range(img_N):
                generated = Gz[i]
                plt.subplot(1, img_N, i+1)
                plt.imshow(generated)
                plt.axis('off')

            plt.savefig('{}/WGAN-gp_iter_{:05d}.jpg'.format(save_dir, ite + 1), bbox_inches='tight')
            plt.show()
            

    torch.save(G.state_dict(), model_path)
    
    

# test
def test():
    # load Generator
    G = Generator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    G.eval()

    np.random.seed(100)
    
    with torch.no_grad():
        for i in range(3):
            mb = 10
            z = np.random.uniform(-1, 1, size=(mb, Z_dim))
            z = torch.tensor(z, dtype=torch.float).to(device)

            Gz = G(z)

            if GPU:
                Gz = Gz.cpu()
                
            Gz = Gz.detach().numpy()
            Gz = (Gz + 1) / 2
            Gz = Gz.transpose(0,2,3,1)

            for i in range(mb):
                generated = Gz[i]
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
