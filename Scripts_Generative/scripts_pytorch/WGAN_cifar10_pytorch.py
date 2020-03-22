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

# config
CLS = {'akahara': [0,0,128],
       'madara': [0,128,0]}

class_N = len(CLS)
img_height, img_width = 32, 32 #572, 572
channel = 3
mb = 64

# GAN parameter
Z_dim = 128

# Gradient penalty parameter
Lambda = 10

model_path = 'WGAN.pt'

save_dir = 'output_gan'
os.makedirs(save_dir, exist_ok=True)


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
    
    
class Generator(torch.nn.Module):

    def __init__(self):
        dim = 256
        
        super(Generator, self).__init__()
        
        self.module = torch.nn.Sequential(OrderedDict({
            "conv1": torch.nn.ConvTranspose2d(100, dim * 4, kernel_size=img_height // 8, stride=1, bias=False),
            "bn1": torch.nn.BatchNorm2d(dim * 4),
            "relu1": torch.nn.ReLU(),
            "conv2": torch.nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            "bn2": torch.nn.BatchNorm2d(dim * 2),
            "relu2": torch.nn.ReLU(),
            "conv3": torch.nn.ConvTranspose2d(dim * 2, dim, kernel_size=4, stride=2, padding=1, bias=False),
            "bn3": torch.nn.BatchNorm2d(dim),
            "relu3": torch.nn.ReLU(),
            "conv4": torch.nn.ConvTranspose2d(dim , channel, kernel_size=4, stride=2, padding=1, bias=False),
            "tanh": torch.nn.Tanh(),
        }))
        
    def forward(self, x):
        x = self.module(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        dim = 256
        
        super(Discriminator, self).__init__()
        
        self.module = torch.nn.Sequential(OrderedDict({
            "conv1": torch.nn.Conv2d(channel, dim, kernel_size=4, stride=2, padding=1),
            "bn1": torch.nn.BatchNorm2d(dim),
            "relu1": torch.nn.LeakyReLU(0.2),
            "conv2": torch.nn.Conv2d(dim,dim * 2, kernel_size=4, stride=2, padding=1),
            "bn2": torch.nn.BatchNorm2d(dim * 2),
            "relu2": torch.nn.LeakyReLU(0.2),
            "conv3": torch.nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1),
            "bn3": torch.nn.BatchNorm2d(dim * 4),
            "relu3": torch.nn.LeakyReLU(0.2),
            "conv4": torch.nn.Conv2d(dim * 4, 1, kernel_size=img_height // 8, stride=1, padding=0)
            #"flatten": Flatten(),
            #"linear1": torch.nn.Linear((img_height // 8) * (img_width // 8) * dim * 4, 1),
            #"sigmoid": torch.nn.Sigmoid(),
        }))

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
    clip_value = 0.01
    n_critic = 5

    opt_D = torch.optim.RMSprop(D.parameters(), lr=0.00005)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=0.00005)


    #xs, paths = data_load('drive/My Drive/Colab Notebooks/datasets/', hf=True, vf=True, rot=False)
    train_x, train_y, test_x, test_y = load_cifar10()
    xs = train_x / 127.5 - 1
    xs = xs.transpose(0, 3, 1, 2)

    # training
    mb = 64
    mbi = 0
    data_N = len(xs)
    train_ind = np.arange(data_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    one = torch.FloatTensor([1])
    mone = one * -1
    if GPU:
        one = one.cuda()
        minus_one = mone.cuda()
    
    for i in range(100000):
        if mbi + mb > len(xs):
            mb_ind = copy(train_ind[mbi:])
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - ( data_N - mbi))]))
            mbi = mb - (data_N - mbi)
        else:
            mb_ind = train_ind[mbi : mbi + mb]
            mbi += mb

        # Discriminator training
        #for _ in range(n_critic):
        
        for _ in range(n_critic):
            opt_D.zero_grad()
            
            # parameter clipping > [-clip_value, clip_value]
            for param in D.parameters():
                param.data.clamp_(-clip_value, clip_value)

            # sample real image from dataset
            x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)

            # sample noize z from normal distribution [-1, 1]
            z = np.random.uniform(-1, 1, size=(mb, 100, 1, 1))
            z = torch.tensor(z, dtype=torch.float).to(device)

            # feedforward G(z)
            Gz = G(z)

            # loss for fake
            loss_D_fake = D(Gz).mean(0).view(1)

            # loss for real
            loss_D_real = D(x).mean(0).view(1)
            loss_D_real.backward(one)
            loss_D_fake.backward(minus_one)

            # total loss
            loss_D = loss_D_fake - loss_D_real 

            Wasserstein_distance = loss_D_real - loss_D_fake
            opt_D.step()


        # Generator training
        opt_G.zero_grad()

        # sample noize z from normal distribution [-1, 1]
        z = np.random.uniform(-1, 1, size=(mb, 100, 1, 1))
        z = torch.tensor(z, dtype=torch.float).to(device)
    
        # loss for fake
        loss_G = D(G(z)).mean(0).view(1)

        loss_G.backward(one)
        opt_G.step()

        if (i + 1) % 50 == 0:
            print("iter :", i+1, "WDistance :", Wasserstein_distance.item(),  ", G:loss :", loss_G.item(), ",D:loss :", loss_D.item())
            
        if (i + 1) % 100 == 0:
            # save training process Generator output
            img_N = 16
            z = np.random.uniform(-1, 1, size=(img_N, 100, 1, 1))
            z = torch.tensor(z, dtype=torch.float).to(device)

            Gz = G(z)

            if GPU:
                Gz = Gz.cpu()

            Gz = Gz.detach().numpy()
            Gz = (Gz + 1) / 2
            Gz = Gz.transpose(0,2,3,1)

            for j in range(img_N):
                generated = Gz[j]
                plt.subplot(1, img_N, j + 1)
                plt.imshow(generated)
                plt.axis('off')

            plt.savefig('{}/wgan_iter_{:05d}.jpg'.format(save_dir, i + 1), bbox_inches='tight')
            

    torch.save(G.state_dict(), model_path)
    
    

# test
def test():
    G = Generator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    G.eval()
    
    np.random.seed(100)
    
    with torch.no_grad():
        for i in range(3):
            mb = 10
            z = np.random.uniform(-1, 1, size=(mb, 100, 1, 1))
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
