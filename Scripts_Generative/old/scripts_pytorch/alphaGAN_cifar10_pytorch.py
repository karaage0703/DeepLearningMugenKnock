import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
from copy import copy


# embedding vector
Z_dim = 128

# L1 reconstruction loss balance
reconstruction_loss_lambda = 1.
# to avoid log(0) of loss
epsilon = 1e-12

img_height, img_width = 32, 32
channel = 3

GPU = True
device = torch.device("cuda" if GPU else "cpu")

torch.manual_seed(0)

save_dir = 'output_gan'
os.makedirs(save_dir, exist_ok=True)

model_path_G = 'alphaGAN_G.pt'
model_path_D = 'alphaGAN_D.pt'
model_path_E = 'alphaGAN_E.pt'
model_path_CD = 'alphaGAN_CD.pt'

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


        
        
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        in_h = img_height // 8
        in_w = img_width // 8
        dim = 128
        
        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(channel, dim, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(dim),
            torch.nn.ReLU(),
            
            ResBlock(dim=dim, activation_fn=torch.nn.ReLU(), batch_norm=True),
            torch.nn.MaxPool2d(2, stride=2),
            
            ResBlock(dim=dim, activation_fn=torch.nn.ReLU(), batch_norm=True),
            torch.nn.MaxPool2d(2, stride=2),
            
            ResBlock(dim=dim, activation_fn=torch.nn.ReLU(), batch_norm=True),
            torch.nn.MaxPool2d(2, stride=2),
            
            Flatten(),
            torch.nn.Linear(dim * in_h * in_w, Z_dim)
        )
        
    def forward(self, x):
        x = self.module(x)
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
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.module(x)
        return x

    
    
class Code_Discriminator(torch.nn.Module):
    def __init__(self):
        super(Code_Discriminator, self).__init__()
        
        hidden_dim = 750
        self.module = torch.nn.Sequential(
            torch.nn.Linear(Z_dim, hidden_dim),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(0.2),
            #torch.nn.Linear(hidden_dim, hidden_dim),
            #torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.module(x)
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
        data_path =  path + '/data_batch_{}'.format(i)
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
    
    data_path = path + '/test_batch'
    
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
    E = Encoder().to(device)
    CD = Code_Discriminator().to(device)
    

    opt_G = torch.optim.Adam(G.parameters(), lr=0.0005, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0005,  betas=(0.5, 0.9))
    opt_E = torch.optim.Adam(E.parameters(), lr=0.0001, betas=(0.5, 0.9))
    opt_CD = torch.optim.Adam(CD.parameters(), lr=0.0005, betas=(0.5, 0.9))

    train_x, train_y, test_x, test_y = load_cifar10()
    xs = train_x / 127.5 - 1
    xs = xs.transpose(0, 3, 1, 2)

    # training
    mb = 64
    mbi = 0
    train_N = len(xs)
    train_ind = np.arange(train_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    BCE_loss = torch.nn.BCELoss()
    L1_loss = torch.nn.L1Loss()


    # get next minibatch index
    def get_next_minibatch(train_ind, mbi, mb=mb):
        train_N = len(train_ind)
        if mbi + mb > train_N:
            mb_ind = copy(train_ind[mbi:])
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (train_N - mbi))]))
            mbi = mb - (train_N - mbi)
        else:
            mb_ind = train_ind[mbi: mbi + mb]
            mbi += mb

        return mb_ind, train_ind
    
    
    for i in range(100000):
        #if mbi + mb > train_N:
        #    mb_ind = copy(train_ind[mbi:])
        #    np.random.shuffle(train_ind)
        #    mb_ind = np.hstack((mb_ind, train_ind[:(mb-(train_N-mbi))]))
        #    mbi = mb - (train_N - mbi)
        #else:
        #    mb_ind = train_ind[mbi: mbi+mb]
        #    mbi += mb

        mb_ind, train_ind = get_next_minibatch(train_ind, mbi, mb=mb)

        z = np.random.randn(mb, Z_dim)
        z = torch.tensor(z, dtype=torch.float).to(device)
        #----
        # update
        
        # Encoder update
        opt_E.zero_grad()
    
        #loss_L1.backward(retain_graph=True)
        #loss_CD_Gen.backward(retain_graph=True)
        #loss_E = loss_L1 + loss_CD_Gen
        #loss_L1.backward()
        #R_Cw_z_hat.backward()
        #loss_E = loss_L1 + R_Cw_z_hat
        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        z_hat = E(x)
        x_hat = G(E(x))
    
        Cw_z_hat = CD(z_hat)
    
        loss_Reconstruction = reconstruction_loss_lambda * L1_loss(x, x_hat)
        loss_E = loss_Reconstruction 
        #loss_E += (- torch.log(Cw_z_hat + epsilon) + torch.log(1 - Cw_z_hat + epsilon)).mean()
        #loss_E += (- torch.log(Cw_z_hat + epsilon)).mean()
        loss_E += BCE_loss(Cw_z_hat, torch.ones(mb).to(device))
        loss_E.backward(retain_graph=True)
        
        opt_E.step()

        
        # Generator update
        opt_G.zero_grad()

        # get x from p(x)
        #mb_ind, train_ind = get_next_minibatch(train_ind, mbi, mb=mb)
        #x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)

        x_hat = G(E(x))
        z_hat = E(x)
        Dphi_x_hat = D(x_hat)
    
        # get z from p(z)
        z = np.random.randn(mb, Z_dim)
        z = torch.tensor(z, dtype=torch.float).to(device)
        Dphi_Gz = D(G(z))
    
        loss_G = reconstruction_loss_lambda * L1_loss(x, x_hat) # Reconstruction loss
        #loss_G += (- torch.log(Dphi_x_hat + epsilon) + torch.log(1 - Dphi_x_hat + epsilon)).mean() # R_Dphi_x_hat loss
        #loss_G += (- torch.log(Dphi_Gz + epsilon) + torch.log(1 - Dphi_Gz + epsilon)).mean() # R_Dphi_Gz loss
        #loss_G += (- torch.log(Dphi_Gz + epsilon)).mean()
        loss_G += BCE_loss(Dphi_Gz, torch.ones(mb).to(device))
        loss_G.backward(retain_graph=True)
        
        for _ in range(2):
            opt_G.step()

        
        # Discriminator update
        opt_D.zero_grad()
        
        # get x from p(x)
        #mb_ind, train_ind = get_next_minibatch(train_ind, mbi, mb=mb)
        #x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)

        Dphi_x = D(x)
        Dphi_x_hat = D(G(E(x)))
        z = np.random.randn(mb, Z_dim)
        z = torch.tensor(z, dtype=torch.float).to(device)
        Dphi_Gz = D(G(z))
        #loss_D = - torch.log(Dphi_x + epsilon).mean() - torch.log(1 - Dphi_x_hat + epsilon).mean()  - torch.log(1 - Dphi_Gz + epsilon).mean()
        #loss_D.backward(retain_graph=True)
        _loss_D = BCE_loss(Dphi_x, torch.ones(mb).to(device)) + BCE_loss(Dphi_x_hat, torch.zeros(mb).to(device))
        _loss_D.backward(retain_graph=True)
        loss_D = _loss_D
        _loss_D_ = BCE_loss(Dphi_Gz, torch.zeros(mb).to(device))
        _loss_D.backward(retain_graph=True)
        loss_D += _loss_D
        
        opt_D.step()
        
        
        # Code Discriminator update
        opt_CD.zero_grad()
        
        # get x from p(x)
        #mb_ind, train_ind = get_next_minibatch(train_ind, mbi, mb=mb)
        #x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)

        z_hat = E(x)    
        z = np.random.randn(mb, Z_dim)
        z = torch.tensor(z, dtype=torch.float).to(device)
        Cw_z = CD(z)
        Cw_z_hat = CD(z_hat)
        #loss_CD = - torch.log(1 - Cw_z_hat + epsilon).mean() - torch.log(Cw_z + epsilon).mean()
        loss_CD = BCE_loss(Cw_z, torch.ones(mb).to(device)) + BCE_loss(Cw_z_hat, torch.zeros(mb).to(device))
        loss_CD.backward(retain_graph=True)
        
        opt_CD.step()
        
        
        if (i + 1) % 50 == 0:
            print('iter : {} , Loss E : {:.5f} , G : {:.5f} , D : {:.5f} , CD : {:.5f}'.format(
                i + 1, loss_E.item(), loss_G.item(), loss_D.item(), loss_CD.item()))

        if (i + 1) % 1000 == 0:
            img_N = 16
            z = np.random.randn(img_N, Z_dim)
            z = torch.tensor(z, dtype=torch.float).to(device)

            Gz = G(z)

            if GPU:
                Gz = Gz.cpu()

            Gz = Gz.detach().numpy()
            Gz = (Gz + 1) / 2
            Gz = Gz.transpose(0, 2, 3, 1)

            for j in range(img_N):
                generated = Gz[j]
                plt.subplot(1, img_N, j + 1)
                plt.imshow(generated)
                plt.axis('off')

            plt.savefig('{}/alphaGAN_iter_{:05d}.jpg'.format(save_dir, i + 1), bbox_inches='tight')
            plt.close()

    torch.save(G.state_dict(), model_path_G)
    torch.save(D.state_dict(), model_path_D)
    torch.save(E.state_dict(), model_path_E)
    torch.save(CD.state_dict(), model_path_CD)

# test
def test():
    G = Generator().to(device)
    G.load_state_dict(torch.load(model_path_G, map_location=torch.device(device)))
    G.eval()
    
    D = Discriminator().to(device)
    D.load_state_dict(torch.load(model_path_D, map_location=torch.device(device)))
    D.eval()
    
    E = Encoder().to(device)
    E.load_state_dict(torch.load(model_path_E, map_location=torch.device(device)))
    E.eval()
    
    CD = Code_Discriminator().to(device)
    CD.load_state_dict(torch.load(model_path_CD, map_location=torch.device(device)))
    CD.eval()

    np.random.seed(100)
    
    with torch.no_grad():
        for i in range(3):
            mb = 10
            #z = np.random.uniform(-1, 1, size=(mb, Z_dim))
            z = np.random.randn(mb, Z_dim)
            z = torch.tensor(z, dtype=torch.float).to(device)

            Gz = G(z)

            if GPU:
                Gz = Gz.cpu()

            Gz = Gz.detach().numpy()
            Gz = (Gz + 1) / 2
            Gz = Gz.transpose(0,2,3,1)

            for j in range(mb):
                generated = Gz[j]
                plt.subplot(1, mb, j + 1)
                plt.imshow(generated)
                plt.axis('off')

            plt.savefig(save_dir + '/alphaGAN_test_{}.jpg'.format(i))
            plt.show()


def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args

train()
test()
