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

GPU = True
torch.manual_seed(0)
    
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        base = 256
        self.l1 = torch.nn.Linear(100, base)
        self.bn1 = torch.nn.BatchNorm1d(base)
        self.l2 = torch.nn.Linear(base, base * 2)
        self.bn2 = torch.nn.BatchNorm1d(base * 2)
        self.l3 = torch.nn.Linear(base * 2, base * 4)
        self.bn3 = torch.nn.BatchNorm1d(base * 4)
        self.l4 = torch.nn.Linear(base * 4, img_height * img_width * channel)
        
        
    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.l2(x)
        x = self.bn2(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.l3(x)
        x = self.bn3(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.l4(x)
        x = torch.nn.functional.tanh(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        base = 256
        self.l1 = torch.nn.Linear(img_height * img_width * channel, base * 2)
        self.l2 = torch.nn.Linear(base * 2, base)
        self.l3 = torch.nn.Linear(bsae, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.l2(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.l3(x)
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
    # GPU
    device = torch.device("cuda" if GPU else "cpu")

    # model
    gen = Generator().to(device)
    dis = Discriminator().to(device)
    gan = Gan(gen, dis)
    #gan = torch.nn.Sequential(gen, dis)

    opt_d = torch.optim.Adam(dis.parameters(), lr=0.0002)
    opt_g = torch.optim.Adam(gen.parameters(), lr=0.0002)


    train_x, train_y, test_x, test_y = load_cifar10()
    xs = train_x / 127.5 - 1
    xs = xs.transpose(0, 3, 1, 2)

    # training
    mb = 256
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(50000):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            mbi = mb - (len(xs) - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        opt_d.zero_grad()
        opt_g.zero_grad()
            
        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)

        #for param in dis.parameters():
        #    param.requires_grad = True
        #dis.train()
        input_noise = np.random.uniform(-1, 1, size=(mb, 100))
        input_noise = torch.tensor(input_noise, dtype=torch.float).to(device)
        g_output = gen(input_noise)
        g_output = torch.reshape(g_output, [mb, channel, img_height, img_width])

        X = torch.cat([x, g_output])
        X = X.view([mb * 2, -1])
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
        input_noise = np.random.uniform(-1, 1, size=(mb, 100))
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
    device = torch.device("cuda" if GPU else "cpu")

    gen = Generator().to(device)
    gen.eval()
    gen.load_state_dict(torch.load('cnn.pt'))

    np.random.seed(100)
    
    for i in range(3):
        mb = 10
        input_noise = np.random.uniform(-1, 1, size=(mb, 100))
        input_noise = torch.tensor(input_noise, dtype=torch.float).to(device)

        g_output = gen(input_noise)

        if GPU:
            g_output = g_output.cpu()
            
        g_output = g_output.detach().numpy()
        g_output = (g_output + 1) / 2
        g_output = g_output.reshape([mb, channel, img_height, img_width])
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
