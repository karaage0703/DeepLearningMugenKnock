import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import gzip

# config
class_N = 10

img_height, img_width = 28, 28
channel = 1

torch.manual_seed(0)

# save directory
save_dir = 'output_gan'
os.makedirs(save_dir, exist_ok=True)

# model path
model_path = 'CGAN.pt'
    
# GPU
GPU = False
device = torch.device("cuda" if GPU else "cpu")
    
class Generator(torch.nn.Module):

    def __init__(self):
        self.in_h = img_height // 4
        self.in_w = img_width // 4
        self.base = 128
        
        super(Generator, self).__init__()
        
        self.lin = torch.nn.ConvTranspose2d(100 +class_N, self.base * 2, kernel_size=self.in_h, stride=1, bias=False)
        self.bnin = torch.nn.BatchNorm2d(self.base * 2)

        self.l3 = torch.nn.ConvTranspose2d(self.base * 2, self.base, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(self.base)
        self.l4 = torch.nn.ConvTranspose2d(self.base, channel, kernel_size=4, stride=2, padding=1, bias=False)
        
        
    def forward(self, x, y, test=False):
        #x = torch.cat((x, y), dim=1)
        con_x = np.zeros((len(y),class_N, 1, 1), dtype=np.float32)
        con_x[np.arange(len(y)), y] = 1
        con_x = torch.tensor(con_x, dtype=torch.float).to(device)
        
        x = torch.cat((x, con_x), dim=1)
        
        x = self.lin(x)
        x = self.bnin(x)
        
        x = torch.nn.functional.relu(x)
        x = self.l3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = self.l4(x)
        x = torch.tanh(x)

        if test:
            return x
        
        else:
            con_x = np.zeros((len(y),class_N, img_height, img_width), dtype=np.float32)
            con_x[np.arange(len(y)), y] = 1
            con_x = torch.tensor(con_x).to(device)
        
            out_x = torch.cat((x, con_x), dim=1)
        
            return out_x


class Discriminator(torch.nn.Module):
    def __init__(self):
        self.base = 64
        
        super(Discriminator, self).__init__()
        self.l1 = torch.nn.Conv2d(channel +class_N, self.base, kernel_size=5, padding=2, stride=2)
        self.l2 = torch.nn.Conv2d(self.base, self.base * 2, kernel_size=5, padding=2, stride=2)
        #self.bn2 = torch.nn.BatchNorm2d(self.base * 2)
        self.l5 = torch.nn.Linear((img_height // 4) * (img_width // 4) * self.base * 2, 1)

    def forward(self, x):
        
        x = self.l1(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.l2(x)
        #x = self.bn2(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = x.view([-1, (img_height // 4) * (img_width // 4) * self.base * 2])
        x = self.l5(x)
        x = torch.sigmoid(x)
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


def load_mnist():
    dir_path = 'drive/My Drive/Colab Notebooks/'  + "mnist_datas"

    files = ["train-images-idx3-ubyte.gz",
             "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz",
             "t10k-labels-idx1-ubyte.gz"]

    # download mnist datas
    if not os.path.exists(dir_path):

        os.makedirs(dir_path)

        data_url = "http://yann.lecun.com/exdb/mnist/"

        for file_url in files:

            after_file = file_url.split('.')[0]
            
            if os.path.exists(dir_path + '/' + after_file):
                continue
            
            os.system("wget {}/{}".format(data_url, file_url))
            os.system("mv {} {}".format(file_url, dir_path))

        
    # load mnist data

    # load train data
    with gzip.open(dir_path + '/' + files[0], 'rb') as f:
        train_x = np.frombuffer(f.read(), np.uint8, offset=16)
        train_x = train_x.astype(np.float32)
        train_x = train_x.reshape((-1, 28, 28, 1))
        print("train images >>", train_x.shape)

    with gzip.open(dir_path + '/' + files[1], 'rb') as f:
        train_y = np.frombuffer(f.read(), np.uint8, offset=8)
        print("train labels >>", train_y.shape)

    # load test data
    with gzip.open(dir_path + '/' + files[2], 'rb') as f:
        test_x = np.frombuffer(f.read(), np.uint8, offset=16)
        test_x = test_x.astype(np.float32)
        test_x = test_x.reshape((-1, 28, 28, 1))
        print("test images >>", test_x.shape)
    
    with gzip.open(dir_path + '/' + files[3], 'rb') as f:
        test_y = np.frombuffer(f.read(), np.uint8, offset=8)
        print("test labels >>", test_y.shape)
        

    return train_x, train_y ,test_x, test_y

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

    ys = np.zeros([train_y.shape[0],class_N, 1, 1], np.float32)
    ys[np.arange(train_y.shape[0]), train_y] = 1

    # training
    mb = 64
    mbi = 0
    train_N = len(xs)
    train_ind = np.arange(train_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(20000):
        if mbi + mb > train_N:
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (train_N - mbi))]))
            mbi = mb - (train_N - mbi)
        else:
            mb_ind = train_ind[mbi : mbi + mb]
            mbi += mb

        opt_d.zero_grad()
        opt_g.zero_grad()
            
        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        x_con = torch.tensor(ys[mb_ind], dtype=torch.float).to(device)
        
        #for param in dis.parameters():
        #    param.requires_grad = True
        #dis.train()
        input_noise = np.random.uniform(-1, 1, size=(mb, 100, 1, 1))
        input_noise = torch.tensor(input_noise, dtype=torch.float).to(device)
        g_output = gen(torch.cat((input_noise, x_con), dim=1))

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
        
        y = gan(torch.cat((input_noise, x_con), dim=1))[..., 0]
        t = torch.tensor([1] * mb, dtype=torch.float).to(device)
        loss_g = torch.nn.BCELoss()(y, t)

        loss_g.backward()
        opt_g.step()

        if (i+1) % 100 == 0:
            print("iter >>", i+1, ',G:loss >>', loss_g.item(), ',D:loss >>', loss_d.item())

    torch.save(gen.state_dict(), model_path)

# test
def test():
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()

    np.random.seed(100)
    
    with torch.no_grad():
        for i in range(3):
            mb = 10
            input_noise = np.random.uniform(-1, 1, size=(mb, 100, 1, 1))
            input_noise = torch.tensor(input_noise, dtype=torch.float).to(device)

            y = np.arange(num_classes, dtype=np.int)

            g_output = gen(input_noise, y, test=True)

            if GPU:
                g_output = g_output.cpu()
                
            g_output = g_output.detach().numpy()
            g_output = (g_output + 1) / 2
            g_output = g_output.transpose(0,2,3,1)

            if channel == 1:
                cmap = 'gray'
            else:
                cmap = None

            for i in range(mb):
                generated = g_output[i, ..., 0]
                plt.subplot(1,mb,i+1)
                plt.title(str(i))
                plt.imshow(generated, cmap=cmap)
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
