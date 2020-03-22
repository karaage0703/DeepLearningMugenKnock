import torch
import torch.nn.functional as F
#import torchvision
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# config
class_N = 10
img_height, img_width = 28, 28
channel = 1

# GPU
GPU = False
device = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# other config
model_path = 'VAE.pt'
    
# VAE paramater
Z_dim = 2
dim = 256

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.enc1 = torch.nn.Linear(img_height * img_width * channel, dim)
        
        self.enc_mu = torch.nn.Linear(dim, Z_dim)
        self.enc_sigma = torch.nn.Linear(dim, Z_dim)
        
    def forward(self, x):
        mb, c, h, w = x.size()
        x = x.view(mb, -1)
        x = F.relu(self.enc1(x))
        mu = self.enc_mu(x)
        sigma = self.enc_sigma(x)
        self.mu = mu
        self.sigma = sigma
        
        return mu, sigma
    
    
class Sampler(torch.nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()
        
    def forward(self, x):
        mu, sigma = x
        mb, _ = mu.size()
        epsilon = torch.tensor(np.random.normal(0, 1, [mb, Z_dim]), dtype=torch.float32).to(device)
        std = torch.exp(0.5 * sigma)
        sample_z = mu + epsilon * std
        self.sample_z = sample_z
        return sample_z
    
    
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.dec1 = torch.nn.Linear(Z_dim, dim)

        self.dec_out = torch.nn.Linear(dim, img_height * img_width * channel)
        
        
    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = self.dec_out(x)
        x = torch.sigmoid(x)
        
        return x
        
def loss_KLDivergence(mu, sigma):
    return -0.5 * torch.sum(1 + sigma - torch.pow(mu, 2) - torch.exp(sigma))
        
def load_mnist():
    dir_path = "mnist_datas"

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
    model_encoder = Encoder().to(device)
    model_sampler = Sampler().to(device)
    model_decoder = Decoder().to(device)
    model = torch.nn.Sequential(model_encoder, model_sampler, model_decoder)
    
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    train_x, train_y, test_x, test_y = load_mnist()
    xs = train_x / 255
    xs = xs.transpose(0, 3, 1, 2)
    
    # training
    mb = 256
    mbi = 0
    data_N = len(xs)
    train_ind = np.arange(data_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(10000):
        if mbi + mb > data_N:
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (data_N - mbi))]))
            mbi = mb - (data_N - mbi)
        else:
            mb_ind = train_ind[mbi : mbi + mb]
            mbi += mb

        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)

        opt.zero_grad()

        #y_mu, y_sigma = model_encoder(x)
        #y = model_decoder(y_mu, y_sigma)
        y = model(x)
        y_mu = model_encoder.mu
        y_sigma = model_encoder.sigma
        
        #loss = torch.nn.BCELoss()(y, t.view(mb, -1))
        loss = F.binary_cross_entropy(y, t.view(mb, img_height * img_width * channel), reduction='sum')
        loss_kld = loss_KLDivergence(y_mu, y_sigma)
        loss = loss + loss_kld
        loss.backward()
        opt.step()
    
        #pred = y.argmax(dim=1, keepdim=True)
        acc = y.eq(t.view_as(y)).sum().item() / mb

        if (i+1) % 100 == 0:
            print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', acc)

    torch.save(model.state_dict(), model_path)

# test
def test():
    model_encoder = Encoder().to(device)
    model_sampler = Sampler().to(device)
    model_decoder = Decoder().to(device)
    model = torch.nn.Sequential(model_encoder, model_sampler, model_decoder)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    train_x, train_y, test_x, test_y = load_mnist()
    xs = test_x / 255
    xs = xs.transpose(0, 3, 1, 2)

    with torch.no_grad():
        for i in range(10):
            x = xs[i]
            
            x = np.expand_dims(x, axis=0)
            x = torch.tensor(x, dtype=torch.float).to(device)
            
            pred = model(x)

            pred = pred.view(channel, img_height, img_width)
            pred = pred.detach().cpu().numpy()
            pred -= pred.min()
            pred /= pred.max()
            pred = pred.transpose(1,2,0)
            
            _x = x.detach().cpu().numpy()[0]
            #_x = (_x + 1) / 2
            if channel == 1:
                pred = pred[..., 0]
                _x = _x[0]
                cmap = 'gray'
            else:
                _x = _x.transpose(1,2,0)
                cmap = None
                
            #print(mu, sigma)
                
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

if __name__ == "__main__":
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
    