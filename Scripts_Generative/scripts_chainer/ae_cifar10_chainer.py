import chainer
import chainer.links as L
import chainer.functions as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

num_classes = 2
img_height, img_width = 32, 32
out_height, out_width = 32, 32
channel = 3

GPU = -1

    
class Mynet(chainer.Chain):
    def __init__(self, train=False):
        self.train = train
        base = 128
        
        super(Mynet, self).__init__()
        with self.init_scope():
            self.dec1 = L.Linear(None, base)
            self.enc1 = L.Linear(None, out_height * out_width * channel)
        
    def forward(self, x):
        x = self.dec1(x)
        x = self.enc1(x)
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
    model = Mynet(train=True)

    if GPU >= 0:
        chainer.cuda.get_device(GPU).use()
        model.to_gpu()
    
    opt = chainer.optimizers.Adam(0.001)
    opt.setup(model)
    #opt.add_hook(chainer.optimizer.WeightDecay(0.0005))

    train_x, train_y, test_x, test_y = load_cifar10()
    xs = train_x / 255
    xs = xs.transpose(0, 3, 1, 2)

    # training
    mb = 512
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(5000):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            mbi = mb - (len(xs) - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = xs[mb_ind]
        t = x.copy().reshape([mb, -1])
            
        if GPU >= 0:
            x = chainer.cuda.to_gpu(x)
            t = chainer.cuda.to_gpu(t)
        #else:
        #    x = chainer.Variable(x)
        #    t = chainer.Variable(t)

        y = model(x)

        loss = F.mean_squared_error(y, t)

        model.cleargrads()
        loss.backward()
        opt.update()

        loss = loss.data
        #accu = accu.data
        if GPU >= 0:
            loss = chainer.cuda.to_cpu(loss)
        
        print("iter >>", i+1, ',loss >>', loss.item())

    chainer.serializers.save_npz('cnn.npz', model)

# test
def test():
    model = Mynet(train=False)

    if GPU >= 0:
        chainer.cuda.get_device_from_id(GPU).use()
        model.to_gpu()

    ## Load pretrained parameters
    chainer.serializers.load_npz('cnn.npz', model)

    train_x, train_y, test_x, test_y = load_cifar10()
    xs = test_x / 255
    xs = xs.transpose(0, 3, 1, 2)

    for i in range(10):
        x = xs[i]
        path = paths[i]
        
        x = np.expand_dims(x, axis=0)
        if GPU >= 0:
            x = chainer.cuda.to_gpu(x)

        pred = model(x).data
        
        if GPU >= 0:
            pred = chainer.cuda.to_cpu(pred)
                
        pred = pred[0]
        #pred = (pred + 1) / 2
        pred = pred.reshape([channel, out_height, out_width])
        pred = pred.transpose([1,2,0])
        pred -= pred.min()
        pred /= pred.max()
        
        x = chainer.cuda.to_cpu(x) if GPU >= 0 else x
        #x = (x + 1) / 2
        
        if channel == 1:
            pred = pred[..., 0]
            _x = x[0, 0]
            cmap = 'gray'
        else:
            _x = x[0].transpose(1,2,0)
            cmap = None
        
        plt.subplot(1,2,1)
        plt.title("input")
        plt.imshow(_x, cmap=cmap)
        plt.subplot(1,2,2)
        plt.title("predicted")
        plt.imshow(pred, cmap=cmap)
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
