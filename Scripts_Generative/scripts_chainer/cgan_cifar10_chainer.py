import chainer
import chainer.links as L
import chainer.functions as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

num_classes = 10
img_height, img_width = 32, 32
channel = 3

GPU = -1
    
class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__()
        base = 64
        with self.init_scope():
            self.l1 = L.Deconvolution2D(None, base * 8, ksize=int(img_height/16), stride=1, nobias=True)
            self.bn1 = L.BatchNormalization(base * 8)
            self.l2 = L.Deconvolution2D(None, base * 4, ksize=4, stride=2, pad=1,  nobias=True)
            self.bn2 = L.BatchNormalization(base * 4)
            self.l3 = L.Deconvolution2D(None, base * 2, ksize=4, stride=2, pad=1,  nobias=True)
            self.bn3 = L.BatchNormalization(base * 2)
            self.l4 = L.Deconvolution2D(None, base, ksize=4, stride=2, pad=1,  nobias=True)
            self.bn4 = L.BatchNormalization(base)
            self.l5 = L.Deconvolution2D(None, channel, ksize=4,  stride=2, pad=1)
        
    def forward(self, x, y, test=False):
        con_x = np.zeros((len(y), num_classes, 1, 1), dtype=np.float32)
        con_x[np.arange(len(y)), y] = 1
        if GPU >= 0:
            con_x = chainer.cuda.to_gpu(con_x)

        x = F.concat([x, con_x], axis=1)
            
        x = self.l1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.l4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.l5(x)
        x = F.tanh(x)

        if test:
            return x
        else:
            con_x = np.zeros((len(y), num_classes, img_height, img_width), dtype=np.float32)
            con_x[np.arange(len(y)), y] = 1
            if GPU >= 0:
                con_x = chainer.cuda.to_gpu(con_x)
            out_x = F.concat([x, con_x], axis=1)
            return out_x

    
class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        base = 64
        with self.init_scope():
            self.l1 = L.Convolution2D(None, base, ksize=5, pad=2, stride=2)
            self.l2 = L.Convolution2D(None, base * 2, ksize=5, pad=2, stride=2)
            self.l3 = L.Convolution2D(None, base * 4, ksize=5, pad=2, stride=2)
            self.l4 = L.Convolution2D(None, base * 8, ksize=5, pad=2, stride=2)
            self.l5 = L.Linear(None, 1)
        
    def forward(self, x):
        x = self.l1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l5(x)
        #x = F.sigmoid(x)
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
    gen = Generator()
    dis = Discriminator()
    gan = chainer.Sequential(gen, dis)

    if GPU >= 0:
        chainer.cuda.get_device(GPU).use()
        gen.to_gpu()
        dis.to_gpu()
        gan.to_gpu()

    opt_d = chainer.optimizers.Adam(0.0002, beta1=0.5)
    opt_d.setup(dis)
    opt_g = chainer.optimizers.Adam(0.0002, beta1=0.5)
    opt_g.setup(gen)
    
    train_x, train_y, test_x, test_y = load_cifar10()
    xs = train_x / 127.5 - 1
    xs = xs.transpose(0, 3, 1, 2)

    # training
    mb = 128
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(30000):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            mbi = mb - (len(xs) - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb


        gen.cleargrads()
        dis.cleargrads()
        gan.cleargrads()
        
        x = xs[mb_ind]
        con_x = train_y[mb_ind]
        input_noise = np.random.uniform(-1, 1, size=(mb, 100, 1, 1)).astype(np.float32)
        dt = np.array([1] * mb + [0] * mb, dtype=np.int32).reshape([mb*2, 1])
        gt = np.array([1] * mb, dtype=np.int32).reshape([mb, 1])

        _con_x = np.zeros((mb, num_classes, img_height, img_width), dtype=np.float32)
        _con_x[np.arange(mb), con_x] = 1
        x = np.hstack([x, _con_x])
        
        if GPU >= 0:
            x = chainer.cuda.to_gpu(x)
            input_noise = chainer.cuda.to_gpu(input_noise)
            dt = chainer.cuda.to_gpu(dt)
            gt = chainer.cuda.to_gpu(gt)

        g_output = gen(input_noise, con_x)

        #if GPU >= 0:
        #    g_output = chainer.cuda.to_cpu(g_output)

        X = F.concat((x, g_output), axis=0)
        y = dis(X)
        
        loss_d = F.sigmoid_cross_entropy(y, dt)
        loss_d.backward()
        opt_d.update()

        y = gan(input_noise, con_x)
        
        loss_g = F.sigmoid_cross_entropy(y, gt)
        loss_g.backward()
        opt_g.update()
        
        loss_d = loss_d.data
        loss_g = loss_g.data
        
        if GPU >= 0:
            loss_d = chainer.cuda.to_cpu(loss_d)
            loss_g = chainer.cuda.to_cpu(loss_g)


        if (i+1) % 100 == 0:
            print("iter >>", i + 1, ',G:loss >>', loss_g.item(), ', D:loss >>', loss_d.item())

    chainer.serializers.save_npz('cgan_cifar10_chainer.npz', gen)

    
# test
def test():
    g = Generator()

    if GPU >= 0:
        chainer.cuda.get_device_from_id(GPU).use()
        g.to_gpu()

    ## Load pretrained parameters
    chainer.serializers.load_npz('cgan_cifar10_chainer.npz', g)

    np.random.seed(100)
    
    labels = ["air¥nplane", "auto¥bmobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]
    

    for i in range(3):
        mb = 10
        input_noise = np.random.uniform(-1, 1, size=(mb, 100, 1, 1)).astype(np.float32)
        con_x = np.arange((num_classes), dtype=np.int)

        if GPU >= 0:
            input_noise = chainer.cuda.to_gpu(input_noise)

        g_output = g(input_noise, con_x, test=True).data

        if GPU >= 0:
            g_output = chainer.cuda.to_cpu(g_output)
            
        g_output = (g_output + 1) / 2
        g_output = g_output.transpose(0,2,3,1)

        for i in range(mb):
            gen = g_output[i]

            if channel == 1:
                gen = gen[..., 0]
                cmap = "gray"
            elif channel == 3:
                cmap = None
                
            plt.subplot(1,mb,i+1)
            plt.title(labels[i])
            plt.imshow(gen, cmap=cmap)
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
