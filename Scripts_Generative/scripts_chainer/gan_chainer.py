import chainer
import chainer.links as L
import chainer.functions as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

num_classes = 2
img_height, img_width = 64, 64
channel = 3

GPU = -1
    
class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 128)
            self.l2 = L.Linear(None, 256)
            self.l3 = L.Linear(None, 512)
            self.l4 = L.Linear(None, img_height * img_width * channel)
        
    def forward(self, x):
        x = self.l1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l4(x)
        x = F.tanh(x)
        return x

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 512)
            self.l2 = L.Linear(None, 256)
            self.l3 = L.Linear(None, 1)
        
    def forward(self, x):
        x = self.l1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l3(x)
        #x = F.sigmoid(x)
        return x  
    
    
CLS = {'akahara': [0,0,128],
       'madara': [0,128,0]}
    
# get train data
def data_load(path, hf=False, vf=False, rot=False):
    xs = []
    paths = []
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            if channel == 1:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x = x / 127.5 - 1.
            if channel == 3:
                x = x[..., ::-1]
            xs.append(x)
            
            paths.append(path)

            if hf:
                xs.append(x[:, ::-1])
                paths.append(path)

            if vf:
                xs.append(x[::-1])
                paths.append(path)

            if hf and vf:
                xs.append(x[::-1, ::-1])
                paths.append(path)

            if rot != False:
                angle = 0
                scale = 1
                while angle < 360:
                    angle += rot
                    if channel == 1:
                        _h, _w = x.shape
                        max_side = max(_h, _w)
                        tmp = np.zeros([max_side, max_side])
                    else:
                        _h, _w, _c = x.shape
                        max_side = max(_h, _w)
                        tmp = np.zeros([max_side, max_side, _c])
                    tx = int((max_side - _w) / 2)
                    ty = int((max_side - _h) / 2)
                    tmp[ty: ty+_h, tx: tx+_w] = x.copy()
                    M = cv2.getRotationMatrix2D((max_side/2, max_side/2), angle, scale)
                    _x = cv2.warpAffine(tmp, M, (max_side, max_side))
                    _x = _x[tx:tx+_w, ty:ty+_h]
                    xs.append(_x)
                    paths.append(path)

    xs = np.array(xs)

    if channel == 1:
        xs = np.expand_dims(xs, axis=-1)
    xs = xs.transpose(0,3,1,2)

    return xs, paths


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

    opt_d = chainer.optimizers.Adam(0.0002)
    opt_d.setup(dis)
    opt_g = chainer.optimizers.Adam(0.0002)
    opt_g.setup(gen)
    
    xs, paths = data_load('../Dataset/train/images/', hf=True, vf=True, rot=1)

    # training
    mb = 64
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    print('A')
    
    for ite in range(5000):
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
        
        x = xs[mb_ind].reshape([mb, -1])
        input_noise = np.random.uniform(-1, 1, size=(mb, 100)).astype(np.float32)
        dt = np.array([1] * mb + [0] * mb, dtype=np.int32).reshape([mb*2, 1])
        gt = np.array([1] * mb, dtype=np.int32).reshape([mb, 1])
            
        if GPU >= 0:
            x = chainer.cuda.to_gpu(x)
            input_noise = chainer.cuda.to_gpu(input_noise)
            dt = chainer.cuda.to_gpu(dt)
            gt = chainer.cuda.to_gpu(gt)

        g_output = gen(input_noise)

        #if GPU >= 0:
        #    g_output = chainer.cuda.to_cpu(g_output)

        X = F.concat((x, g_output), axis=0)
        y = dis(X)
        
        loss_d = F.sigmoid_cross_entropy(y, dt)
        loss_d.backward()
        opt_d.update()

        y = gan(input_noise)
        
        loss_g = F.sigmoid_cross_entropy(y, gt)
        loss_g.backward()
        opt_g.update()
        
        loss_d = loss_d.data
        loss_g = loss_g.data
        
        if GPU >= 0:
            loss_d = chainer.cuda.to_cpu(loss_d)
            loss_g = chainer.cuda.to_cpu(loss_g)


        if ite % 500 == 0:
            print("iter >>", ite + 1, ',G:loss >>', loss_g.item(), ', D:loss >>', loss_d.item())

    chainer.serializers.save_npz('cnn.npz', gen)

# test
def test():
    gen = Generator()

    if GPU >= 0:
        chainer.cuda.get_device_from_id(GPU).use()
        gen.to_gpu()

    ## Load pretrained parameters
    chainer.serializers.load_npz('cnn.npz', gen)

    np.random.seed(100)
    
    for i in range(3):
        mb = 10
        input_noise = np.random.uniform(-1, 1, size=(mb, 100)).astype(np.float32)

        if GPU >= 0:
            input_noise = chainer.cuda.to_gpu(input_noise)

        g_output = gen(input_noise).data

        if GPU >= 0:
            g_output = chainer.cuda.to_cpu(g_output)
            
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
