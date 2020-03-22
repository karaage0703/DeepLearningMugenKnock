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
out_height, out_width = 64, 64
channel = 3

GPU = -1
    
class Mynet(chainer.Chain):
    def __init__(self, train=False):
        self.train = train
        
        super(Mynet, self).__init__()
        with self.init_scope():
            self.dec1 = L.Convolution2D(None, 32, ksize=3, pad=1, stride=1)
            self.dec2 = L.Convolution2D(None, 16, ksize=3, pad=1, stride=1)
            self.enc2 = L.Deconvolution2D(None, 32, ksize=2, stride=2)
            self.enc1 = L.Deconvolution2D(None, channel, ksize=2, stride=2)
        
    def forward(self, x):
        x = self.dec1(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        x = self.dec2(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        x = self.enc2(x)
        x = self.enc1(x)
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
    model = Mynet(train=True)

    if GPU >= 0:
        chainer.cuda.get_device(GPU).use()
        model.to_gpu()
    
    opt = chainer.optimizers.Adam(0.001)
    opt.setup(model)
    #opt.add_hook(chainer.optimizer.WeightDecay(0.0005))

    xs, paths = data_load('../Dataset/train/images/', hf=True, vf=True, rot=1)

    # training
    mb = 64
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(300):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            mbi = mb - (len(xs) - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = xs[mb_ind]
        t = x.copy()
            
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

    xs, paths = data_load('../Dataset/test/images/')

    for i in range(len(paths)):
        x = xs[i]
        path = paths[i]
        
        x = np.expand_dims(x, axis=0)
        if GPU >= 0:
            x = chainer.cuda.to_gpu(x)

        pred = model(x)
        
        if GPU >= 0:
            pred = chainer.cuda.to_cpu(pred)
                
        pred = pred[0].data
        pred = (pred + 1) / 2
        pred = pred.transpose([1,2,0])
        pred -= pred.min()
        pred /= pred.max()
        
        x = chainer.cuda.to_cpu(x) if GPU >= 0 else x
        x = (x + 1) / 2
        
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
