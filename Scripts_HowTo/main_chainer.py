import chainer
import chainer.links as L
import chainer.functions as F
import argparse
import cv2
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 64, 64
GPU = -1

class Mynet(chainer.Chain):
    def __init__(self, train=True):
        self.train = train
        super(Mynet, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=False)
            self.bn1_1 = L.BatchNormalization(32)
            self.conv1_2 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=False)
            self.bn1_2 = L.BatchNormalization(32)
            self.conv2_1 = L.Convolution2D(None, 64, ksize=3, pad=1, nobias=False)
            self.bn2_1 = L.BatchNormalization(64)
            self.conv2_2 = L.Convolution2D(None, 64, ksize=3, pad=1, nobias=False)
            self.bn2_2 = L.BatchNormalization(64)
            self.conv3_1 = L.Convolution2D(None, 128, ksize=3, pad=1, nobias=False)
            self.bn3_1 = L.BatchNormalization(128)
            self.conv3_2 = L.Convolution2D(None, 128, ksize=3, pad=1, nobias=False)
            self.bn3_2 = L.BatchNormalization(128)
            self.conv4_1 = L.Convolution2D(None, 256, ksize=3, pad=1, nobias=False)
            self.bn4_1 = L.BatchNormalization(256)
            self.conv4_2 = L.Convolution2D(None, 256, ksize=3, pad=1, nobias=False)
            self.bn4_2 = L.BatchNormalization(256)
            self.fc1 = L.Linear(None, 512, nobias=False)
            self.fc2 = L.Linear(None, 512, nobias=False)
            self.fc_out = L.Linear(None, num_classes, nobias=False)

    def __call__(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.bn1_1(x)
        x = F.relu(self.conv1_2(x))
        x = self.bn1_2(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        x = F.relu(self.conv2_1(x))
        x = self.bn2_1(x)
        x = F.relu(self.conv2_2(x))
        x = self.bn2_2(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        x = F.relu(self.conv3_1(x))
        x = self.bn3_1(x)
        x = F.relu(self.conv3_2(x))
        x = self.bn3_2(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        x = F.relu(self.conv4_1(x))
        x = self.bn4_1(x)
        x = F.relu(self.conv4_2(x))
        x = self.bn4_2(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        x = F.relu(self.fc1(x))
        if self.train:
            x = F.dropout(x, ratio=0.5)
        x = F.relu(self.fc2(x))
        if self.train:
            x = F.dropout(x, ratio=0.5)
        x = self.fc_out(x)
        return x


# get train data
def data_load(dir_path):
    xs = np.ndarray((0, img_height, img_width, 3), dtype=np.float32)
    ts = np.ndarray((0), dtype=np.int32)
    paths = []
    
    for dir_path in glob(dir_path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs = np.r_[xs, x[None, ...]]

            t = np.zeros((1))
            if 'akahara' in path:
                t = np.array((0))
            elif 'madara' in path:
                t = np.array((1))
            ts = np.r_[ts, t]

            paths.append(path)

    xs = xs.transpose(0,3,1,2)

    return xs, ts, paths


# train
def train():
    # model
    model = Mynet(train=True)

    if GPU >= 0:
        chainer.cuda.get_device(GPU).use()
        model.to_gpu()
    
    opt = chainer.optimizers.MomentumSGD(0.001)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.WeightDecay(0.0005))

    xs, ts, _ = data_load('../Dataset/train/images/')

    # training
    mb = 8
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(100):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            mbi = mb - (len(xs) - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = xs[mb_ind]
        t = ts[mb_ind]
            
        if GPU >= 0:
            x = chainer.cuda.to_gpu(x)
            t = chainer.cuda.to_gpu(t)
        #else:
        #    x = chainer.Variable(x)
        #    t = chainer.Variable(t)

        y = model(x)

        loss = F.softmax_cross_entropy(y, t)
        accu = F.accuracy(y, t)

        model.cleargrads()
        loss.backward()
        opt.update()

        loss = loss.data
        accu = accu.data
        if GPU >= 0:
            loss = chainer.cuda.to_cpu(loss)
            accu = chainer.cuda.to_cpu(accu)
        
        print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', accu)

    chainer.serializers.save_npz('cnn.npz', model)

# test
def test():
    model = Mynet(train=False)

    if GPU >= 0:
        chainer.cuda.get_device_from_id(cf.GPU).use()
        model.to_gpu()

    ## Load pretrained parameters
    chainer.serializers.load_npz('cnn.npz', model)

    xs, ts, paths = data_load('../Dataset/test/images/')

    for i in range(len(paths)):
        x = xs[i]
        t = ts[i]
        path = paths[i]
        x = np.expand_dims(x, axis=0)
        
        if GPU >= 0:
            x = chainer.cuda.to_gpu(x)
            
        pred = model(x).data
        pred = F.softmax(pred)

        if GPU >= 0:
            pred = chainer.cuda.to_cpu(pred)
                
        pred = pred[0].data
                
        print("in {}, predicted probabilities >> {}".format(path, pred))
    

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
