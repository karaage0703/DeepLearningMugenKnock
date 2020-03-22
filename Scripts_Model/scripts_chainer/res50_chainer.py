import chainer
import chainer.links as L
import chainer.functions as F
import argparse
import cv2
import numpy as np
from glob import glob
import copy

num_classes = 2
img_height, img_width = 224, 224
channel = 3
GPU = -1


class ResBlock(chainer.Chain):
    def __init__(self, in_f, f_1, out_f, stride=1):
        super(ResBlock, self).__init__()

        self.stride = stride
        self.fit_dim = False
        
        with self.init_scope():
            self.block = chainer.Sequential(
                L.Convolution2D(None, f_1, ksize=1, pad=0, stride=stride),
                L.BatchNormalization(f_1),
                F.relu,
                L.Convolution2D(None, f_1, ksize=3, pad=1, stride=1),
                L.BatchNormalization(f_1),
                F.relu,
                L.Convolution2D(None, out_f, ksize=3, pad=1, stride=1),
                L.BatchNormalization(out_f),
                F.relu
                )

            if in_f != out_f:
                self.fit_conv = L.Convolution2D(None, out_f, ksize=1, pad=0, stride=1)
                self.fit_bn = L.BatchNormalization(out_f)
                self.fit_dim = True
                

    def __call__(self, x):
        res_x = self.block(x)

        if self.fit_dim:
            x = self.fit_conv(x)
            x = self.fit_bn(x)
            x = F.relu(x)

        if self.stride == 2:
            x = F.max_pooling_2d(x, ksize=2, pad=0, stride=self.stride)

        x = F.add(res_x, x)
        x = F.relu(x)
        
        return x


class Res50(chainer.Chain):
    def __init__(self):
        self.train = train
        super(Res50, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, ksize=7, pad=3, stride=2)
            self.bn1 = L.BatchNormalization(64)

            self.resblock2_1 = ResBlock(64, 64, 256)
            self.resblock2_2 = ResBlock(256, 64, 256)
            self.resblock2_3 = ResBlock(256, 64, 256)

            self.resblock3_1 = ResBlock(256, 128, 512, stride=2)
            self.resblock3_2 = ResBlock(512, 128, 512)
            self.resblock3_3 = ResBlock(512, 128, 512)
            self.resblock3_4 = ResBlock(512, 128, 512)

            self.resblock4_1 = ResBlock(512, 256, 1024, stride=2)
            self.resblock4_2 = ResBlock(1024, 256, 1024)
            self.resblock4_3 = ResBlock(1024, 256, 1024)
            self.resblock4_4 = ResBlock(1024, 256, 1024)
            self.resblock4_5 = ResBlock(1024, 256, 1024)
            self.resblock4_6 = ResBlock(1024, 256, 1024)
            
            self.resblock5_1 = ResBlock(1024, 512, 2048, stride=2)
            self.resblock5_2 = ResBlock(2048, 512, 2048)
            self.resblock5_3 = ResBlock(2048, 512, 2048)  

            self.linear = L.Linear(None, num_classes)
            

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pooling_2d(x, ksize=3, pad=1, stride=2)
        
        x = self.resblock2_1(x)
        x = self.resblock2_2(x)
        x = self.resblock2_3(x)

        x = self.resblock3_1(x)
        x = self.resblock3_2(x)
        x = self.resblock3_3(x)
        x = self.resblock3_4(x)

        x = self.resblock4_1(x)
        x = self.resblock4_2(x)
        x = self.resblock4_3(x)
        x = self.resblock4_4(x)
        x = self.resblock4_5(x)
        x = self.resblock4_6(x)

        x = self.resblock5_1(x)
        x = self.resblock5_2(x)
        x = self.resblock5_3(x)

        x = F.average_pooling_2d(x, ksize=[img_height//32, img_width//32], pad=0, stride=1)
        x = self.linear(x)
        
        return x


CLS = ['akahara', 'madara']


# get train data
def data_load(path, hf=False, vf=False, rot=False):
    xs = []
    ts = []
    paths = []
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            x = x[..., ::-1]
            xs.append(x)

            for i, cls in enumerate(CLS):
                if cls in path:
                    t = i
            
            ts.append(t)

            paths.append(path)

            if hf:
                xs.append(x[:, ::-1])
                ts.append(t)
                paths.append(path)

            if vf:
                xs.append(x[::-1])
                ts.append(t)
                paths.append(path)

            if hf and vf:
                xs.append(x[::-1, ::-1])
                ts.append(t)
                paths.append(path)

            if rot != False:
                angle = rot
                scale = 1

                # show
                a_num = 360 // rot
                w_num = np.ceil(np.sqrt(a_num))
                h_num = np.ceil(a_num / w_num)
                count = 1
                #plt.subplot(h_num, w_num, count)
                #plt.axis('off')
                #plt.imshow(x)
                #plt.title("angle=0")
                
                while angle < 360:
                    _h, _w, _c = x.shape
                    max_side = max(_h, _w)
                    tmp = np.zeros((max_side, max_side, _c))
                    tx = int((max_side - _w) / 2)
                    ty = int((max_side - _h) / 2)
                    tmp[ty: ty+_h, tx: tx+_w] = x.copy()
                    M = cv2.getRotationMatrix2D((max_side/2, max_side/2), angle, scale)
                    _x = cv2.warpAffine(tmp, M, (max_side, max_side))
                    _x = _x[tx:tx+_w, ty:ty+_h]
                    xs.append(_x)
                    ts.append(t)
                    paths.append(path)

                    # show
                    #count += 1
                    #plt.subplot(h_num, w_num, count)
                    #plt.imshow(_x)
                    #plt.axis('off')
                    #plt.title("angle={}".format(angle))

                    angle += rot
                #plt.show()


    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)
    
    xs = xs.transpose(0,3,1,2)

    return xs, ts, paths



# train
def train():
    # model
    model = Res50()

    if GPU >= 0:
        chainer.cuda.get_device(GPU).use()
        model.to_gpu()
    
    opt = chainer.optimizers.MomentumSGD(0.001, momentum=0.9)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.WeightDecay(0.0005))

    xs, ts, _ = data_load('../Dataset/train/images/', hf=True, vf=True, rot=1)

    # training
    mb = 16
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(1000):
        if mbi + mb > len(xs):
            mb_ind = copy.copy(train_ind)[mbi:]
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

        if (i+1) % 10 == 0:
            print("iter >>", i+1, ', loss >>', loss.item(), ', accuracy >>', accu)

    chainer.serializers.save_npz('cnn.npz', model)

    
# test
def test():
    model = Res50()

    if GPU >= 0:
        chainer.cuda.get_device_from_id(GPU).use()
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
            
        pred = model(x)
        pred = F.softmax(pred)
                
        pred = pred[0].data

        if GPU >= 0:
            pred = chainer.cuda.to_cpu(pred)
                
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
