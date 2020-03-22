import chainer
import chainer.links as L
import chainer.functions as F
import argparse
import cv2
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 224, 224
channel = 3
GPU = -1


class InceptionModule(chainer.Chain):
    def __init__(self, f_1, f_2_1, f_2_2, f_3_1, f_3_2, f_4_2):
        super(InceptionModule, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, f_1, ksize=1, pad=0, stride=1)

            self.conv2_1 = L.Convolution2D(None, f_2_1, ksize=1, pad=0, stride=1)
            self.conv2_2 = L.Convolution2D(None, f_2_2, ksize=3, pad=1, stride=1)
            
            self.conv3_1 = L.Convolution2D(None, f_3_1, ksize=1, pad=0, stride=1)
            self.conv3_2 = L.Convolution2D(None, f_3_2, ksize=5, pad=2, stride=1)
            
            self.conv4_2 = L.Convolution2D(None, f_4_2, ksize=1, pad=0, stride=1)

    def __call__(self, x):
        x1 = F.relu(self.conv1(x))

        x2 = F.relu(self.conv2_1(x))
        x2 = F.relu(self.conv2_2(x2))

        x3 = F.relu(self.conv3_1(x))
        x3 = F.relu(self.conv3_2(x3))

        x4 = F.max_pooling_2d(x, ksize=3, pad=1, stride=1)
        x4 = F.relu(self.conv4_2(x4))

        x = F.concat([x1, x2, x3, x4], axis=1)

        return x


class GoogLeNetv1(chainer.Chain):
    def __init__(self):
        self.train = train
        super(GoogLeNetv1, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, ksize=7, pad=0, stride=1)
            self.conv2_1 = L.Convolution2D(None, 64, ksize=1, pad=0, stride=1)
            self.conv2_2 = L.Convolution2D(None, 192, ksize=3, pad=1, stride=1)

            self.inception3a = InceptionModule(64, 96, 128, 16, 32, 32)
            self.inception3b = InceptionModule(128, 128, 192, 32, 96, 64)

            self.inception4a = InceptionModule(192, 96, 208, 16, 48, 64)
            self.inception4b = InceptionModule(160, 112, 224, 24, 64, 64)
            self.inception4c = InceptionModule(128, 128, 256, 24, 64, 64)
            self.inception4d = InceptionModule(112, 144, 288, 32, 64, 64)
            self.inception4e = InceptionModule(256, 160, 320, 32, 128 ,128)

            self.inception5a = InceptionModule(256, 160, 320, 32, 128, 128)
            self.inception5b = InceptionModule(384, 192, 384, 48, 128, 128)

            self.linear = L.Linear(None, num_classes)

            self.aux1_conv1 = L.Convolution2D(None, 128, ksize=1, pad=0, stride=1)
            self.aux1_linear1 = L.Linear(None, 1024)
            self.aux1_linear2 = L.Linear(None, num_classes)

            self.aux2_conv1 = L.Convolution2D(None, 128, ksize=1, pad=0, stride=1)
            self.aux2_linear1 = L.Linear(None, 1024)
            self.aux2_linear2 = L.Linear(None, num_classes)
            

    def __call__(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pooling_2d(x, ksize=3, pad=1, stride=2)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pooling_2d(x, ksize=3, pad=1, stride=2)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = F.max_pooling_2d(x, ksize=3, pad=1, stride=2)

        x = self.inception4a(x)

        x_aux1 = F.average_pooling_2d(x, ksize=5, pad=2, stride=1)
        x_aux1 = F.relu(self.aux1_conv1(x_aux1))
        x_aux1 = F.relu(self.aux1_linear1(x_aux1))
        x_aux1 = F.dropout(x_aux1, ratio=0.7)
        x_aux1 = self.aux1_linear2(x_aux1)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        x_aux2 = F.average_pooling_2d(x, ksize=5, pad=2, stride=1)
        x_aux2 = F.relu(self.aux2_conv1(x_aux2))
        x_aux2 = F.relu(self.aux2_linear1(x_aux2))
        x_aux2 = F.dropout(x_aux2, ratio=0.7)
        x_aux2 = self.aux2_linear2(x_aux2)

        x = self.inception4e(x)
        x = F.max_pooling_2d(x, ksize=3, pad=1, stride=2)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = F.average_pooling_2d(x, ksize=7, pad=0, stride=1)
        x = self.linear(x)
        
        return x, x_aux1, x_aux2


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
    model = GoogLeNetv1()

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

        y, y_aux1, y_aux2 = model(x)

        loss = F.softmax_cross_entropy(y, t)
        loss_aux1 = F.softmax_cross_entropy(y_aux1, t)
        loss_aux2 = F.softmax_cross_entropy(y_aux2, t)

        loss = loss_aux1 + loss_aux2 + loss
        
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
            print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', accu)

    chainer.serializers.save_npz('cnn.npz', model)

    
# test
def test():
    model = GoogLeNetv1()

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
            
        pred, _, _ = model(x)
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
