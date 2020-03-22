import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt

np.random.seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class FullyConnectedLayer():
    def __init__(self, in_n, out_n, use_bias=True, activation=None):
        self.w = np.random.normal(0, 1, [in_n, out_n])
        if use_bias:
            self.b = np.random.normal(0, 1, [out_n])
        else:
            self.b = None
        if activation is not None:
            self.activation = activation
        else:
            self.activation = None

    def set_lr(self, lr=0.1):
        self.lr = lr

    def forward(self, feature_in):
        self.x_in = feature_in
        x = np.dot(feature_in, self.w)
        
        if self.b is not None:
            x += self.b
            
        if self.activation is not None:
            x = self.activation(x)
        self.x_out = x
        
        return x

    
    def backward(self, w_pro, grad_pro):
        grad = np.dot(grad_pro, w_pro.T)
        if self.activation is sigmoid:
            grad *= (self.x_out * (1 - self.x_out))
        grad_w = np.dot(self.x_in.T, grad)
        self.w -= self.lr * grad_w

        if self.b is not None:
            grad_b = np.dot(np.ones([grad.shape[0]]), grad)
            self.b -= self.lr * grad_b

        return grad

    
class Model():
    def __init__(self, *args, lr=0.1):
        self.layers = args
        for l in self.layers:
            l.set_lr(lr=lr)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.output = x
        
        return x

    def backward(self, t):
        En = (self.output - t) * self.output * (1 - self.output)
        grad_pro = En
        w_pro = np.eye(En.shape[-1])
        
        for i, layer in enumerate(self.layers[::-1]):
            grad_pro = layer.backward(w_pro=w_pro, grad_pro=grad_pro)
            w_pro = layer.w


    def loss(self, t):
        Loss = np.sum((self.output - t) ** 2) / 2 / t.shape[0]
        return Loss
    

num_classes = 2
img_height, img_width = 64, 64

CLS = ['akahara', 'madara']

# get train data
def data_load(path, hf=False, vf=False, rot=None):
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

            if rot is not None:
                angle = rot
                scale = 1

                # show
                a_num = 360 // rot
                w_num = np.ceil(np.sqrt(a_num))
                h_num = np.ceil(a_num / w_num)
                count = 1
                
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
                    xs.append(x)
                    ts.append(t)
                    paths.append(path)
                    angle += rot

    ts = [[t] for t in ts]
                    
    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)
    
    xs = xs.transpose(0,3,1,2)

    return xs, ts, paths



model = Model(FullyConnectedLayer(in_n=img_height * img_width * 3, out_n=64, activation=sigmoid),
              FullyConnectedLayer(in_n=64, out_n=32, activation=sigmoid),
              FullyConnectedLayer(in_n=32, out_n=1, activation=sigmoid), lr=0.1)


xs, ts, paths = data_load("../Dataset/train/images/", hf=True, vf=True, rot=1)

mb = 64
mbi = 0
train_ind = np.arange(len(xs))
np.random.shuffle(train_ind)

for ite in range(1000):
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

    x = x.reshape(mb, -1)

    model.forward(x)
    model.backward(t)
    loss = model.loss(t)

    if ite % 50 == 0:
        print("ite:", ite+1, "Loss >>", loss)
    

# test
xs, ts, paths = data_load("../Dataset/test/images/")

for i in range(len(xs)):
    x = xs[i]
    x = x.reshape(1, -1)
    out = model.forward(x)
    print("in >>", paths[i], ", out >>", out)
    
