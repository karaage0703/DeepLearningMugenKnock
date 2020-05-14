import chainer
import chainer.links as L
import chainer.functions as F
import argparse
import cv2
import numpy as np
from glob import glob
import os

GPU = -1
n_gram = 10

_chars = "あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっー１２３４５６７８９０！？、。@#"
chars = [c for c in _chars]
num_classes = len(chars)

class Mynet(chainer.Chain):
    def __init__(self, train=True):
        self.train = train
        super(Mynet, self).__init__()
        with self.init_scope():
            self.h = L.LSTM(None, 64)
            self.out = L.Linear(None, num_classes)
        
    def forward(self, x):
        x = self.h(x)
        x = self.out(x)
        return x
    
def data_load():
    fname = 'sandwitchman.txt'
    xs = []
    ts = []
    txt = ''
    for _ in range(n_gram):
        txt += '@'
    onehots = []
    
    with open(fname, 'r') as f:
        for l in f.readlines():
            txt += l.strip() + '#'
        txt = txt[:-1] + '@'

        for c in txt:
            onehot = [0 for _ in range(num_classes)]
            onehot[chars.index(c)] = 1
            onehots.append(onehot)
        
        for i in range(len(txt) - n_gram - 1):
            xs.append(onehots[i:i+n_gram])
            ts.append(chars.index(txt[i+n_gram]))

    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)
    
    return xs, ts


# train
def train():
    # model
    model = Mynet(train=True)

    if GPU >= 0:
        chainer.cuda.get_device(GPU).use()
        model.to_gpu()
    
    opt = chainer.optimizers.Adam(0.01)
    opt.setup(model)
    #opt.add_hook(chainer.optimizer.WeightDecay(0.0005))

    xs, ts = data_load()
    
    # training
    mb = 128
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(200):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
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

    def decode(x):
        return chars[x.argmax()]
    
    gens = ''

    for _ in range(n_gram):
        gens += '@'

    pred = 0
    count = 0
        
    while pred != '@' and count < 1000:
        in_txt = gens[-n_gram:]
        x = []
        for _in in in_txt:
            _x = [0 for _ in range(num_classes)]
            _x[chars.index(_in)] = 1
            x.append(_x)
        
        x = np.expand_dims(np.array(x, dtype=np.float32), axis=0)

        if GPU >= 0:
            x = chainer.cuda.to_gpu(x)
            
        pred = model(x).data
        pred = F.softmax(pred)

        if GPU >= 0:
            pred = chainer.cuda.to_cpu(pred)
                
        pred = pred[0].data

        # sample random from probability distribution
        ind = np.random.choice(num_classes, 1, p=pred)
        
        pred = chars[ind[0]]
        gens += pred
        count += 1

    # pose process
    gens = gens.replace('#', os.linesep).replace('@', '')
        
    print('--\ngenerated')
    print(gens)
    

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
