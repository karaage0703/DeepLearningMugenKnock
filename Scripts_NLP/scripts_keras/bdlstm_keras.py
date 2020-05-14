import keras
import cv2
import numpy as np
import argparse
from glob import glob

# GPU config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
sess = tf.Session(config=config)
K.set_session(sess)

# network
from keras.models import Sequential, Model
from keras.layers import Dense, Input, SimpleRNN, LSTM, Bidirectional

n_gram = 10

_chars = "あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっー１２３４５６７８９０！？、。@#"
chars = [c for c in _chars]
num_classes = len(chars)

def Mynet():
    inputs = Input([n_gram, num_classes])
    x = Bidirectional(LSTM(128, return_sequences=False))(inputs)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x, name='model')
    
    return model

    
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
            ts.append(onehots[i+n_gram])

    xs = np.array(xs)
    ts = np.array(ts)
    
    return xs, ts


# train
def train():
    # model
    model = Mynet()

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy'])

    xs, ts = data_load()
    
    # training
    mb = 128
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(1000):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = xs[mb_ind]
        t = ts[mb_ind]

        loss, acc = model.train_on_batch(x=x, y=t)
        print("iter >>", i+1, ",loss >>", loss, ',accuracy >>', acc)

    model.save('model.h5')

# test
def test():
    # load trained model
    model = Mynet()
    model.load_weights('model.h5')

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
        
        x = np.expand_dims(np.array(x), axis=0)
        
        pred = model.predict_on_batch(x)[0]

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
