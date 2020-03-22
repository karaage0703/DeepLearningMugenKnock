import keras
import cv2
import numpy as np
import argparse
from glob import glob
import matplotlib.pyplot as plt

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
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization, Reshape

num_classes = 2
img_height, img_width = 32, 32
out_height, out_width = 32, 32
channel = 3


def Mynet(train=False):
    inputs = Input((img_height, img_width, channel), name='in')
    x = Conv2D(32, (3, 3), padding='same', strides=1, name='enc1')(inputs)
    x = MaxPooling2D((2,2), 2)(x)
    x = Conv2D(16, (3, 3), padding='same', strides=1, name='enc2')(x)
    x = MaxPooling2D((2,2), 2)(x)
    x = keras.layers.Conv2DTranspose(32, (2,2), strides=2, padding='same', name='dec2')(x)
    out = keras.layers.Conv2DTranspose(channel, (2,2), strides=2, padding='same', name='out')(x)
    
    model = Model(inputs=inputs, outputs=out, name='model')
    return model


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
    model = Mynet(train=True)

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        loss={'out': 'mse'},
        optimizer=keras.optimizers.Adam(lr=0.001),#"adam", #keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=False),
        loss_weights={'out': 1},
        metrics=['accuracy'])

    train_x, train_y, test_x, test_y = load_cifar10()
    xs = train_x / 255

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
        #t = x.copy().reshape([mb, -1])

        loss, acc = model.train_on_batch(x={'in':x}, y={'out':x})

        if (i+1) % 100 == 0:
            print("iter >>", i+1, ",loss >>", loss, ',accuracy >>', acc)

    model.save('model.h5')

# test
def test():
    # load trained model
    model = Mynet(train=False)
    model.load_weights('model.h5')

    train_x, train_y, test_x, test_y = load_cifar10()
    xs = test_x / 255

    for i in range(10):
        x = xs[i]
        
        x = np.expand_dims(x, axis=0)
        
        pred = model.predict_on_batch(x={'in': x})[0]
        pred -= pred.min()
        pred /= pred.max()

        if channel == 1:
            pred = pred[..., 0]
            _x = x[0, ..., 0]
            #_x = (x[0, ..., 0] + 1) / 2
            cmap = 'gray'
        else:
            _x = x[0]
            #_x = (x[0] + 1) / 2
            cmap = None
            
        plt.subplot(1,2,1)
        plt.title("input")
        plt.imshow(_x, cmap=cmap)
        plt.subplot(1,2,2)
        plt.title("predicted")
        plt.imshow(pred, cmap=cmap)
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
