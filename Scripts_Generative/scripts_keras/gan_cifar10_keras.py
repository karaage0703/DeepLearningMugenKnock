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
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization, Reshape, UpSampling2D, LeakyReLU

num_classes = 2
img_height, img_width = 32, 32
channel = 3

def G_model():
    inputs = Input((100,))
    base = 128
    x = Dense(base, name='g_dense1')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(base * 2, name='g_dense2')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(base * 4, name='g_dense3')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(img_height * img_width * channel, activation='tanh', name='g_out')(x)
    x = Reshape((img_height, img_width, channel))(x)
    model = Model(inputs, x, name='G')
    return model

def D_model():
    inputs = Input((img_height, img_width, channel))
    base = 512
    x = Flatten()(inputs)
    x = Dense(base * 2, name='d_dense1')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(base, name='d_dense2')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid', name='d_out')(x)
    model = Model(inputs, x, name='D')
    return model

def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
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
    g = G_model()
    d = D_model()
    gan = Combined_model(g=g, d=d)

    g_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    d_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    #g_opt = keras.optimizers.SGD(lr=0.0002, momentum=0.3, decay=1e-5)
    #d_opt = keras.optimizers.SGD(lr=0.0002, momentum=0.1, decay=1e-5)

    d.trainable = True
    for layer in d.layers:
        layer.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_opt)
    g.compile(loss='binary_crossentropy', optimizer=d_opt)
    d.trainable = False
    for layer in d.layers:
        layer.trainable = False
    gan = Combined_model(g=g, d=d)
    gan.compile(loss='binary_crossentropy', optimizer=g_opt)

    train_x, train_y, test_x, test_y = load_cifar10()
    xs = train_x / 127.5 - 1

    # training
    mb = 64
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

        input_noise = np.random.uniform(-1, 1, size=(mb, 100))
        g_output = g.predict(input_noise, verbose=0)
        X = np.concatenate((x, g_output))
        Y = [1] * mb + [0] * mb
        d_loss = d.train_on_batch(X, Y)
        # Generator training
        input_noise = np.random.uniform(-1, 1, size=(mb, 100))
        g_loss = gan.train_on_batch(input_noise, [1] * mb)

        if (i+1) % 100 == 0:
            print("iter >>", i+1, ",g_loss >>", g_loss, ',d_loss >>', d_loss)
    
    g.save('model.h5')

# test
def test():
    # load trained model
    g = G_model()
    g.load_weights('model.h5', by_name=True)

    np.random.seed(100)
    
    for i in range(3):
        input_noise = np.random.uniform(-1, 1, size=(9, 100))
        g_output = g.predict(input_noise, verbose=0)
        g_output = (g_output + 1) / 2

        for i in range(9):
            gen = g_output[i]
            plt.subplot(1,9,i+1)
            plt.imshow(gen)
            plt.axis('off')
            #plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            
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
