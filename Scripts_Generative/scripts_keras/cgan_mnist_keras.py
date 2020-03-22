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
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization, Reshape, UpSampling2D, LeakyReLU, Conv2DTranspose, concatenate, Lambda

num_classes = 10
img_height, img_width = 28, 28
channel = 1

from keras.regularizers import l1_l2
from keras.initializers import RandomNormal as RN, Constant



def G_model():
    inputs = Input([100, ], name="x")
    con_x = Input([num_classes, ], name="con_x")
    con_x2 = Input([img_height, img_width, num_classes], name="con_x2")
    
    #con_x = K.zeros([None, num_classes, 1, 1])
    #print(con_x.shape)
    #con_x = np.zeros([len(_con_x), num_classes, 1, 1], dtype=np.float32)
    #con_x[np.arange(len(_con_x)), _con_x] = 1

    x = concatenate([inputs, con_x], axis=-1)
    
    in_h = int(img_height / 4)
    in_w = int(img_width / 4)
    d_dim = 256
    base = 128
    x = Dense(in_h * in_w * d_dim, name='g_dense1',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = Reshape((in_h, in_w, d_dim), input_shape=(d_dim * in_h * in_w,))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_dense1_bn')(x)
    # 1/8
    #x = Conv2DTranspose(base*4, (5, 5), name='g_conv1', padding='same', strides=(2,2),
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = Activation('relu')(x)
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv1_bn')(x)
    # 1/4
    #x = Conv2DTranspose(base*2, (5, 5), name='g_conv2', padding='same', strides=(2,2),
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = Activation('relu')(x)
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv2_bn')(x)
    # 1/2
    x = Conv2DTranspose(base, (5, 5), name='g_conv3', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv3_bn')(x)
    # 1/1
    x = Conv2DTranspose(channel, (5, 5), name='g_out', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02),  bias_initializer=Constant())(x)
    x = Activation('tanh')(x)

    #con_x = np.zerns([len(_con_x), num_classes, img_height, img_width], dtype=np.float32)
    #con_x[np.arange(len(_con_x)), _con_x] = 1
    x2 = concatenate([x, con_x2], axis=-1)

    model = Model(inputs=[inputs, con_x], outputs=[x], name='G')
    gan_g_model = Model(inputs=[inputs, con_x, con_x2], outputs=[x2], name='GAN_G')
    
    return model, gan_g_model


def D_model():
    base = 32
    inputs = Input([img_height, img_width, channel + num_classes])
    x = Conv2D(base, (5, 5), padding='same', strides=(2,2), name='d_conv1',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(base*2, (5, 5), padding='same', strides=(2,2), name='d_conv2',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = Conv2D(base*4, (5, 5), padding='same', strides=(2,2), name='d_conv3',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = LeakyReLU(alpha=0.2)(x)
    #x = Conv2D(base*8, (5, 5), padding='same', strides=(2,2), name='d_conv4',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid', name='d_out',
        kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    model = Model(inputs=inputs, outputs=x, name='D')
    return model


def Combined_model(g, d):
    inputs = Input([100, ], name="x")
    con_x = Input([num_classes, ], name="con_x")
    con_x2 = Input([img_height, img_width, num_classes], name="con_x2")
    x = g(inputs=[inputs, con_x, con_x2])
    x = d(x)
    model = Model(inputs=[inputs, con_x, con_x2], outputs=[x])

    #model = Sequential()
    #model.add(g)
    #model.add(d)
    return model


import pickle
import os
import gzip
    
def load_mnist():
    dir_path = "mnist_datas"

    files = ["train-images-idx3-ubyte.gz",
             "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz",
             "t10k-labels-idx1-ubyte.gz"]

    # download mnist datas
    if not os.path.exists(dir_path):

        os.makedirs(dir_path)

        data_url = "http://yann.lecun.com/exdb/mnist/"

        for file_url in files:

            after_file = file_url.split('.')[0]
            
            if os.path.exists(dir_path + '/' + after_file):
                continue
            
            os.system("wget {}/{}".format(data_url, file_url))
            os.system("mv {} {}".format(file_url, dir_path))

        
    # load mnist data

    # load train data
    with gzip.open(dir_path + '/' + files[0], 'rb') as f:
        train_x = np.frombuffer(f.read(), np.uint8, offset=16)
        train_x = train_x.astype(np.float32)
        train_x = train_x.reshape((-1, 28, 28, 1))
        print("train images >>", train_x.shape)

    with gzip.open(dir_path + '/' + files[1], 'rb') as f:
        train_y = np.frombuffer(f.read(), np.uint8, offset=8)
        print("train labels >>", train_y.shape)

    # load test data
    with gzip.open(dir_path + '/' + files[2], 'rb') as f:
        test_x = np.frombuffer(f.read(), np.uint8, offset=16)
        test_x = test_x.astype(np.float32)
        test_x = test_x.reshape((-1, 28, 28, 1))
        print("test images >>", test_x.shape)
    
    with gzip.open(dir_path + '/' + files[3], 'rb') as f:
        test_y = np.frombuffer(f.read(), np.uint8, offset=8)
        print("test labels >>", test_y.shape)
        

    return train_x, train_y ,test_x, test_y



# train
def train():
    _, g = G_model()
    d = D_model()

    g_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    d_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    
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

    train_x, train_y, test_x, test_y = load_mnist()
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
        con_x = train_y[mb_ind]

        # Disciminator training
        
        input_noise = np.random.uniform(-1, 1, size=(mb, 100))
        _con_x = np.zeros([mb, num_classes], dtype=np.float32)
        _con_x[np.arange(mb), con_x] = 1

        _con_x2 = np.zeros([mb, img_height, img_width, num_classes], dtype=np.float32)
        _con_x2[np.arange(mb), ..., con_x] = 1
        
        g_output = g.predict(
            x={"x":input_noise, "con_x": _con_x, "con_x2": _con_x2}, verbose=0)

        x = np.concatenate([x, _con_x2], axis=-1)
        X = np.concatenate((x, g_output))
        
        Y = [1] * mb + [0] * mb
        d_loss = d.train_on_batch(X, Y)
        
        # Generator training
        #input_noise = np.random.uniform(-1, 1, size=(mb, 100))
        g_loss = gan.train_on_batch(
            x={"x":input_noise, "con_x":_con_x, "con_x2": _con_x2}, y=np.array([1] * mb))

        if (i+1) % 100 == 0:
            print("iter >>", i+1, ",g_loss >>", g_loss, ',d_loss >>', d_loss)
    
    g.save('cgan_cifar10_keras.h5')

# test
def test():
    # load trained model
    g, _ = G_model()
    g.load_weights('cgan_cifar10_keras.h5', by_name=True)

    np.random.seed(100)
    
    con_x = np.zeros([10, num_classes])
    con_x[np.arange(10), np.arange(num_classes)] = 1
    
    for i in range(3):
        input_noise = np.random.uniform(-1, 1, size=(10, 100))
        g_output = g.predict(x={"x":input_noise, "con_x": con_x}, verbose=0)
        g_output = (g_output + 1 ) / 2

        for i in range(10):
            gen = g_output[i]
            
            if channel == 1:
                gen = gen[..., 0]
                cmap = "gray"
            elif channel == 3:
                cmap = None
                
            plt.subplot(1,10,i+1)
            plt.title(str(i))
            plt.imshow(gen, cmap=cmap)
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

        
