#%tensorflow_version 2.x
import tensorflow as tf

print(tf.__version__)

import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import RandomNormal as RN, Constant
import pickle
import os

# config
class_N = 2
img_height, img_width = 32, 32
channel = 3

# GAN config
Z_dim = 100

test_N = 10

# model path
model_path = 'DCGAN.h5'



def Generator():
    inputs = Input((Z_dim,))
    in_h = int(img_height / 16)
    in_w = int(img_width / 16)
    base = 256
    # 1/16
    x = Dense(in_h * in_w * base, name='Dense1', use_bias=False)(inputs)
    x = Reshape((in_h, in_w, base), input_shape=(base * in_h * in_w,))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='dense1_bn')(x)
    # 1/8
    x = Conv2DTranspose(base*4, (5, 5), name='tconv1', padding='same', strides=(2,2), use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='tconv1_bn')(x)
    # 1/4
    x = Conv2DTranspose(base*2, (5, 5), name='tconv2', padding='same', strides=(2,2), use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='tconv2_bn')(x)
    # 1/2
    x = Conv2DTranspose(base, (5, 5), name='tconv3', padding='same', strides=(2,2), use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='tconv3_bn')(x)
    # 1/1
    x = Conv2DTranspose(channel, (5, 5), name='tconv_out', padding='same', strides=(2,2))(x)
    x = Activation('tanh')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name='G')
    return model


def Discriminator():
    base = 32
    inputs = Input((img_height, img_width, channel))
    x = Conv2D(base, (5, 5), padding='same', strides=(2,2), name='conv1')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(base*2, (5, 5), padding='same', strides=(2,2), name='conv2')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(base*4, (5, 5), padding='same', strides=(2,2), name='conv3')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(base*8, (5, 5), padding='same', strides=(2,2), name='conv4')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1, activation=None, name='dense_out')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name='D')
    return model
    
def load_cifar10():
    path = 'drive/My Drive/Colab Notebooks/' + 'cifar-10-batches-py'

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
    # model
    G = Generator()
    D = Discriminator()

    train_x, train_y, test_x, test_y = load_cifar10()
    xs = train_x / 127.5 - 1

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #loss_fn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=Y))

    # training
    mb = 64
    mbi = 0
    train_N = len(xs)
    train_ind = np.arange(train_N)
    np.random.seed(0)

    @tf.function
    def train_iter(x, z):
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            # feed forward
            # z -> G -> Gz
            Gz = G(z, training=True)

            # x -> D -> Dx
            # z -> G -> Gz -> D -> DGz
            Dx = D(x, training=True)
            DGz = D(Gz, training=True)

            # get loss
            loss_G = loss_fn(tf.ones_like(DGz), DGz)
            loss_D_real = loss_fn(tf.ones_like(Dx), Dx)
            loss_D_fake = loss_fn(tf.zeros_like(DGz), DGz)
            loss_D = loss_D_real + loss_D_fake

        # feed back
        gradients_of_G = G_tape.gradient(loss_G, G.trainable_variables)
        gradients_of_D = D_tape.gradient(loss_D, D.trainable_variables)

        # update parameter
        G_optimizer.apply_gradients(zip(gradients_of_G, G.trainable_variables))
        D_optimizer.apply_gradients(zip(gradients_of_D, D.trainable_variables))

        return loss_G, loss_D

    #with strategy.scope():
    # optimizer
    G_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    D_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    for ite in range(10000):
        if mbi + mb > train_N:
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (train_N - mbi))]))
            mbi = mb - (train_N - mbi)
        else:
            mb_ind = train_ind[mbi : mbi + mb]
            mbi += mb

        x = xs[mb_ind]

        z = np.random.uniform(-1, 1, size=(mb, Z_dim))
        #z = tf.random.normal([mb, Z_dim])

        loss_G, loss_D = train_iter(x, z)
        
        if (ite + 1) % 100 == 0:
            print("iter :", ite+1, ', Loss_G :', loss_G.numpy(), ',Loss_D :', loss_D.numpy())

        # display generated image
        if (ite + 1) % 100 == 0:
            Gz = G(z)
            _Gz = (Gz * 127.5 + 127.5).numpy().astype(int)
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(_Gz[i])
                plt.axis('off')
            plt.show()

    # save model
    G.save_weights(model_path)

# test
def test():
    # model
    G = Generator()
    G.load_weights(model_path)

    for i in range(3):
        z = np.random.uniform(-1, 1, size=(test_N, Z_dim))
        Gz = G(z, training=False)
        Gz = Gz * 127.5 + 127.5
        Gz = Gz.numpy().astype(int)

        for i in range(test_N):
            _Gz = Gz[i]
            plt.subplot(1, test_N, i + 1)
            plt.imshow(_Gz)
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
