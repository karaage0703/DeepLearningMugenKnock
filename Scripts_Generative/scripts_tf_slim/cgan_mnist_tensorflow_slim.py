from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.contrib import slim

import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


num_classes = 10
img_height, img_width = 28, 28
channel = 1


def Generator(x, y, y2=None):
    in_h = int(img_height / 4)
    in_w = int(img_width / 4)
    base = 128

    x = tf.concat([x, y], axis=-1)
    
    x = slim.fully_connected(x, base * 2 * in_h * in_w, activation_fn=tf.nn.relu, normalizer_fn=lambda x: x, reuse=tf.AUTO_REUSE, scope='g_dense1')
    x = tf.reshape(x, [-1, in_h, in_w, base * 2])
    x = slim.batch_norm(x, reuse=tf.AUTO_REUSE, decay=0.9, epsilon=1e-5, scope="g_bn")

    # 1/8
    #x = slim.conv2d_transpose(x, base * 4, [5, 5], stride=[2,2], activation_fn=None, normalizer_fn=lambda x: x, reuse=tf.AUTO_REUSE, scope="g_deconv1")
    #x = tf.nn.relu(x)
    #x = slim.batch_norm(x, reuse=tf.AUTO_REUSE, decay=0.9, epsilon=1e-5, scope="g_bn1")
    # 1/4
    #x = slim.conv2d_transpose(x, base * 2, [5, 5], stride=[2,2], activation_fn=None, normalizer_fn=lambda x: x, reuse=tf.AUTO_REUSE, scope="g_deconv2")
    #x = tf.nn.relu(x)
    #x = slim.batch_norm(x, reuse=tf.AUTO_REUSE, decay=0.9, epsilon=1e-5, scope="g_bn2")
    # 1/2
    x = slim.conv2d_transpose(x, base, [5, 5], stride=[2,2], activation_fn=None, normalizer_fn=lambda x: x, reuse=tf.AUTO_REUSE,  scope="g_deconv3")
    x = tf.nn.relu(x)
    x = slim.batch_norm(x, reuse=tf.AUTO_REUSE, decay=0.9, epsilon=1e-5, scope="g_bn3")
    # 1/1
    x = slim.conv2d_transpose(x, channel, [5, 5], stride=[2,2], activation_fn=None, reuse=tf.AUTO_REUSE, scope="g_deconv4")
    #x = slim.batch_norm(x)
    x = tf.nn.tanh(x)

    if y2 is not None:
        x = tf.concat([x, y2], axis=-1)

    return x


def Discriminator(x):
    base = 64
    x = slim.conv2d(x, base, [5,5], stride=[2,2], activation_fn=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE,  scope="d_conv1")
    x = slim.conv2d(x, base * 2, [5,5], stride=[2,2], activation_fn=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE, scope="d_conv2")
    #x = slim.conv2d(x, base * 4, [5,5], stride=[2,2], activation_fn=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE, scope="d_conv3")
    #x = slim.conv2d(x, base * 8, [5,5], stride=[2,2], activation_fn=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE, scope="d_conv4")
    x = slim.flatten(x)
    x = slim.fully_connected(x, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope="d_dense")

    return x
    

    
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
    tf.reset_default_graph()

    # place holder
    X = tf.placeholder(tf.float32, [None, 100])
    X2 = tf.placeholder(tf.float32, [None, img_height, img_width, channel + num_classes])
    X_CON = tf.placeholder(tf.float32, [None, num_classes])
    X_CON2 = tf.placeholder(tf.float32, [None, img_height, img_width, num_classes])
    Y = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32)
    
    g_logits = Generator(X, X_CON, X_CON2)
    d_logits = Discriminator(X2)
    gan_logits = Discriminator(g_logits)

    tvars = tf.trainable_variables()
    
    d_preds = d_logits
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=Y))
    #loss = tf.reduce_mean(tf.square(logits - Y))
    D_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    D_vars = [var for var in tvars if 'd_' in var.name]
    D_train = D_optimizer.minimize(D_loss, var_list=D_vars)

    gan_preds = gan_logits
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits, labels=Y))
    G_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    G_vars = [var for var in tvars if 'g_' in var.name]
    G_train = G_optimizer.minimize(G_loss, var_list=G_vars)

    train_x, train_y, test_x, test_y = load_mnist()
    xs = train_x / 127.5 - 1

    # training
    mb = 64
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    #d_losses = [0]
    #g_losses = [0]
    #ites = [0]
    #fig, ax = plt.subplots(1, 1)
    #lines, = ax.plot(d_losses, g_losses)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
    
        for ite in range(10000):
            if mbi + mb > len(xs):
                mb_ind = train_ind[mbi:]
                np.random.shuffle(train_ind)
                mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
                mbi = mb - (len(xs) - mbi)
            else:
                mb_ind = train_ind[mbi: mbi+mb]
                mbi += mb

            x = xs[mb_ind]
            x_con = train_y[mb_ind]

            input_noise = np.random.uniform(-1, 1, size=(mb, 100))

            _x_con = np.zeros([mb, num_classes], dtype=np.float32)
            _x_con[np.arange(mb), x_con] = 1

            _x_con2 = np.zeros([mb, img_height, img_width, num_classes], dtype=np.float32)
            _x_con2[np.arange(mb), ..., x_con] = 1

            g_output = sess.run(g_logits, feed_dict={X: input_noise, X_CON: _x_con, X_CON2: _x_con2})

            x = np.concatenate([x, _x_con2], axis=-1)

            _X = np.concatenate([x, g_output])
            _Y = np.array([1] * mb + [0] * mb, dtype=np.float32)
            _Y = _Y[..., None]
    
            _, d_loss = sess.run([D_train, D_loss], feed_dict={X2:_X, Y:_Y})

            _Y = np.array([1] * mb, dtype=np.float32)
            _Y = _Y[..., None]
            _, g_loss = sess.run([G_train, G_loss], feed_dict={X:input_noise, X_CON: _x_con, X_CON2: _x_con2, Y: _Y})
            
            #d_losses.append(d_loss)
            #g_losses.append(g_loss)
            #ites.append(ite + 1)
            #lines.set_data(ites, d_losses)
            #ax.set_xlim((0, ite+2))
            #plt.pause(0.001)
            
            
            if (ite+1) % 100 == 0:
                print("iter >>", ite+1, ',G:loss >>', g_loss, ',D:loss >>', d_loss)

        saver = tf.train.Saver()
        saver.save(sess, './cnn.ckpt')

# test
def test():
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, 100])
    X_CON = tf.placeholder(tf.float32, [None, num_classes])

    logits = Generator(X, X_CON)
    
    np.random.seed(100)
    


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./cnn.ckpt")

        for i in range(3):
            input_noise = np.random.uniform(-1, 1, size=(10, 100))
            x_con = np.zeros([10, num_classes], dtype=np.float32)
            x_con[np.arange(10), np.arange(num_classes)] = 1
            
            g_output = sess.run(logits, feed_dict={X: input_noise, X_CON: x_con})
            g_output = (g_output + 1 ) / 2

            for i in range(10):
                gen = g_output[i]

                if channel == 1:
                    gen = gen[..., 0]
                    cmap = "gray"
                elif channel == 3:
                    cmap = None

                plt.subplot(1,10,i+1)
                plt.imshow(gen, cmap=cmap)
                plt.title(str(i))
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
