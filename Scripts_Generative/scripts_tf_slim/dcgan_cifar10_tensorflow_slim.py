import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.contrib import slim

import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


num_classes = 2
img_height, img_width = 32, 32
channel = 3


def Generator(x):
    in_h = int(img_height / 16)
    in_w = int(img_width / 16)
    base = 128
    
    x = slim.fully_connected(x, base * 4 * in_h * in_w, activation_fn=tf.nn.relu, normalizer_fn=lambda x: x, reuse=tf.AUTO_REUSE, scope='g_dense1')
    x = tf.reshape(x, [-1, in_h, in_w, base * 4])
    x = slim.batch_norm(x, reuse=tf.AUTO_REUSE, decay=0.9, epsilon=1e-5, scope="g_bn")

    # 1/8
    x = slim.conv2d_transpose(x, base * 4, [5, 5], stride=[2,2], activation_fn=None, normalizer_fn=lambda x: x, reuse=tf.AUTO_REUSE, scope="g_deconv1")
    x = tf.nn.relu(x)
    x = slim.batch_norm(x, reuse=tf.AUTO_REUSE, decay=0.9, epsilon=1e-5, scope="g_bn1")
    # 1/4
    x = slim.conv2d_transpose(x, base * 2, [5, 5], stride=[2,2], activation_fn=None, normalizer_fn=lambda x: x, reuse=tf.AUTO_REUSE, scope="g_deconv2")
    x = tf.nn.relu(x)
    x = slim.batch_norm(x, reuse=tf.AUTO_REUSE, decay=0.9, epsilon=1e-5, scope="g_bn2")
    # 1/2
    x = slim.conv2d_transpose(x, base, [5, 5], stride=[2,2], activation_fn=None, normalizer_fn=lambda x: x, reuse=tf.AUTO_REUSE,  scope="g_deconv3")
    x = tf.nn.relu(x)
    x = slim.batch_norm(x, reuse=tf.AUTO_REUSE, decay=0.9, epsilon=1e-5, scope="g_bn3")
    # 1/1
    x = slim.conv2d_transpose(x, channel, [5, 5], stride=[2,2], activation_fn=None, reuse=tf.AUTO_REUSE, scope="g_deconv4")
    #x = slim.batch_norm(x)
    x = tf.nn.tanh(x)

    return x


def Discriminator(x):
    base = 64
    x = slim.conv2d(x, base, [5,5], stride=[2,2], activation_fn=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE,  scope="d_conv1")
    x = slim.conv2d(x, base * 2, [5,5], stride=[2,2], activation_fn=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE, scope="d_conv2")
    x = slim.conv2d(x, base * 4, [5,5], stride=[2,2], activation_fn=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE, scope="d_conv3")
    x = slim.conv2d(x, base * 8, [5,5], stride=[2,2], activation_fn=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE, scope="d_conv4")
    x = slim.flatten(x)
    x = slim.fully_connected(x, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope="d_dense")

    return x
    

    
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
    tf.reset_default_graph()

    # place holder
    X = tf.placeholder(tf.float32, [None, 100])
    X2 = tf.placeholder(tf.float32, [None, img_height, img_width, channel])
    Y = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32)
    
    g_logits = Generator(X)
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

    train_x, train_y, test_x, test_y = load_cifar10()
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

            input_noise = np.random.uniform(-1, 1, size=(mb, 100))

            g_output = sess.run(g_logits, feed_dict={X: input_noise})

            _X = np.concatenate([x, g_output])
            _Y = np.array([1] * mb + [0] * mb, dtype=np.float32)
            _Y = _Y[..., None]
    
            _, d_loss = sess.run([D_train, D_loss], feed_dict={X2:_X, Y:_Y})

            _Y = np.array([1] * mb, dtype=np.float32)
            _Y = _Y[..., None]
            _, g_loss = sess.run([G_train, G_loss], feed_dict={X:input_noise, Y: _Y})
            
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

    logits = Generator(X)
    
    np.random.seed(100)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./cnn.ckpt")

        for i in range(3):
            input_noise = np.random.uniform(-1, 1, size=(10, 100))
            g_output = sess.run(logits, feed_dict={X: input_noise})
            g_output = (g_output + 1 ) / 2

            for i in range(10):
                gen = g_output[i]
                plt.subplot(1,10,i+1)
                plt.imshow(gen)
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
