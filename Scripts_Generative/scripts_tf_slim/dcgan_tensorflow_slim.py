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
    

CLS = {'background': [0,0,0],
       'akahara': [0,0,128],
       'madara': [0,128,0]}
    
# get train data
def data_load(path, hf=False, vf=False, rot=False):
    xs = []
    ts = []
    paths = []

    data_num = 0
    for dir_path in glob(path + '/*'):
        data_num += len(glob(dir_path + "/*"))
            
    pbar = tqdm(total = data_num)
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            if channel == 1:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x = x / 127.5 - 1
            if channel == 1:
                x = x[..., None]
            else:
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
                angle = 0
                scale = 1
                while angle < 360:
                    angle += rot
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

            pbar.update(1)
                    
    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)
    #xs = np.transpose(xs, (0,3,1,2))
    pbar.close()
    
    return xs, paths



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

    xs, paths = data_load('../Dataset/train/images/', hf=True, vf=True, rot=1)

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
