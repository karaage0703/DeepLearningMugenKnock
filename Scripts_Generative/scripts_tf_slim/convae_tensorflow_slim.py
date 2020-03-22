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
img_height, img_width = 64, 64
out_height, out_width = 64, 64
channel = 3

    
def Mynet(x, keep_prob, train=False):
    x = slim.conv2d(x, 32, [3,3], padding='same', scope='enc1')
    x = slim.max_pool2d(x, [2,2], scope='pool1')
    x = slim.conv2d(x, 16, [3,3], padding='same', scope='enc2')
    x = slim.max_pool2d(x, [2,2], scope='pool2')
    x = slim.conv2d_transpose(x, 32, [2,2], stride=2, scope='dec2')
    x = slim.conv2d_transpose(x, channel, [2,2], stride=2, scope='dec1')
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
    X = tf.placeholder(tf.float32, [None, img_height, img_width, channel])
    Y = tf.placeholder(tf.float32, [None, out_height, out_width, channel])
    keep_prob = tf.placeholder(tf.float32)
    
    logits = Mynet(X, keep_prob, train=True)
    
    preds = logits
    loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=logits, labels=Y))
    #loss = tf.reduce_mean(tf.square(logits - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)

    correct_pred = tf.equal(preds, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    

    xs, paths = data_load('../Dataset/train/images/', hf=True, vf=True, rot=1)

    # training
    mb = 64
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
    
        for i in range(500):
            if mbi + mb > len(xs):
                mb_ind = train_ind[mbi:]
                np.random.shuffle(train_ind)
                mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
                mbi = mb - (len(xs) - mbi)
            else:
                mb_ind = train_ind[mbi: mbi+mb]
                mbi += mb

            x = xs[mb_ind]
            t = x.copy()

            _, acc, los = sess.run([train, accuracy, loss], feed_dict={X: x, Y: t, keep_prob: 0.5})
            print("iter >>", i+1, ',loss >>', los / mb, ',accuracy >>', acc)

        saver = tf.train.Saver()
        saver.save(sess, './cnn.ckpt')

# test
def test():
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, img_height, img_width, channel])
    Y = tf.placeholder(tf.float32, [None, out_height * out_width * channel])
    keep_prob = tf.placeholder(tf.float32)

    logits = Mynet(X, keep_prob, train=True)

    xs, paths = data_load("../Dataset/test/images/")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./cnn.ckpt")

        for i in range(len(paths)):
            x = xs[i]
            path = paths[i]
            
            x = np.expand_dims(x, axis=0)

            pred = sess.run([logits], feed_dict={X: x, keep_prob:1.0})[0]
            pred = (pred[0] + 1) / 2
            pred -= pred.min()
            pred /= pred.max()

            if channel == 1:
                pred = pred[..., 0]
                _x = (x[0, ..., 0] + 1) / 2
                cmap = 'gray'
            else:
                _x = (x[0] + 1) / 2
                cmap = None
            
            print("in {}".format(path))
            
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
