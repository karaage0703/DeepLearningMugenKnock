import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import argparse
import cv2
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 32, 32
tf.set_random_seed(0)

def conv2d(x, k=3, in_num=1, out_num=32, strides=1, padding='SAME', activ=None, bias=True, name='conv'):
    w = tf.Variable(tf.random_normal([k, k, in_num, out_num]), name=name+'_w')
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
    tf.add_to_collections('vars', w)
    if bias:
        b = tf.Variable(tf.random_normal([out_num]), name=name+'_b')
        tf.add_to_collections('vars', b)
        x = tf.nn.bias_add(x, b)
    if activ is not None:
        x = activ(x)
    return x

def maxpool2d(x, k=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)

def fc(x, in_num=100, out_num=100, bias=True, activ=None, name='fc'):
    w = tf.Variable(tf.random_normal([in_num, out_num]), name=name+'_w')
    x = tf.matmul(x, w)
    tf.add_to_collections('vars', w)
    if bias:
        b = tf.Variable(tf.random_normal([out_num]), name=name+'_b')
        tf.add_to_collections('vars', b)
        x = tf.add(x, b)
    if activ is not None:
        x = activ(x)
    return x

def LeNet(x, keep_prob):
    x = conv2d(x, k=5, in_num=3, out_num=6, padding='VALID', activ=None, name='conv1')
    x = tf.nn.sigmoid(maxpool2d(x, k=2))
    x = conv2d(x, k=5, in_num=6, out_num=16, padding='VALID', activ=None, name='conv2')
    x = tf.nn.sigmoid(maxpool2d(x, k=2))

    mb, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h*w*c])
    x = fc(x, in_num=w*h*c, out_num=120, activ=None, name='fc1')
    x = fc(x, in_num=120, out_num=64, activ=None, name='fc2')
    x = fc(x, in_num=64, out_num=num_classes, name='fc_out')
    return x

CLS = ['akahara', 'madara']

# get train data
def data_load(path, hf=False, vf=False):
    xs = []
    ts = []
    paths = []
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs.append(x)

            t = [0 for _ in range(num_classes)]
            for i, cls in enumerate(CLS):
                if cls in path:
                    t[i] = 1
            
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

    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)

    return xs, ts, paths


# train
def train():
    tf.reset_default_graph()

    # place holder
    X = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    
    logits = LeNet(X, keep_prob)
    preds = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y))
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9)
    train = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    

    xs, ts, paths = data_load('../Dataset/train/images/')

    # training
    mb = 8
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
            t = ts[mb_ind]

            _, acc, los = sess.run([train, accuracy, loss], feed_dict={X: x, Y: t, keep_prob: 0.5})
            print("iter >>", i+1, ',loss >>', los / mb, ',accuracy >>', acc)

        saver = tf.train.Saver()
        saver.save(sess, './cnn.ckpt')

# test
def test():
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    logits = LeNet(X, keep_prob)
    out = tf.nn.softmax(logits)

    xs, ts, paths = data_load("../Dataset/test/images/")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        #saver = tf.train.import_meta_graph("./cnn.ckpt.meta")
        saver.restore(sess, "./cnn.ckpt")

        for i in range(len(paths)):
            x = xs[i]
            t = ts[i]
            path = paths[i]
            
            x = np.expand_dims(x, axis=0)

            pred = sess.run([out], feed_dict={X:x, keep_prob:1.})[0]
            pred = pred[0]
            #pred = out.eval(feed_dict={X: x, keep_prob: 1.0})[0]

            print("in {}, predicted probabilities >> {}".format(path, pred))
    

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
