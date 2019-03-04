import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import argparse
import cv2
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 64, 64

def conv2d(x, k=3, in_num=1, out_num=32, strides=1, activ=None, bias=True, name='conv'):
    w = tf.Variable(tf.random_normal([k, k, in_num, out_num]), name=name+'_w')
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    tf.add_to_collections('vars', w)
    if bias:
        b = tf.Variable(tf.random_normal([out_num]), name=name+'_b')
        tf.add_to_collections('vars', b)
        x = tf.nn.bias_add(x, b)
    if activ is not None:
        x = activ(x)
    return x

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

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

def Mynet(x, keep_prob):
    x = conv2d(x, k=3, in_num=3, out_num=32, activ=tf.nn.relu, name='conv1_1')
    x = conv2d(x, k=3, in_num=32, out_num=32, activ=tf.nn.relu, name='conv1_2')
    x = maxpool2d(x, k=2)
    x = conv2d(x, k=3, in_num=32, out_num=64, activ=tf.nn.relu, name='conv2_1')
    x = conv2d(x, k=3, in_num=64, out_num=64, activ=tf.nn.relu, name='conv2_2')
    x = maxpool2d(x, k=2)
    x = conv2d(x, k=3, in_num=64, out_num=128, activ=tf.nn.relu, name='conv3_1')
    x = conv2d(x, k=3, in_num=128, out_num=128, activ=tf.nn.relu, name='conv3_2')
    x = maxpool2d(x, k=2)

    mb, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h*w*c])
    x = fc(x, in_num=w*h*c, out_num=1024, activ=tf.nn.relu, name='fc1')
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    x = fc(x, in_num=1024, out_num=num_classes, name='fc_out')
    return x

# get train data
def data_load(dir_path):
    xs = np.ndarray((0, img_height, img_width, 3))
    ts = np.ndarray((0, num_classes))
    paths = []
    
    for dir_path in glob(dir_path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs = np.r_[xs, x[None, ...]]

            t = np.zeros((num_classes))
            if 'akahara' in path:
                t[0] = 1
            elif 'madara' in path:
                t[1] = 1
            ts = np.r_[ts, t[None, ...]]

            paths += [path]

    return xs, ts, paths


# train
def train():
    tf.reset_default_graph()

    # place holder
    X = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    
    logits = Mynet(X, keep_prob)
    preds = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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
    
        for i in range(100):
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

    logits = Mynet(X, keep_prob)
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
