import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


num_classes = 2
img_height, img_width = 64, 64#572, 572
out_height, out_width = 64, 64#388, 388
    
def Mynet(x, keep_prob, train=False):
    # block conv1
    for i in range(6):
        x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], strides=1, padding='same', name='conv1_{}'.format(i+1))
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=train)
        #x = tf.nn.relu(x)

    x = tf.layers.conv2d(inputs=x, filters=1, kernel_size=[1, 1], strides=1, padding='same', name='out')
    
    return x

    
CLS = {'akahara': [0,0,128],
       'madara': [0,128,0]}
    
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
            x = x[..., ::-1]
            xs.append(x)

            gt_path = path.replace("images", "seg_images").replace(".jpg", ".png")
            gt = cv2.imread(gt_path)
            gt = cv2.resize(gt, (out_width, out_height), interpolation=cv2.INTER_NEAREST)

            t = np.zeros((out_height, out_width, 1), dtype=np.float)

            for i , (label, vs) in enumerate(CLS.items()):
                ind = (gt[...,0] == vs[0]) * (gt[...,1] == vs[1]) * (gt[...,2] == vs[2])
                t[ind] = 1
            #ind = (gt[...,0] > 0) + (gt[..., 1] > 0) + (gt[...,2] > 0)
            #t[ind] = 1

            #print(gt_path)
            #import matplotlib.pyplot as plt
            #plt.imshow(t, cmap='gray')
            #plt.show()

            ts.append(t)
            
            paths.append(path)

            if hf:
                xs.append(x[:, ::-1])
                ts.append(t[:, ::-1])
                paths.append(path)

            if vf:
                xs.append(x[::-1])
                ts.append(t[::-1])
                paths.append(path)

            if hf and vf:
                xs.append(x[::-1, ::-1])
                ts.append(t[::-1, ::-1])
                paths.append(path)

    xs = np.array(xs)
    ts = np.array(ts)

    return xs, ts, paths


# train
def train():
    tf.reset_default_graph()

    # place holder
    X = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
    Y = tf.placeholder(tf.float32, [None, out_height, out_width, 1])
    keep_prob = tf.placeholder(tf.float32)
    
    logits = Mynet(X, keep_prob, train=True)
    preds = tf.nn.sigmoid(logits)
    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=Y))
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    train = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    

    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True)

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

    logits = Mynet(X, keep_prob, train=True)
    pred_prob = tf.nn.sigmoid(logits)

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

            pred = sess.run([pred_prob], feed_dict={X:x, keep_prob:1.})[0]
            pred = pred[0, ..., 0]
            #pred = out.eval(feed_dict={X: x, keep_prob: 1.0})[0, ..., 0]
            
            ## binalization
            bin_pred = pred.copy()
            th = 0.5
            bin_pred[bin_pred >= th] = 1
            bin_pred[bin_pred < th] = 0
            
            print("in {}".format(path))
            
            plt.subplot(1,3,1)
            plt.imshow(x[0])
            plt.title("input")
            plt.subplot(1,3,2)
            plt.imshow(pred, cmap='gray')
            plt.title("predicted")
            plt.subplot(1,3,3)
            plt.imshow(bin_pred, cmap='gray')
            plt.title("after binalization")
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
