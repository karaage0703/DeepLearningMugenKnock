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
img_height, img_width = 236, 236 #572, 572
out_height, out_width = 52, 52 #388, 388

def crop_layer(layer, size):
    _, h, w, _ = layer.get_shape().as_list()
    _, _h, _w, _ = size
    ph = int((h - _h) / 2)
    pw = int((w - _w) / 2)
    return layer[:, ph:ph+_h, pw:pw+_w]
    
    
def Mynet(x, keep_prob, train=False):
    # block conv1

    base = 64
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      #activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
                      #weights_regularizer=slim.l2_regularizer(0.0005)):
        enc1 = x
        for i in range(2):
            enc1 = slim.conv2d(enc1, base, [3,3], padding="valid", scope='conv1_{}'.format(i+1))
            enc1 = tf.nn.relu(enc1)
            enc1 = slim.batch_norm(enc1, is_training=train)
            
        enc2 = slim.max_pool2d(enc1, [2,2], scope='pool1')
        for i in range(2):
            enc2 = slim.conv2d(enc2, base*2, [3,3], padding="valid", scope='conv2_{}'.format(i+1))
            enc2 = tf.nn.relu(enc2)
            enc2 = slim.batch_norm(enc2, is_training=train)

        enc3 = slim.max_pool2d(enc2, [2,2], scope='pool2')
        for i in range(2):
            enc3 = slim.conv2d(enc3, base*4, [3,3], padding="valid", scope='conv3_{}'.format(i+1))
            enc3 = tf.nn.relu(enc3)
            enc3 = slim.batch_norm(enc3, is_training=train)

        enc4 = slim.max_pool2d(enc3, [2,2], scope='pool3')
        for i in range(2):
            enc4 = slim.conv2d(enc4, base*8, [3,3], padding="valid", scope='conv4_{}'.format(i+1))
            enc4 = tf.nn.relu(enc4)
            enc4 = slim.batch_norm(enc4, is_training=train)

        enc5 = slim.max_pool2d(enc4, [2,2], scope='pool4')
        for i in range(2):
            enc5 = slim.conv2d(enc5, base*16, [3,3], padding="valid", scope='conv5_{}'.format(i+1))
            enc5 = tf.nn.relu(enc5)
            enc5 = slim.batch_norm(enc5, is_training=train)

        # decoder4
        dec4 = slim.conv2d_transpose(enc5, base*8, [2,2], stride=2, scope='tconv4')
        dec4 = tf.nn.relu(dec4)
        dec4 = slim.batch_norm(dec4, is_training=train)

        _enc4 = crop_layer(enc4, dec4.get_shape().as_list())

        dec4 = tf.concat((dec4, _enc4), axis=-1)
        
        for i in range(2):
            dec4 = slim.conv2d(dec4, base*8, [3,3], padding='valid', scope='dec4_{}'.format(i+1))
            dec4 = tf.nn.relu(dec4)
            dec4 = slim.batch_norm(dec4, is_training=train)

        # decoder 3
        dec3 = slim.conv2d_transpose(dec4, base*4, [2,2], stride=2, scope='tconv3')
        dec3 = tf.nn.relu(dec3)
        dec3 = slim.batch_norm(dec3, is_training=train)

        _enc3 = crop_layer(enc3, dec3.get_shape().as_list())
        dec3 = tf.concat((dec3, _enc3), axis=-1)
        
        for i in range(2):
            dec3 = slim.conv2d(dec3, base*4, [3,3], padding='valid', scope='dec3_{}'.format(i+1))
            dec3 = tf.nn.relu(dec3)
            dec3 = slim.batch_norm(dec3, is_training=train)

        # decoder 2
        dec2 = slim.conv2d_transpose(dec3, base*2, [2,2], stride=2, scope='tconv2')
        dec2 = tf.nn.relu(dec2)
        dec2 = slim.batch_norm(dec2, is_training=train)

        _enc2 = crop_layer(enc2, dec2.get_shape().as_list())
        dec2 = tf.concat((dec2, _enc2), axis=-1)
        
        for i in range(2):
            dec2 = slim.conv2d(dec2, base*2, [3,3], padding='valid', scope='dec2_{}'.format(i+1))
            dec2 = tf.nn.relu(dec2)
            dec2 = slim.batch_norm(dec2, is_training=train)

        # decoder 1
        dec1 = slim.conv2d_transpose(dec2, base, [2,2], stride=2, scope='tconv1')
        dec1 = tf.nn.relu(dec1)
        dec1 = slim.batch_norm(dec1, is_training=train)

        _enc1 = crop_layer(enc1, dec1.get_shape().as_list())

        dec1 = tf.concat((dec1, _enc1), axis=-1)
        
        for i in range(2):
            dec1 = slim.conv2d(dec1, base, [3,3], padding='valid', scope='dec1_{}'.format(i+1))
            dec1 = tf.nn.relu(dec1)
            dec1 = slim.batch_norm(dec1, is_training=train)
            
    out = slim.conv2d(dec1, num_classes+1, [1, 1], scope='out')

    return out

    
CLS = {'background': [0,0,0],
       'akahara': [0,0,128],
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

            t = np.zeros((out_height, out_width, num_classes+1), dtype=np.float)

            for i , (label, vs) in enumerate(CLS.items()):
                ind = (gt[...,0] == vs[0]) * (gt[...,1] == vs[1]) * (gt[...,2] == vs[2])
                ind = np.where(ind == True)
                t[ind[0], ind[1], i] = 1

            #ind = (gt[..., 0] == 0) * (gt[..., 1] == 0) * (gt[..., 2] == 0)
            #ind = np.where(ind == True)
            #t[ind[0], ind[1], 0] = 1
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
    Y = tf.placeholder(tf.float32, [None, num_classes+1])
    keep_prob = tf.placeholder(tf.float32)
    
    logits = Mynet(X, keep_prob, train=True)
    logits = tf.reshape(logits, [-1, num_classes+1])
    
    preds = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y))
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    train = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    

    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True)

    # training
    mb = 4
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

            t = np.reshape(t, [-1, num_classes+1])

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
    logits = tf.reshape(logits, [-1, num_classes+1])
    logits = tf.nn.softmax(logits)

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

            pred = sess.run([logits], feed_dict={X: x, keep_prob:1.0})[0]
            pred = np.reshape(pred, [out_height, out_width, num_classes+1])
            pred = np.argmax(pred, axis=-1)

            # visualize
            out = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            for i, (_, vs) in enumerate(CLS.items()):
                out[pred == i] = vs
            
            print("in {}".format(path))
            
            plt.subplot(1,2,1)
            plt.imshow(x[0])
            plt.title("input")
            plt.subplot(1,2,2)
            plt.imshow(out[..., ::-1])
            plt.title("predicted")
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
