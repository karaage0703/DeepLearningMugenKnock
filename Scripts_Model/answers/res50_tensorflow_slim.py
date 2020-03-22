import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.contrib import slim

import argparse
import cv2
import numpy as np
from glob import glob
import copy

num_classes = 2
img_height, img_width = 224, 224
channel = 3
tf.set_random_seed(0)


def Res50(x, keep_prob):

    def ResBlock(x, in_f, f_1, out_f, stride=1):
        res_x = slim.conv2d(x, f_1, [1, 1], stride=stride, padding="SAME", activation_fn=None)
        res_x = slim.batch_norm(res_x)
        res_x = tf.nn.relu(res_x)

        res_x = slim.conv2d(res_x, f_1, [3, 3], stride=1, padding="SAME", activation_fn=None)
        res_x = slim.batch_norm(res_x)
        res_x = tf.nn.relu(res_x)

        res_x = slim.conv2d(res_x, out_f, [1, 1], stride=1, padding="SAME", activation_fn=None)
        res_x = slim.batch_norm(res_x)
        res_x = tf.nn.relu(res_x)

        if in_f != out_f:
            x = slim.conv2d(x, out_f, [1, 1], stride=1, padding="SAME", activation_fn=None)
            x = slim.batch_norm(x)
            x = tf.nn.relu(x)

        if stride != 1:
            x = slim.max_pool2d(x, [2, 2], stride=stride, padding="SAME")
        
        x = tf.add(res_x, x)

        return x

    
    x = slim.conv2d(x, 64, [7, 7], stride=2, padding="SAME", activation_fn=None)
    x = slim.batch_norm(x)
    x = tf.nn.relu(x)
    
    x = slim.max_pool2d(x, [3, 3], stride=2, padding='SAME')

    x = ResBlock(x, 64, 64, 256)
    x = ResBlock(x, 256, 64, 256)
    x = ResBlock(x, 256, 64, 256)

    x = ResBlock(x, 256, 128, 512, stride=2)
    x = ResBlock(x, 512, 128, 512)
    x = ResBlock(x, 512, 128, 512)
    x = ResBlock(x, 512, 128, 512)

    x = ResBlock(x, 512, 256, 1024, stride=2)
    x = ResBlock(x, 1024, 256, 1024)
    x = ResBlock(x, 1024, 256, 1024)
    x = ResBlock(x, 1024, 256, 1024)
    x = ResBlock(x, 1024, 256, 1024)
    x = ResBlock(x, 1024, 256, 1024)

    x = ResBlock(x, 1024, 512, 2048, stride=2)
    x = ResBlock(x, 2048, 256, 2048)
    x = ResBlock(x, 2048, 256, 2048)

    x = slim.avg_pool2d(x, [img_height // 32, img_width // 32], stride=1, padding='VALID')
    mb, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * c])
    x = slim.fully_connected(x, num_classes)
    
    return x


CLS = ['akahara', 'madara']

# get train data
def data_load(path, hf=False, vf=False, rot=False):
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

            if rot != False:
                angle = rot
                scale = 1

                # show
                a_num = 360 // rot
                w_num = np.ceil(np.sqrt(a_num))
                h_num = np.ceil(a_num / w_num)
                count = 1
                #plt.subplot(h_num, w_num, count)
                #plt.axis('off')
                #plt.imshow(x)
                #plt.title("angle=0")
                
                while angle < 360:
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

                    # show
                    #count += 1
                    #plt.subplot(h_num, w_num, count)
                    #plt.imshow(_x)
                    #plt.axis('off')
                    #plt.title("angle={}".format(angle))

                    angle += rot
                #plt.show()


    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)

    return xs, ts, paths



# train
def train():
    tf.reset_default_graph()

    # place holder
    X = tf.placeholder(tf.float32, [None, img_height, img_width, channel])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    
    logits = Res50(X, keep_prob)
    preds = tf.nn.softmax(logits)
    
    #loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9)
    train = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    

    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True, rot=1)

    # training
    mb = 16
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
                mb_ind = copy.copy(train_ind)[mbi:]
                np.random.shuffle(train_ind)
                mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
                mbi = mb - (len(xs) - mbi)
            else:
                mb_ind = train_ind[mbi: mbi+mb]
                mbi += mb

            x = xs[mb_ind]
            t = ts[mb_ind]

            _, acc, los = sess.run([train, accuracy, loss], feed_dict={X: x, Y: t, keep_prob: 0.7})
            if (i + 1) % 10 == 0:
                print("iter >>", i+1, ', loss >>', los / mb, ', accuracy >>', acc)

        saver = tf.train.Saver()
        saver.save(sess, './cnn.ckpt')


        
# test
def test():
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    logits = Res50(X, keep_prob)
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
