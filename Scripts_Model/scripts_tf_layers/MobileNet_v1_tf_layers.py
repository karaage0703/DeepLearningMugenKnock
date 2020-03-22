import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import argparse
import cv2
import numpy as np
from glob import glob
import copy

CLS = ['akahara', 'madara']
num_classes = len(CLS)
img_height, img_width = 128, 128
channel = 3
#tf.random.set_random_seed(0)


def MobileNet_v1(x, keep_prob, train=True):

    def MobileNetBlock(x, in_dim, out_dim, repeat=1, strides=1):
        for _ in range(repeat):
            x = tf.layers.separable_conv2d(x, in_dim, [3, 3], strides=strides, padding='same', activation=None, depth_multiplier=in_dim)
            x = tf.layers.batch_normalization(x, training=train)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(x, out_dim, [1, 1], strides=1, padding='same', activation=None)
            x = tf.layers.batch_normalization(x, training=train)
            x = tf.nn.relu(x)
        
        return x

    #-----
    # 1/1 x 1/1 x 3
    #-----
    x = tf.layers.conv2d(x, 32, [3, 3], strides=2, padding="same", activation=None)
    x = tf.layers.batch_normalization(x, training=train)
    x = tf.nn.relu(x)

    #-----
    # 1/2 x 1/2 x 32
    #-----
    MobileNetBlock(x, 32, 64)

    #-----
    # 1/4 x 1/4 x 64
    #-----
    MobileNetBlock(x, 64, 128, strides=2)
    MobileNetBlock(x, 128, 128)

    #-----
    # 1/8 x 1/8 x 128
    #-----
    MobileNetBlock(x, 128, 256, strides=2)
    MobileNetBlock(x, 256, 256)

    #-----
    # 1/16 x 1/16 x 256
    #-----
    MobileNetBlock(x, 256, 512, strides=2)
    MobileNetBlock(x, 512, 512, repeat=5)

    #-----
    # 1/32 x 1/32 x 1024
    #-----
    MobileNetBlock(x, 512, 1024, strides=2)

    x = tf.reduce_mean(x, axis=[1, 2])
    x = tf.layers.dense(x, num_classes)
    
    return x


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
    #tf.reset_default_graph()

    # place holder
    X = tf.placeholder(tf.float32, [None, img_height, img_width, channel])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    
    logits = MobileNet_v1(X, keep_prob, train=False)
    preds = tf.nn.softmax(logits)
    
    #loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    train = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    

    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True, rot=10)

    # training
    mb = 32
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
    
        for i in range(1000):
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

    logits = MobileNet_v1(X, keep_prob, train=False)
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
