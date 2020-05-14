import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.contrib import slim
import argparse
import cv2
import numpy as np
from glob import glob

n_gram = 10

_chars = "あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっー１２３４５６７８９０！？、。@#"
chars = [c for c in _chars]
num_classes = len(chars)

print(num_classes)

def Mynet(x):        
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(x)
    x = slim.fully_connected(x, num_classes)
    return x

    
def data_load():
    fname = 'sandwitchman.txt'
    xs = []
    ts = []
    txt = ''
    for _ in range(n_gram):
        txt += '@'
    onehots = []
    
    with open(fname, 'r') as f:
        for l in f.readlines():
            txt += l.strip() + '#'
        txt = txt[:-1] + '@'

        for c in txt:
            onehot = [0 for _ in range(num_classes)]
            onehot[chars.index(c)] = 1
            onehots.append(onehot)
        
        for i in range(len(txt) - n_gram - 1):
            xs.append(onehots[i:i+n_gram])
            ts.append(onehots[i+n_gram])

    xs = np.array(xs)
    ts = np.array(ts)
    
    return xs, ts


# train
def train():
    tf.reset_default_graph()

    # place holder
    X = tf.placeholder(tf.float32, [None, n_gram, num_classes])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    
    logits = Mynet(X)
    preds = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    xs, ts = data_load()
    
    # training
    mb = 128
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
    
        for i in range(2000):
            if mbi + mb > len(xs):
                mb_ind = train_ind[mbi:]
                np.random.shuffle(train_ind)
                mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            else:
                mb_ind = train_ind[mbi: mbi+mb]
                mbi += mb

            x = xs[mb_ind]
            t = ts[mb_ind]

            _, acc, los = sess.run([train, accuracy, loss], feed_dict={X:x, Y:t, keep_prob:0.5})
            print("iter >>", i+1, ',loss >>', los / mb, ',accuracy >>', acc)

        saver = tf.train.Saver()
        saver.save(sess, './cnn.ckpt')


# test
def test():
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, n_gram, num_classes])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    logits = Mynet(X)
    out = tf.nn.softmax(logits)

    def decode(x):
        return chars[x.argmax()]
    
    gens = ''

    for _ in range(n_gram):
        gens += '@'

    pred = 0
    count = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./cnn.ckpt")
        
        while pred != '@' and count < 1000:
            in_txt = gens[-n_gram:]
            x = []
            for _in in in_txt:
                _x = [0 for _ in range(num_classes)]
                _x[chars.index(_in)] = 1
                x.append(_x)
        
            x = np.expand_dims(np.array(x), axis=0)
            
            pred = sess.run([out], feed_dict={X:x, keep_prob:1.})[0]
            pred = pred[0]
        
            # sample random from probability distribution
            ind = np.random.choice(num_classes, 1, p=pred)

            pred = chars[ind[0]]
            gens += pred
            count += 1

    # pose process
    gens = gens.replace('#', os.linesep).replace('@', '')
        
    print('--\ngenerated')
    print(gens)
    

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
