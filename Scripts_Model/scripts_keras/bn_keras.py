import keras
import cv2
import numpy as np
import argparse
from glob import glob

# GPU config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
sess = tf.Session(config=config)
K.set_session(sess)

# network
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization

num_classes = 2
img_height, img_width = 64, 64

def VGG16():
    inputs = Input((img_height, img_width, 3))
    x = inputs
    # block conv1
    for i in range(2):
        x = Conv2D(64, (3, 3), padding='same', strides=1, activation='relu', name='conv1_{}'.format(i+1))(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=2,  padding='same')(x)
    
    # block conv2
    for i in range(2):
        x = Conv2D(128, (3, 3), padding='same', strides=1, activation='relu', name='conv2_{}'.format(i+1))(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=2,  padding='same')(x)
    
    # block conv3
    for i in range(3):
        x = Conv2D(256, (3, 3), padding='same', strides=1, activation='relu', name='conv3_{}'.format(i+1))(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=2,  padding='same')(x)
    
    # block conv4
    for i in range(3):
        x = Conv2D(512, (3, 3), padding='same', strides=1, activation='relu', name='conv4_{}'.format(i+1))(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=2,  padding='same')(x)
    
    # block conv5
    for i in range(3):
        x = Conv2D(512, (3, 3), padding='same', strides=1, activation='relu', name='conv5_{}'.format(i))(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=2,  padding='same')(x)
    
    x = Flatten()(x)
    x = Dense(4096, name='dense1', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, name='dense2', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x, name='model')
    return model

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
    model = VGG16()

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    print(model.summary())

    xs, ts, paths = data_load('../Dataset/train/images', hf=True, vf=True)

    # training
    mb = 8
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
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

        loss, acc = model.train_on_batch(x=x, y=t)
        print("iter >>", i+1, ",loss >>", loss, ',accuracy >>', acc)

    model.save('model.h5')

# test
def test():
    # load trained model
    model = VGG16()
    model.load_weights('model.h5')

    xs, ts, paths = data_load("../Dataset/test/images/")

    for i in range(len(paths)):
        x = xs[i]
        t = ts[i]
        path = paths[i]
        
        x = np.expand_dims(x, axis=0)
        
        pred = model.predict_on_batch(x)[0]
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
