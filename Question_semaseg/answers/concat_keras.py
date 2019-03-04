import keras
import cv2
import numpy as np
import argparse
from glob import glob
import matplotlib.pyplot as plt

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
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization, Reshape

num_classes = 2
img_height, img_width = 64, 64#572, 572
out_height, out_width = 64, 64#388, 388
    
def Mynet(train=False):
    inputs = Input((img_height, img_width, 3), name='in')
    x = inputs
    for i in range(2):
        x = Conv2D(32, (3, 3), padding='same', strides=1, name='conv1_{}'.format(i+1))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

    enc1 = x

    x = MaxPooling2D((2,2), 2)(x)

    for i in range(2):
        x = Conv2D(32, (3, 3), padding='same', strides=1, name='conv2_{}'.format(i+1))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

    x = keras.layers.Conv2DTranspose(32, (2,2), strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = keras.layers.concatenate([x, enc1])
    x = Conv2D(32, (1, 1), padding='same', strides=1, name='concat_conv')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    
    for i in range(2):
        x = Conv2D(32, (3, 3), padding='same', strides=1, name='dec1_{}'.format(i+1))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
    
    x = Conv2D(num_classes+1, (1, 1), padding='same', strides=1)(x)
    x = Reshape([-1, num_classes+1])(x)
    x = Activation('softmax', name='out')(x)
    
    model = Model(inputs=inputs, outputs=x, name='model')
    return model

    
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

            t = np.zeros((out_height, out_width, num_classes+1), dtype=np.int)

            for i, (_, vs) in enumerate(CLS.items()):
                ind = (gt[...,0] == vs[0]) * (gt[...,1] == vs[1]) * (gt[...,2] == vs[2])
                ind = np.where(ind == True)
                t[ind[0], ind[1], i] = 1

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
    model = Mynet(train=True)

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        loss={'out': 'categorical_crossentropy'},
        optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False),
        loss_weights={'out': 1},
        metrics=['accuracy'])


    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True)

    # training
    mb = 4
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

        t = np.reshape(t, (mb, -1, num_classes+1))

        loss, acc = model.train_on_batch(x={'in':x}, y={'out':t})
        print("iter >>", i+1, ",loss >>", loss, ',accuracy >>', acc)

    model.save('model.h5')

# test
def test():
    # load trained model
    model = Mynet(train=False)
    model.load_weights('model.h5')

    xs, ts, paths = data_load('../Dataset/test/images/')

    for i in range(len(paths)):
        x = xs[i]
        t = ts[i]
        path = paths[i]
        
        x = np.expand_dims(x, axis=0)
        
        pred = model.predict_on_batch(x={'in': x})[0]
        pred = np.reshape(pred, (out_height, out_width, num_classes+1))

        pred = pred.argmax(axis=-1)

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
