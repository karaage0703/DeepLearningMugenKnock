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
img_height, img_width = 64, 64 #572, 572
out_height, out_width = 64, 64 #388, 388

def crop_layer(layer, size):
    _, h, w, _ = keras.backend.int_shape(layer)
    _, _h, _w, _ = size
    ph = int((h - _h) / 2)
    pw = int((w - _w) / 2)
    return keras.layers.Cropping2D(cropping=((ph, ph), (pw, pw)))(layer)


def Mynet(train=False):
    base = 16
    
    inputs = Input((img_height, img_width, 3), name='in')
    enc1= inputs

    for i in range(2):
        enc1 = Conv2D(base, (3, 3), padding='same', strides=1, name='conv1_{}'.format(i+1))(enc1)
        enc1 = Activation('relu')(enc1)
        enc1 = BatchNormalization()(enc1)

    enc2 = MaxPooling2D((2,2), 2)(enc1)
    
    for i in range(2):
        enc2 = Conv2D(base*2, (3, 3), padding='same', strides=1, name='conv2_{}'.format(i+1))(enc2)
        enc2 = Activation('relu')(enc2)
        enc2 = BatchNormalization()(enc2)

    enc3 = MaxPooling2D((2,2), 2)(enc2)

    for i in range(2):
        enc3 = Conv2D(base*4, (3, 3), padding='same', strides=1, name='conv3_{}'.format(i+1))(enc3)
        enc3 = Activation('relu')(enc3)
        enc3 = BatchNormalization()(enc3)

    enc4 = MaxPooling2D((2,2), 2)(enc3)

    for i in range(2):
        enc4 = Conv2D(base*8, (3, 3), padding='same', strides=1, name='conv4_{}'.format(i+1))(enc4)
        enc4 = Activation('relu')(enc4)
        enc4 = BatchNormalization()(enc4)

    enc5 = MaxPooling2D((2,2), 2)(enc4)

    for i in range(2):
        enc5 = Conv2D(base*16, (3, 3), padding='same', strides=1, name='conv5_{}'.format(i+1))(enc5)
        enc5 = Activation('relu')(enc5)
        enc5 = BatchNormalization()(enc5)

    dec4 = keras.layers.Conv2DTranspose(base*8, (2,2), strides=2, padding='same')(enc5)
    dec4 = Activation('relu')(dec4)
    dec4 = BatchNormalization()(dec4)
    _enc4 = crop_layer(enc4, keras.backend.int_shape(dec4))
    dec4 = keras.layers.concatenate([dec4, _enc4])
    for i in range(2):
        dec4 = Conv2D(base*8, (3, 3), padding='same', strides=1, name='dec4_{}'.format(i+1))(dec4)
        dec4 = Activation('relu')(dec4)
        dec4 = BatchNormalization()(dec4)

    dec3 = keras.layers.Conv2DTranspose(base*4, (2,2), strides=2, padding='same')(dec4)
    dec3 = Activation('relu')(dec3)
    dec3 = BatchNormalization()(dec3)
    _enc3 = crop_layer(enc3, keras.backend.int_shape(dec3))
    dec3 = keras.layers.concatenate([dec3, _enc3])
    for i in range(2):
        dec3 = Conv2D(base*4, (3, 3), padding='same', strides=1, name='dec3_{}'.format(i+1))(dec3)
        dec3 = Activation('relu')(dec3)
        dec3 = BatchNormalization()(dec3)

    dec2 = keras.layers.Conv2DTranspose(base*2, (2,2), strides=2, padding='same')(dec3)
    dec2 = Activation('relu')(dec2)
    dec2 = BatchNormalization()(dec2)
    _enc2 = crop_layer(enc2, keras.backend.int_shape(dec2))
    dec2 = keras.layers.concatenate([dec2, _enc2])
    for i in range(2):
        dec2 = Conv2D(base*2, (3, 3), padding='same', strides=1, name='dec2_{}'.format(i+1))(dec2)
        dec2 = Activation('relu')(dec2)
        dec2 = BatchNormalization()(dec2)

    dec1 = keras.layers.Conv2DTranspose(base, (2,2), strides=2, padding='same')(dec2)
    dec1 = Activation('relu')(dec1)
    dec1 = BatchNormalization()(dec1)
    _enc1 = crop_layer(enc1, keras.backend.int_shape(dec1))
    dec1 = keras.layers.concatenate([dec1, _enc1])
    for i in range(2):
        dec1 = Conv2D(base, (3, 3), padding='same', strides=1, name='dec1_{}'.format(i+1))(dec1)
        dec1 = Activation('relu')(dec1)
        dec1 = BatchNormalization()(dec1)

    out = Conv2D(num_classes+1, (1, 1), padding='same', strides=1)(dec1)
    out = Reshape([-1, num_classes+1])(out)
    out = Activation('softmax', name='out')(out)
    
    model = Model(inputs=inputs, outputs=out, name='model')
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
    
    for i in range(1000):
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
