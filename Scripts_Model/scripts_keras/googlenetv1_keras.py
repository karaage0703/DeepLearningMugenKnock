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
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization, concatenate, AveragePooling2D

num_classes = 2
img_height, img_width = 224, 224
channel = 3

def GoogLeNetv1():

    def inception_module(x, f_1, f_2_1, f_2_2, f_3_1, f_3_2, f_4_2):
        x1 = Conv2D(f_1, [1, 1], strides=1, padding='same', activation='relu')(x)

        x2_1 = Conv2D(f_2_1, [1, 1], strides=1, padding='same', activation='relu')(x)
        x2_2 = Conv2D(f_2_2, [3, 3], strides=1, padding='same', activation='relu')(x2_1)

        x3_1 = Conv2D(f_3_1, [1, 1], strides=1, padding='same', activation='relu')(x)
        x3_2 = Conv2D(f_3_2, [5, 5], strides=1, padding='same', activation='relu')(x3_1)

        x4_1 = MaxPooling2D([3, 3], strides=1, padding='same')(x)
        x4_2 = Conv2D(f_4_2, [1, 1], strides=1, padding='same', activation='relu')(x4_1)

        x = concatenate([x1, x2_2, x3_2, x4_2])

        return x
        
    
    inputs = Input((img_height, img_width, 3))
    x = inputs
    
    x = Conv2D(64, [7, 7], strides=2, padding='valid', activation='relu')(x)
    x = MaxPooling2D([3, 3], strides=2, padding='same')(x)

    x = Conv2D(64, [1, 1], strides=1, padding='same', activation='relu')(x)
    x = Conv2D(192, [3, 3], strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D([3, 3], strides=2, padding='same')(x)

    # inception 3a, 3b
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)
    x = MaxPooling2D([3, 3], strides=2, padding='same')(x)

    # inception 4a
    x = inception_module(x, 192, 96, 208, 16, 48, 64)

    # auxiliary loss1
    x_aux1 = AveragePooling2D([5, 5], strides=1, padding='same')(x)
    x_aux1 = Conv2D(128, [1, 1], strides=1, padding='same', activation='relu')(x_aux1)
    x_aux1 = Flatten()(x_aux1)
    x_aux1 = Dense(1024, activation='relu')(x_aux1)
    x_aux1 = Dropout(0.7)(x_aux1)
    x_aux1 = Dense(num_classes, activation='softmax', name='out_aux1')(x_aux1)

    # inception 4b, 4c, 4d
    x = inception_module(x, 160, 112, 224, 24, 64, 64)
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    x = inception_module(x, 112, 144, 288, 32, 64, 64)

    # auxiliary loss2
    x_aux2 = AveragePooling2D([5, 5], strides=1, padding='same')(x)
    x_aux2 = Conv2D(128, [1, 1], strides=1, padding='same', activation='relu')(x_aux2)
    x_aux2 = Flatten()(x_aux2)
    x_aux2 = Dense(1024, activation='relu')(x_aux2)
    x_aux2 = Dropout(0.7)(x_aux2)
    x_aux2 = Dense(num_classes, activation='softmax', name='out_aux2')(x_aux2)
    

    # inception 4e, 5a, 5b
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = MaxPooling2D([3, 3], strides=2, padding='same')(x)
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = inception_module(x, 384, 192, 384, 48, 128, 128)

    x = AveragePooling2D([7, 7], strides=1, padding='same')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='out')(x)

    model = Model(inputs=inputs, outputs=[x, x_aux1, x_aux2])

    return model
    


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
    model = GoogLeNetv1()

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    xs, ts, paths = data_load('../Dataset/train/images', hf=True, vf=True, rot=1)

    # training
    mb = 64
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

        loss_total, loss, loss_aux1, loss_aux2, acc, acc_aux1, acc_aux2 = \
                model.train_on_batch(x=x, y={'out':t, 'out_aux1':t, 'out_aux2':t})

        if (i+1) % 10 == 0:
            print("iter >>", i+1, ",loss >>", loss_total, ',accuracy >>', acc)

    model.save('model.h5')

# test
def test():
    # load trained model
    model = GoogLeNetv1()
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
