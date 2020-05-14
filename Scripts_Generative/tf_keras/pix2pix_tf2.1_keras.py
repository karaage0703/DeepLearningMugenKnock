import tensorflow as tf

print(tf.__version__)

import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from copy import copy
from collections import OrderedDict
from tqdm import tqdm

# class config
class_label = OrderedDict({'background' : [0, 0, 0], 'akahara' : [0, 0, 128], 'madara' : [0, 128, 0]})
class_N = len(class_label)

# model config
img_height, img_width = 64, 64 #572, 572
out_height, out_width = 64, 64 #388, 388
channel = 3
out_channel = class_N

model_path_G = 'pix2pix_G.h5'
model_path_D = 'pix2pix_D.h5'

# training config
Batchsize = 16
Iteration = 5000
Loss_Lambda = 10.

def UNet():
    def UNet_block_downSampling(x, filters, size, name, apply_batchnorm=False):
        x = tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', use_bias=False, name=name + '_conv')(x)
        x = tf.keras.layers.ReLU(name=name + '_ReLU')(x)
        x = tf.keras.layers.BatchNormalization(name=name + '_bn')(x)
        
        return x

    def UNet_block_upSampling(x, filters, size, name, apply_dropout=False):
        x = tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=False, name=name + '_transposedConv')(x)
        x = tf.keras.layers.BatchNormalization(name=name + '_bn')(x)
        x = tf.keras.layers.Dropout(0.5, name=name + 'dropout')(x) if apply_dropout else x
        
        return x

    stride_N = int(np.log(img_height) / np.log(2))

    x_encoders = []

    _input = tf.keras.layers.Input(shape=[img_height, img_width, channel])
    # down sample
    x = _input
    for i in range(1, stride_N + 1):
        x = UNet_block_downSampling(x, min(64 ** i, 512), 3, name='Encoder{}'.format(i))
        x_encoders.append(x)

    # up sample
    for i in range(stride_N - 1, 0, -1):
        x = UNet_block_upSampling(x, min(64 ** i, 512), 3, name='Decoder{}'.format(i - 1))
        x = tf.keras.layers.concatenate([x, x_encoders[i - 1]])

    x_output = tf.keras.layers.Conv2DTranspose(out_channel, 3, strides=2, padding='same', activation='tanh')(x)
    return tf.keras.Model(inputs=_input, outputs=x_output)


def Discriminator():
    def Discriminator_block_downSampling(x, filters, size, name, apply_batchnorm=False):
        x = tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', use_bias=False, name=name + '_conv')(x)
        x = tf.keras.layers.LeakyReLU(name=name + '_leakyReLU')(x) if apply_batchnorm else x
        return x

    stride_N = int(np.log(out_height) / np.log(2))

    _input1 = tf.keras.layers.Input(shape=[out_height, out_width, channel], name='input1')
    _input2 = tf.keras.layers.Input(shape=[out_height, out_width, out_channel], name='input2')
    
    x = tf.keras.layers.concatenate([_input1, _input2]) # (bs, 256, 256, channels*2)

    for i in range(1, stride_N):
        x = Discriminator_block_downSampling(x, min(64 ** i, 512), 5, name='D{}'.format(i))
    
    x = tf.keras.layers.Conv2D(1, 2, strides=1, padding='same')(x)
    
    return tf.keras.Model(inputs=[_input1, _input2], outputs=x)


# get train data
def data_load(path, hf=False, vf=False, rot=False):
    if (rot == 0) and (rot != False):
        raise Exception('invalid rot >> ', rot, 'should be [1, 359] or False')

    paths = []
    paths_gt = []
    
    data_num = 0
    for dir_path in glob(path + '/*'):
        data_num += len(glob(dir_path + "/*"))
            
    pbar = tqdm(total = data_num)
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            for i, cls in enumerate(class_label):
                if cls in path:
                    t = i

            paths.append({'path': path, 'hf': False, 'vf': False, 'rot': 0})
            
            gt_path = path.replace("images", "seg_images").replace(".jpg", ".png")
            paths_gt.append({'path': gt_path, 'hf': False, 'vf': False, 'rot': 0})

            # horizontal flip
            if hf:
                paths.append({'path': path, 'hf': True, 'vf': False, 'rot': 0})
                paths_gt.append({'path': gt_path, 'hf': True, 'vf': False, 'rot': 0})
            # vertical flip
            if vf:
                paths.append({'path': path, 'hf': False, 'vf': True, 'rot': 0})
                paths_gt.append({'path': gt_path, 'hf': False, 'vf': True, 'rot': 0})
            # horizontal and vertical flip
            if hf and vf:
                paths.append({'path': path, 'hf': True, 'vf': True, 'rot': 0})
                paths_gt.append({'path': gt_path, 'hf': True, 'vf': True, 'rot': 0})
            # rotation
            if rot is not False:
                angle = rot
                while angle < 360:
                    paths.append({'path': path, 'hf': False, 'vf': False, 'rot': rot})
                    paths_gt.append({'path': gt_path, 'hf': False, 'vf': False, 'rot': rot})
                    angle += rot
                
            pbar.update(1)
                    
    pbar.close()
    
    return np.array(paths), np.array(paths_gt)

def get_image(infos, gt=False):
    xs = []
    
    for info in infos:
        path = info['path']
        hf = info['hf']
        vf = info['vf']
        rot = info['rot']
        x = cv2.imread(path)

        # resize
        if gt:
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
        else:
            x = cv2.resize(x, (out_width, out_height)).astype(np.float32)
        
        # channel BGR -> Gray
        if channel == 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = np.expand_dims(x, axis=-1)

        # horizontal flip
        if hf:
            x = x[:, ::-1]

        # vertical flip
        if vf:
            x = x[::-1]

        # rotation
        scale = 1
        _h, _w, _c = x.shape
        max_side = max(_h, _w)
        tmp = np.zeros((max_side, max_side, _c))
        tx = int((max_side - _w) / 2)
        ty = int((max_side - _h) / 2)
        tmp[ty: ty+_h, tx: tx+_w] = x.copy()
        M = cv2.getRotationMatrix2D((max_side / 2, max_side / 2), rot, scale)
        _x = cv2.warpAffine(tmp, M, (max_side, max_side))
        x = _x[tx:tx+_w, ty:ty+_h]

        if gt:
            _x = x
            x = np.zeros((out_height, out_width, class_N), dtype=np.int)

            for i, (_, vs) in enumerate(class_label.items()):
                ind = (_x[..., 0] == vs[0]) * (_x[..., 1] == vs[1]) * (_x[..., 2] == vs[2])
                x[..., i][ind] = 1
        else:
            # normalization [0, 255] -> [-1, 1]
            x = x / 127.5 - 1

            # channel BGR -> RGB
            if channel == 3:
                x = x[..., ::-1]

        xs.append(x)
                
    xs = np.array(xs, dtype=np.float32)

    return xs


# train
def train():
    # model
    G = UNet()
    D = Discriminator()

    # optimizer
    G_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    D_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    paths, paths_gt = data_load('../Dataset/train/images/', hf=True, vf=True, rot=False)

    @tf.function
    def train_step(x, target):
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            Gx = G(x, training=True)

            D_real = D([x, target], training=True)
            D_fake = D([x, Gx], training=True)

            # Generator loss
            G_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(D_fake), D_fake)
            G_loss_L1 = tf.reduce_mean(tf.abs(target - Gx))

            G_loss = G_loss_fake + Loss_Lambda * G_loss_L1

            # Discriminator loss
            D_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(D_real), D_real)
            D_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(D_fake), D_fake)

            D_loss = D_loss_real + D_loss_fake


        G_gradients = G_tape.gradient(G_loss, G.trainable_variables)
        D_gradients = D_tape.gradient(D_loss, D.trainable_variables)

        G_optimizer.apply_gradients(zip(G_gradients, G.trainable_variables))
        D_optimizer.apply_gradients(zip(D_gradients, D.trainable_variables))

        return {'G_loss' : G_loss, 'G_loss_fake' : G_loss_fake, 'G_loss_L1' : G_loss_L1, 'D_loss' : D_loss, 'D_loss_real' : D_loss_real, 'D_loss_fake' : D_loss_fake}
    
    # training
    mbi = 0
    train_N = len(paths)
    train_ind = np.arange(train_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)

    for i in range(Iteration):
        if mbi + Batchsize > train_N:
            mb_ind = copy(train_ind[mbi:])
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(Batchsize - (train_N - mbi))]))
            mbi = Batchsize - (train_N - mbi)
        else:
            mb_ind = train_ind[mbi : mbi + Batchsize]
            mbi += Batchsize
        
        Xs = get_image(paths[mb_ind])
        Xs_target = get_image(paths_gt[mb_ind], gt=True)
        
        loss_dict = train_step(Xs, Xs_target)
        
        print('|', end='')

        if (i + 1) % 10 == 0:
            print(i + 1, end='')
        
        if (i + 1) % 50 == 0:
            print('\r' + ' ' * 100, end='')
            print('\riter : {} , G_Loss : {:.4f} (fake : {:.4f} , L1 : {:.4f}) , D_Loss : {:.4f} (fake : {:.4f} , real : {:.4f})'.format(
                i + 1, loss_dict['G_loss'], loss_dict['G_loss_fake'], loss_dict['G_loss_L1'], loss_dict['D_loss'], loss_dict['D_loss_fake'], loss_dict['D_loss_real']))

    G.save_weights(model_path_G)
    D.save_weights(model_path_D)

    
# test
def test():
    G = UNet()
    G.load_weights(model_path_G)

    paths, paths_gt = data_load('../Dataset/test/images/', hf=False, vf=False, rot=False)

    for i in range(len(paths)):
        path = paths[[i]]
        path_gt = paths_gt[[i]]
        x = get_image(path)
        
        pred = G(x)[0].numpy()

        pred = pred.argmax(axis=-1)

        # visualize
        out = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        for i, (label_name, v) in enumerate(class_label.items()):
            out[pred == i] = v

        print("in {}".format(path['path']))
        
        plt.subplot(1, 2, 1)
        plt.imshow(((x[0] + 1) / 2).astype(np.float32))
        plt.subplot(1, 2, 2)
        plt.imshow(out[..., ::-1])
        plt.show()
    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args


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
