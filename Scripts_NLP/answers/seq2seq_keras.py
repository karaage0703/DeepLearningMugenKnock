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
from keras.layers import Dense, Input, SimpleRNN, LSTM


_chars = "あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポァィゥェォャュョッー、。「」1234567890!?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.@#"
chars = [c for c in _chars]
num_classes = len(chars)

d_num = 1024

def Encoder():
    enc_in = Input([None, num_classes], name='enc_in')
    enc_out, esh, esc = LSTM(d_num, return_state=True, name='enc')(enc_in)
    return enc_in, enc_out, esh, esc
    

def Decoder(dec_sh, dec_sc):
    dec_in = Input([None, num_classes], name='dec_in')
    dec_out, dsh , dsc = LSTM(d_num, return_sequences=True, return_state=True, name='dec')(dec_in, initial_state=[dec_sh, dec_sc])
    #dec_out = Dense(256, activation='sigmoid')(dec_out)
    dec_out = Dense(num_classes, activation='softmax', name='dec_out')(dec_out)
    return dec_in, dec_out, dsh, dsc

    
def data_load():
    fname = 'sandwitchman.txt'
    onehots = []
    txts = []
    max_len = 0
    
    with open(fname, 'r') as f:
        for l in f.readlines():
            txt = '@' + l.strip() + '@'
            txts.append(txt)
            max_len = max(max_len, len(txt))

    for txt in txts:
        onehot = [[0 for _ in range(num_classes)] for _ in range(max_len)]
        for i, c in enumerate(txt):
            onehot[i][chars.index(c)] = 1
        onehots.append(onehot)

    onehots = np.array(onehots)

    # enc_xs, dec_xs, ts ... [batch, time, num_classes]    
    enc_xs = np.array([v for v in onehots[:-1]])
    dec_xs = np.array([v for v in onehots[1:]])
    ts = np.zeros_like(dec_xs)
    ts[:, :-1] = onehots[1:, 1:]

    return enc_xs, dec_xs, ts


# train
def train():
    # model
    enc_in, enc_out, esh, esc = Encoder()
    dec_in, dec_out, dsh, dsc = Decoder(esh, esc)
    model = Model(inputs=[enc_in, dec_in], outputs=[dec_out])
    

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(lr=0.01),
        metrics=['accuracy'])

    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    enc_xs, dec_xs, ts = data_load()
    
    # training
    mb = 32
    mbi = 0
    train_ind = np.arange(len(enc_xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for ite in range(500):
        if mbi + mb > len(enc_xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(enc_xs)-mbi))]))
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        enc_x = enc_xs[mb_ind]
        dec_x = dec_xs[mb_ind]
        t = ts[mb_ind]

        loss, acc = model.train_on_batch(x={'enc_in': enc_x, 'dec_in': dec_x}, y=t)
        print("iter >>", ite+1, ",loss >>", loss, ',accuracy >>', acc)

    model.save('model.h5')

# test
def test():
    # load trained model
    enc_in, enc_out, esh, esc = Encoder()
    model_encoder = Model(inputs=[enc_in], outputs=[esh, esc])
    model_encoder.load_weights('model.h5', by_name=True)

    dec_sh = Input([d_num], name='dec_state_h')
    dec_sc = Input([d_num], name='dec_state_c')
    dec_in, dec_out, dsh, dsc = Decoder(dec_sh, dec_sc)
    model_decoder = Model(inputs=[dec_in, dec_sh, dec_sc], outputs=[dec_out, dsh, dsc])
    model_decoder.load_weights('model.h5', by_name=True)

    
    def decode(x):
        return chars[x.argmax()]
    
    encs = '@ちょっとなにいってるかわからない@'

    enc_x = []
    for enc in encs:
        onehot = [0 for _ in range(num_classes)]
        onehot[chars.index(enc)] = 1
        enc_x.append(onehot)
    enc_x = np.expand_dims(np.array(enc_x), axis=0)

    dec_state_h, dec_state_c = model_encoder.predict_on_batch(x={'enc_in':enc_x})
    
    pred = 0
    count = 0

    dec_x = np.zeros((1, 1, num_classes))
    dec_x[..., chars.index('@')] = 1
    
    gens = ''
    
    while pred != '@' and count < 1000:
        
        pred, dec_state_h, dec_state_c = model_decoder.predict_on_batch(
            x={'dec_in': dec_x, 'dec_state_h':dec_state_h, 'dec_state_c':dec_state_c})

        pred = pred[0,0]
        
        # sample random from probability distribution
        ind = np.random.choice(num_classes, 1, p=pred)
        
        pred = chars[ind[0]]
        gens += pred
        count += 1
        
        dec_x = np.zeros((1, 1, num_classes))
        dec_x[..., ind] = 1

    # pose process
    gens = gens.replace('@', '')
        
    print('--\ngenerated')
    print('[In]Speaker.A: >>', encs.replace('@', ''))
    print('[Out]Speaker.B: >>', gens)
    

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
