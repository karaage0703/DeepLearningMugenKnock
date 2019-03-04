import chainer
import chainer.links as L
import chainer.functions as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

num_classes = 2
img_height, img_width = 64, 64#572, 572
out_height, out_width = 64, 64#388, 388
GPU = -1
    
    
CLS = {'akahara': [0,0,128],
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

            t = np.zeros((out_height, out_width), dtype=np.int)

            for i, (_, vs) in enumerate(CLS.items()):
                ind = (gt[...,0] == vs[0]) * (gt[...,1] == vs[1]) * (gt[...,2] == vs[2])
                t[ind] = i + 1

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

    xs = xs.transpose(0,3,1,2)

    return xs, ts, paths

    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    args = arg_parse()

    #if args.train:
    #    train()
    #if args.test:
    #    test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
