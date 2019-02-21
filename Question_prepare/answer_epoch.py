import cv2
import numpy as np
from glob import glob

np.random.seed(0)

num_classes = 2
img_height, img_width = 64, 64

CLS = ['akahara', 'madara']

# get train data
def data_load(path):
    xs = []
    ts = []
    paths = []
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs.append(x)

            for i, cls in enumerate(CLS):
                if cls in path:
                    t = i

            paths.append(path)

    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.float32)
            
    xs = xs.transpose(0,3,1,2)

    return xs, ts, paths


xs, ts, paths = data_load('../Dataset/train/images/')

mb = 3
mbi = 0
train_ind = np.arange(len(xs))
np.random.seed(0)
np.random.shuffle(train_ind)

epoch_max = 3
epoch = 0

while epoch < epoch_max:
    if mbi + mb > len(xs):
        mb_ind = train_ind[mbi:]
        np.random.shuffle(train_ind)
        mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
        epoch += 1
        mbi = mb - (len(xs) - mbi)
    else:
        mb_ind = train_ind[mbi: mbi+mb]
        mbi += mb

    print(mb_ind)
