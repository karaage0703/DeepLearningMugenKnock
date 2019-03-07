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
            
            ts.append(t)

            paths.append(path)

    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)
    
    xs = xs.transpose(0,3,1,2)

    return xs, ts, paths

xs, ts, paths = data_load("../Dataset/train/images/")
