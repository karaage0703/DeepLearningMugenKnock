import numpy as np

import os
import pickle

labels = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

def load_cifar10():

    path = 'cifar-10-batches-py'

    if not os.path.exists(path):
        os.system("wget {}".format('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'))
        os.system("tar xvf {}".format('cifar-10-python.tar.gz'))


load_cifar10()
