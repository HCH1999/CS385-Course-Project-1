import os
import numpy as np
from scipy.io import loadmat

def load_data():
    train_filepath = './data/train_32x32.mat'
    test_filepath = './data/test_32x32.mat'

    if not os.path.isfile(train_filepath):
        raise ValueError('Training set missed')
    if not os.path.isfile(test_filepath):
        raise ValueError('Test set missed')

    train_set = loadmat(train_filepath)
    test_set = loadmat(test_filepath)
    train_X = train_set['X']
    train_Y = train_set['y']
    train_X = train_X.transpose((3, 0, 1, 2))
    print(train_X.shape)
    print(train_Y.shape)

def main():
    load_data()
    return

if __name__ == '__main__':
    main()

