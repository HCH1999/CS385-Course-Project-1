import os
from skimage import io
import torchvision as tv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.io import loadmat

def Cifar100(root, objective):
    train_character = [[] for i in range(10)]
    test_character = [[] for i in range(10)]

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
    test_X = test_set['X']
    test_Y = test_set['y']
    test_X = test_X.transpose((3, 0, 1, 2))
    print(test_X.shape)

    for i in range(train_Y.shape[0]):
        if train_Y[i, 0] == 10:
            train_Y[i, 0] = 0
    for i in range(test_Y.shape[0]):
        if test_Y[i, 0] == 10:
            test_Y[i, 0] = 0

    for i in range(train_X.shape[0]):  
        train_character[train_Y[i, 0]].append(train_X[i, :, :, :])
    for i in range(test_X.shape[0]):  
        test_character[test_Y[i, 0]].append(test_X[i, :, :, :])


    meta_training = [[] for i in range(10)]
    meta_validation = [[] for i in range(10)]

    for idx, cls in enumerate(train_character):
        for i in range(len(cls)):
            meta_training[idx].append(cls[i])
    for idx, cls in enumerate(test_character):
        for i in range(len(cls)):
            meta_validation[idx].append(cls[i])

        
    print(len(meta_training[0]))

    character = []

    os.mkdir(os.path.join(objective, 'train'))
    for i, per_class in enumerate(meta_training):
        character_path = os.path.join(objective, 'train', str(i))
        os.mkdir(character_path)
        for j, img in enumerate(per_class):
            # print(img.shape)
            # img = img.transpose((1, 2, 0))
            # img = img.numpy()
            img_path = character_path + '/' + str(j) + ".jpg"
            io.imsave(img_path, img)

    os.mkdir(os.path.join(objective, 'val'))
    for i, per_class in enumerate(meta_validation):
        character_path = os.path.join(objective, 'val', str(i))
        os.mkdir(character_path)
        for j, img in enumerate(per_class):
            # img = img[0]
            # img = img.transpose((1, 2, 0))
            # img = img.numpy()
            img_path = character_path + '/' + str(j) + ".jpg"
            io.imsave(img_path, img)


if __name__ == '__main__':
    root = './data'
    objective = './data'
    Cifar100(root, objective)
    print("-----------------")