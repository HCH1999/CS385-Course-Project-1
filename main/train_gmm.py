import os
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
import torch
from sklearn.svm import SVC
from scipy.io import loadmat
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import Logger

def main():
    train_path = os.path.join('./data', 'train_32x32.mat')
    val_path = os.path.join('./data', 'test_32x32.mat')

    logger = Logger(os.path.join('./checkpoints', 'log.txt'), title='svm')
    logger.set_names(['Mean', 'Param'])

    train_set = loadmat(train_path)
    val_set = loadmat(val_path)
    train_X = train_set['X']
    train_Y = train_set['y']
    val_X = val_set['X']
    val_Y = val_set['y']
    train_X = train_X.transpose((3, 2, 0, 1))
    train_X = train_X/float(255)
    train_X = (train_X - 0.5)/0.5
    val_X = val_X.transpose((3, 2, 0, 1))
    val_X = val_X/float(255)
    val_X = (val_X - 0.5)/0.5
    # print("train x shape:{}".format(train_X.shape))
    for i in range(train_Y.shape[0]):
        if train_Y[i, 0] == 10:
            train_Y[i, 0] = 0
    for i in range(val_Y.shape[0]):
        if val_Y[i, 0] == 10:
            val_Y[i, 0] = 0
    train_Y = train_Y.squeeze()
    val_Y = val_Y.squeeze()
    train_X = train_X.reshape(train_X.shape[0], -1)
    val_X = val_X.reshape(val_X.shape[0], -1)

    
    parameters = {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 'C': [0.1 ,1 ,3 ], 'gamma': [0.1, 1, 3]}
    svc = SVC()
    model = GridSearchCV(svc, parameters, cv=5, scoring='accuracy')
    model.fit(train_X, train_Y)
    model.score(val_X, val_Y)
    means = model.cv_results_['mean_test_score']
    params = model.cv_results_['params']
    mean_list=[]
    prarm_list=[]
    for mean, param in zip(means, params):
        mean_list.append(mean)
        prarm_list.append(params)
        logger.append([mean, param])
    torch.save({"mean":means, "params":params}, "./checkpoints/svm/model_best.pth.tar") 

if __name__ == '__main__':
    main()