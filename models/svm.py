import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.averager import AverageMeter
from utils.accuracy import accuracy
from utils.logger import Logger
from tensorboardX import SummaryWriter


class SVM_OVR(nn.Module):

    def __init__(self, cls_num=10, datapath='./data/'):
        super(SVM_OVR, self).__init__()
        self.train_path = os.path.join(datapath, 'train_32x32.mat')
        self.test_path = os.path.join(datapath, 'test_32x32.mat')
        self.cls_num = cls_num

        train_set = loadmat(self.train_path)
        train_X = train_set['X']
        train_X = train_X.transpose((3, 0, 1, 2))
        train_X = train_X[0:500, :, :, :]
        train_X = train_X/float(255)
        train_X_mean = np.mean(train_X, axis=(0, 1, 2), keepdims=True)
        train_X_std = np.std(train_X, axis=(0, 1, 2), keepdims=True)

        train_X = (train_X - train_X_mean)/train_X_std

        self.x = torch.tensor(train_X, requires_grad=False, dtype=torch.float32)
        self.x = self.x.reshape(self.x.shape[0], -1).cuda()
        # self.x = self.x.cuda()

        #self.alpha = torch.randn(self.x.shape[0], self.cls, requires_grad=True).cuda()
        self.alpha = nn.Linear(self.x.shape[0], 64)
        self.classifier = nn.Linear(64, self.cls_num)
        self.softmax = nn.Softmax()
        self.norm = nn.BatchNorm1d(64)

        
    def kernel(self, x1, x2, type='linear'):
        if type == 'linear':
            return torch.dot(x1, x2)
        elif type == 'rbf':
            return 0
        else:
            raise NotImplementedError("Not implemented kernel: {}".format(type))


    def forward(self, x):
        x = x.reshape(x.shape[0], -1).cuda()
        F = x.shape[1]
        kernel = torch.matmul(x, self.x.T)/float(F)
        # print("kernel:{}".format(kernel[0, 0]))
        res = self.alpha(kernel)
        res = self.norm(res)
        res = self.softmax(res)
        res = self.classifier(res)
        return res

        
