import os
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat


class SVM_OVR(nn.Module):

    def __init__(self, cls=10, datapath='./data/'):
        self.train_path = os.path.join(datapath, 'train_32x32.mat')
        self.test_path = os.path.join(datapath, 'test_32x32.mat')
        self.cls = cls

        train_set = loadmat(self.train_path)
        train_X = train_set['X']
        train_Y = train_set['y']
        train_X = train_X.transpose((3, 0, 1, 2))
        self.x = torch.tensor(train_X, requires_grad=False).view(self.x.shape[0], -1)
        self.y = torch.tensor(train_Y, requires_grad=False).squeeze()

        self.alpha = torch.randn(self.x.shape[0], self.cls, requires_grad=True)
        
    def kernel(self, x1, x2, type='linear'):
        if type == 'linear':
            return torch.dot(x1, x2)
        elif type == 'rbf':
            return 0
        else:
            raise NotImplementedError("Not implemented kernel: {}".format(type))


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        res = torch.matmul(torch.matmul(x, self.x.T), self.alpha)
        return res

    def loss(self, x, y):

        batchsize = x.shape[0]
        # y-mask: [N, 10]
        mask = (-1) * torch.ones((batchsize, self.cls), requires_grad=False)
        mask[y] = 1
        x = x * mask
        loss = x.sum()
        return loss