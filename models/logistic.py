import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from scipy.io import loadmat

class LogisticRegression(nn.Module):
    def __init__(self, cls_num=10, in_feat=3*32*32, hidden_layer=64):
        super(LogisticRegression, self).__init__()
        self.input_layer = nn.Linear(in_feat, hidden_layer)
        self.hidden_layer = nn.Linear(hidden_layer, hidden_layer)
        self.classifier = nn.Linear(hidden_layer, cls_num)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.classifier(x)
        softmax = self.softmax(x)
        return softmax

class WideLogisticRegression(nn.Module):
    def __init__(self, cls_num=10, in_feat=3*32*32, hidden_layer=128):
        super(WideLogisticRegression, self).__init__()
        # type 1: 128, 
        self.input_layer = nn.Linear(in_feat, hidden_layer)
        self.classifier = nn.Linear(hidden_layer, cls_num)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.input_layer(x)
        x = self.classifier(x)
        softmax = self.softmax(x)
        return softmax

class DeepLogisticRegression(nn.Module):
    def __init__(self, cls_num=10, in_feat=3*32*32, hidden_layer=32):
        super(DeepLogisticRegression, self).__init__()
        # type 1: 128, 
        self.input_layer = nn.Linear(in_feat, hidden_layer)
        self.hidden_layer_0 = nn.Linear(hidden_layer, hidden_layer)
        self.hidden_layer_1 = nn.Linear(hidden_layer, hidden_layer)
        self.hidden_layer_2 = nn.Linear(hidden_layer, hidden_layer)
        self.classifier = nn.Linear(hidden_layer, cls_num)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.input_layer(x)
        x = self.hidden_layer_0(x)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.classifier(x)
        softmax = self.softmax(x)
        return softmax

class LogisticRegressionReLU(nn.Module):
    def __init__(self, cls_num=10, in_feat=3*32*32, hidden_layer=64):
        super(LogisticRegressionReLU, self).__init__()
        self.input_layer = nn.Linear(in_feat, hidden_layer)
        self.hidden_layer = nn.Linear(hidden_layer, hidden_layer)
        self.classifier = nn.Linear(hidden_layer, cls_num)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.input_layer(x)
        x = F.relu(self.hidden_layer(x))
        x = self.classifier(x)
        softmax = self.softmax(x)
        return softmax

class WideLogisticRegressionReLU(nn.Module):
    def __init__(self, cls_num=10, in_feat=3*32*32, hidden_layer=128):
        super(WideLogisticRegressionReLU, self).__init__()
        # type 1: 128, 
        self.input_layer = nn.Linear(in_feat, hidden_layer)
        self.classifier = nn.Linear(hidden_layer, cls_num)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.input_layer(x))
        x = self.classifier(x)
        softmax = self.softmax(x)
        return softmax

class DeepLogisticRegressionReLU(nn.Module):
    def __init__(self, cls_num=10, in_feat=3*32*32, hidden_layer=32):
        super(DeepLogisticRegressionReLU, self).__init__()
        # type 1: 128, 
        self.input_layer = nn.Linear(in_feat, hidden_layer)
        self.hidden_layer_0 = nn.Linear(hidden_layer, hidden_layer)
        self.hidden_layer_1 = nn.Linear(hidden_layer, hidden_layer)
        self.hidden_layer_2 = nn.Linear(hidden_layer, hidden_layer)
        self.classifier = nn.Linear(hidden_layer, cls_num)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.input_layer(x)
        x = self.hidden_layer_0(x)
        x = self.hidden_layer_1(x)
        x = F.relu(self.hidden_layer_2(x))
        x = self.classifier(x)
        softmax = self.softmax(x)
        return softmax

class DeepLogisticRegressionReLUPLUS(nn.Module):
    def __init__(self, cls_num=10, in_feat=3*32*32, hidden_layer=32):
        super(DeepLogisticRegressionReLUPLUS, self).__init__()
        # type 1: 128, 
        self.input_layer = nn.Linear(in_feat, hidden_layer)
        self.hidden_layer_0 = nn.Linear(hidden_layer, hidden_layer)
        self.hidden_layer_1 = nn.Linear(hidden_layer, hidden_layer)
        self.hidden_layer_2 = nn.Linear(hidden_layer, hidden_layer)
        self.classifier = nn.Linear(hidden_layer, cls_num)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.input_layer(x)
        x = F.relu(self.hidden_layer_0(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))
        x = self.classifier(x)
        softmax = self.softmax(x)
        return softmax

class KernelRegression(nn.Module):
    def __init__(self, cls_num=10, hidden_layer=64, kernel='linear'):
        super(KernelRegression, self).__init__()
        self.train_path = os.path.join('./data', 'train_32x32.mat')
        self.cls_num = cls_num

        train_set = loadmat(self.train_path)
        train_X = train_set['X']
        train_X = train_X.transpose((3, 2, 0, 1))
        train_X = train_X[0:500, :, :, :]
        train_X = train_X/float(255)
        print("x total shape:{}".format(train_X.shape))
        train_X = (train_X - 0.5)/0.5
        self.kernel = kernel

        self.x = torch.tensor(train_X, requires_grad=False, dtype=torch.float32)
        self.x = self.x.reshape(self.x.shape[0], -1).cuda()
        self.beta = torch.randn((self.x.shape[0], ), requires_grad=True)
        self.preprocess = nn.Linear(self.x.shape[1], hidden_layer)
        self.classifier = nn.Linear(self.x.shape[0], cls_num)
        self.softmax = nn.Softmax()

    def kernel(self, A, B):
        if self.kernel == 'linear':
            return torch.matmul(A, B)
        elif self.kernel == 'rbf':
            x1 = A.repeat(1, B.shape[0], 1)
            x2 = B.unsqueeze(0)
            rbf = torch.exp(self.beta.matmul(torch.pow(x1-x2, 2))).sum(dim=1, keepdims=False)
            return rbf
            

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        A = self.preprocess(x)
        B = self.preprocess(self.x)
        x = torch.matmul(x, self.x.T)
        x = self.classifier(x)
        softmax = self.softmax(x)
        return softmax
