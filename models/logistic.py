import torch
import torch.nn as nn
import torch.nn.functional as F 

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

class KernelRegression(nn.Module):
    def __init__(self, cls_num=10, in_feat=3*32*32, hidden_layer=64):
        super(LogisticRegression, self).__init__()
