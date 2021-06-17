import sys
import os
import numpy as np
import argparse
import json
import torch
import torch.nn as nn
from scipy.io import loadmat

from utils import AverageMeter, accuracy, Logger
from models import SVM_OVR
from tensorboardX import SummaryWriter
INF = 100

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--checkpoint', default='./checkpoints/', type=str)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--gpu-id', default='1', type=str)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def train(x, y, model, batchsize, optimizer):
    losses = AverageMeter()
    acc = AverageMeter()
    model.train()

    batch_num = x.shape[0] // batchsize
    for batch_id in range(batch_num):
        batch_x = x[batch_id*batchsize: (batch_id+1)*batchsize, :]
        batch_y = y[batch_id*batchsize: (batch_id+1)*batchsize]
        res = model(batch_x)
        loss = model.loss(res, batch_y)
        prec = accuracy(res.data, batch_y.data, batch_y.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), batchsize)

    # last batch
    batch_x = x[batch_id*batchsize:, :]
    batch_y = y[batch_id*batchsize:]
    res = model(batch_x)
    loss = model.loss(res, batch_y)
    prec = accuracy(res.data, batch_y.data, batch_y.shape[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc.update(prec[0], batchsize)
    losses.update(loss.item(), batchsize)

    return losses.avg, acc.avg

def validate(x, y, model, batchsize):
    losses = AverageMeter()
    acc = AverageMeter()
    model.eval()

    batch_num = x.shape[0] // batchsize
    for batch_id in range(batch_num):
        batch_x = x[batch_id*batchsize: (batch_id+1)*batchsize, :]
        batch_y = y[batch_id*batchsize: (batch_id+1)*batchsize]
        res = model(batch_x)
        loss = model.loss(res, batch_y)
        prec = accuracy(res.data, batch_y.data, batch_y.shape[0])

        acc.update(prec[0], batchsize)
        losses.update(loss.item(), batchsize)

    # last batch
    batch_x = x[batch_id*batchsize:, :]
    batch_y = y[batch_id*batchsize:]
    res = model(batch_x)
    loss = model.loss(res, batch_y)
    prec = accuracy(res.data, batch_y.data, batch_y.shape[0])
    acc.update(prec[0], batchsize)
    losses.update(loss.item(), batchsize)

    return losses.avg, acc.avg
    
def main():
    EPOCH = 50
    logger = Logger(os.path.join('./log/{}', 'log.txt'), title=args.arch)
    logger.set_args(state)
    logger.set_names(['Train Loss', 'Valid Loss',
        'Train Acc', 'Valid Acc'])

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

    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)

    model = SVM_OVR()
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
        momentum=0.9, weight_decay=1e-5)

    writer = SummaryWriter('./log/{}/tensorboardX'.format(args.arch))
    
    print("==>Training begins!")
    best_acc = 0.0
    for epoch in range(EPOCH):
        
        train_loss, train_acc = train(train_X, train_Y, model, 128, optimizer)
        val_loss, val_acc = validate(train_X, train_Y, model, 128,)
        print("==>Epoch:{}, Train Loss:{}, Val Loss:{}".format(epoch, train_loss, val_loss))
        logger.append([train_loss, val_loss, train_acc, val_acc])

        writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('acc', {'train acc': train_acc, 'validation acc': val_acc}, epoch + 1)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), './log/{}/model_best.pth.tar'.format(args.arch))

    return

if __name__ == "__main__":
    main()
            