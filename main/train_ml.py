import os
import argparse
import time
import sys
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils.averager import AverageMeter
from utils.accuracy import accuracy
from utils.logger import Logger
from utils.bar import Bar
from models.logistic import *

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='logistic', type=str)
parser.add_argument('--datapath', default='./data/', type=str)
parser.add_argument('--train_batchsize', default=64, type=int)
parser.add_argument('--val_batchsize', default=16, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--checkpoint', default='./checkpoints/', type=str)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--gpu-id', default='0', type=str)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--schedule', type=int, nargs='+', default=[70, 120, 160])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--reg', type=float, default=0.0001)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def main():
    
    # Prepare dataset
    traindir = args.datapath + '/train/'
    valdir = args.datapath + '/val/'
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = ImageFolder(traindir, train_transform)
    val_dataset = ImageFolder(valdir, val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batchsize, shuffle=True,
        num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.val_batchsize, shuffle=False,
        num_workers=4, pin_memory=True)

    # Prepare the model & loss & optimizer
    if args.arch == 'logistic' or args.arch == 'ridge' or args.arch == 'lasso':
        model = LogisticRegression()
    elif args.arch == 'kernel':
        model = KernelRegression()
    else:
        raise NotImplementedError("Arch {} is not implemented.".format(args.arch))
    model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
        momentum=args.momentum)
    
    # Resume & Create Log
    best_acc = 0.0
    start_epoch = 0
    checkpoint_path = os.path.join(args.checkpoint, args.arch + '/')
    logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=args.arch)
    logger.set_args(state)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss',
        'Train Acc', 'Valid Acc'])

    # Evaluate Branch
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_prec=validate(val_loader, model, criterion)
        print(' Test Loss:  %.8f, Test Acc Top1:  %.2f' % (test_loss, test_prec))
        return

    # Visualization
    writer = SummaryWriter(os.path.join(checkpoint_path, 'logs'))

    # Train
    for epoch in range(start_epoch, args.epochs):
        # adjust lr
        adjust_learning_rate(optimizer, epoch)
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion)

        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, val_acc])

        # tensorboardX
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {'train accuracy': train_acc, 'validation accuracy': val_acc}, epoch + 1)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)

        is_best = val_acc > best_acc
        print(is_best)
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, epoch, is_best, checkpoint=checkpoint_path)

    return

def train(train_loader, model, criterion, optimizer, epoch):
    bar = Bar('Processing', max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (image, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
        output = model(image)
        loss = criterion(output, target)

        # regularization
        reg_loss = 0.0
        if args.arch == 'lasso':
            for param in model.parameters():
                reg_loss += torch.sum(abs(param))
            reg_loss = reg_loss * args.reg
        elif args.arch == 'ridge':
            for param in model.parameters():
                reg_loss += torch.sum(param**2)
            reg_loss = reg_loss * args.reg
        loss += reg_loss

        # measure accuracy and record loss
        prec = accuracy(output.data, target.data, topk=(1,))
        losses.update(loss.item(), image.size(0))
        acc.update(prec[0], image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '(Epoch: {} | {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | prec: {prec: .4f}'.format(
                    epoch,
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    prec=acc.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, acc.avg)

def validate(val_loader, model, criterion):
    bar = Bar('Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to val mode
    model.eval()

    end = time.time()
    for i, (image, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
        output = model(image)
        loss = criterion(output, target)

        # regularization
        reg_loss = 0.0
        if args.arch == 'lasso':
            for param in model.parameters():
                reg_loss += torch.sum(abs(param))
            reg_loss = reg_loss * args.reg
        elif args.arch == 'ridge':
            for param in model.parameters():
                reg_loss += torch.sum(param**2)
            reg_loss = reg_loss * args.reg
        loss += reg_loss

        # measure accuracy and record loss
        prec = accuracy(output.data, target.data, topk=(1,))
        losses.update(loss.item(), image.size(0))
        acc.update(prec[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | prec: {prec: .4f}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    prec=acc.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, acc.avg)

    
def save_checkpoint(state, epoch, is_best, checkpoint='checkpoints'):
    if is_best:
        print("==>Achieve best acc!")
        filename = 'model_best.pth.tar'
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()