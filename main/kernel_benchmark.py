import os
import argparse
import cv2
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='vgg', type=str)
parser.add_argument('--checkpoint', default='./checkpoints/', type=str)
parser.add_argument('--gpu-id', default='1', type=str)
parser.add_argument('--inputdir', default='./grad_cam/input/', type=str)
parser.add_argument('--outputdir', default='./grad_cam/output/', type=str)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def kernel(A, B, kernel='rbf'):
    if kernel == 'linear':
        # print(A[0])
        # print(B[0])
        # print("sum:{}".format(torch.sum(A - B)))
        return torch.abs(torch.sum(A - B))
    elif kernel == 'rbf':
        rbf = torch.exp(torch.pow(A-B, 2)).sum()
        return rbf
    else:
        return 0

def main():    

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    map_lin = torch.zeros((10, 10), dtype=float)
    map_rbf = torch.zeros((10, 10), dtype=float)

    for cls_id, cls_path in enumerate(os.listdir(args.inputdir)):
        cls_path_abs = os.path.join(args.inputdir, cls_path)
        for img_path in os.listdir(cls_path_abs):
            # load the image
            img_path_abs = os.path.join(cls_path_abs, img_path)
            img = cv2.imread(img_path_abs)
            img = np.float32(img) / 255 # 0-255 to 0-1
            img = img[:, :, ::-1]   #BGR to RGB

            # generate heatmap
            #print(x.shape)
            x = transform(img.copy())
            x = x.unsqueeze(0)
            x = x.reshape(x.shape[0], -1).squeeze()

            for cls_id_, cls_path_ in enumerate(os.listdir(args.inputdir)):
                cls_path_abs_ = os.path.join(args.inputdir, cls_path_)
                for img_path_ in os.listdir(cls_path_abs_):
                    # load the image
                    img_path_abs_ = os.path.join(cls_path_abs_, img_path_)
                    img_ = cv2.imread(img_path_abs_)
                    img_ = np.float32(img_) / 255 # 0-255 to 0-1
                    img_ = img_[:, :, ::-1]   #BGR to RGB

                    # generate heatmap
                    #print(x.shape)
                    x_ = transform(img_.copy())
                    x_ = x_.unsqueeze(0)
                    x_ = x_.reshape(x_.shape[0], -1).squeeze()

                    
                    # compute kernel value
                    ker_lin = kernel(x, x_, 'linear')
                    ker_rbf = kernel(x, x_, 'rbf')
                    map_lin[cls_id, cls_id_] += ker_lin
                    map_lin[cls_id_, cls_id] += ker_lin
                    map_rbf[cls_id, cls_id_] += ker_rbf
                    map_rbf[cls_id_, cls_id] += ker_rbf


    map_lin = map_lin.numpy()/float(10)
    map_lin = (map_lin-np.mean(map_lin))/np.std(map_lin)
    map_lin = np.around(map_lin, 2)
    print(map_lin)
    map_rbf = map_rbf.numpy()/float(10)
    map_rbf = (map_rbf-np.mean(map_rbf))/np.std(map_rbf)
    map_rbf = np.around(map_rbf, 2)
    print(map_rbf)
    lin = pd.DataFrame(map_lin)
    rbf = pd.DataFrame(map_rbf)
    lin.to_csv("./data/lin.csv", index=False, sep=',')
    rbf.to_csv("./data/rbf.csv", index=False, sep=',')

    
    return

if __name__ == '__main__':
    main()