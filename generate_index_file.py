# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:59:16 2017

@author: lhd
"""

import numpy as np
import scipy.io as spio
train_size = 1200
test_size = 800

mall_gt = spio.loadmat('mall_dataset/mall_gt.mat')
count = mall_gt['count']
with open('train.txt', 'w') as f:
    for i in range(train_size):
        seq = 'seq_%06d' % (i+1)
        train_input = 'mall_dataset/frames/' + seq + '.jpg'
        seqd = '%d' % (i+1)
        train_dmap = 'mall_dataset/dmap/dmap_' + seqd + '.mat'
        #dmap = spio.loadmat(train_dmap)
        cnt_i = count[i]
        wstr = train_input + '    ' + train_dmap + '    ' + '%d'%(cnt_i) + '\n'
        f.write(wstr);
        
with open('test.txt', 'w') as f:
    for i in range(test_size):
        index = i + 1201
        seq = 'seq_%06d' % (index)
        train_input = 'mall_dataset/frames/' + seq + '.jpg'
        seqd = '%d' % (index)
        train_dmap = 'mall_dataset/dmap/dmap_' + seqd + '.mat'
        #dmap = spio.loadmat(train_dmap)
        cnt_i = count[index - 1]
        wstr = train_input + '    ' + train_dmap + '    ' + '%d'%(cnt_i) + '\n'
        f.write(wstr);