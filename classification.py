#!/usr/bin/python2.7

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.network import Net
from utils.dataset import TranscriptDataset, SegmentDataset

### read label2index mapping and index2label mapping ###########################
label2index = dict()
index2label = dict()
with open('data/mapping.txt', 'r') as f:
    content = f.read().split('\n')[0:-1]
    for line in content:
        label2index[line.split()[1]] = int(line.split()[0])
        index2label[int(line.split()[0])] = line.split()[1]

### read segmented test data ###################################################
print('read data...')
with open('data/split1.train', 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = SegmentDataset('data', video_list, label2index, shuffle = True)
print('done')

### report segment accuracy on test set ########################################
net = Net(dataset.input_dimension, dataset.n_classes)
net.load_state_dict( torch.load('results/nn.net') )
net.cuda()
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

n_correct = 0
for data in dataloader:
    inputs, labels = data
    inputs = inputs.cuda()
    labels = labels.cuda()
    outputs = net(inputs.unsqueeze(0))
    prediction = torch.max(outputs, dim = 1)[1]
    n_correct += torch.sum( prediction == labels )

print('Accuracy: %.4f' % (float(n_correct) / len(dataset)))

