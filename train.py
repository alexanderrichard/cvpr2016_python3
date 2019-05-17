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

### read labels of training data ###############################################
print('read labels...')
with open('data/split1.train', 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = TranscriptDataset('data', video_list, label2index)
print('done')

### generate grammar file for inference ########################################
paths = set()
for transcript in dataset.transcripts:
    paths.add( ' '.join([index2label[index] for index in transcript]) )
with open('results/grammar.txt', 'w') as f:
    f.write('\n'.join(paths) + '\n')

### estimate segment prior #######################################################################
# Since the outputs of the neural networ are posterior robabiliteis the authors follow the       #
# hybrid approach presented in [3]. During the training the author count the amount of frames    # 
# that have been labeled witha a class c for all sequences that have been processed so far.      #
# Normalizing these counts to sum up to one  finally results in our estimate of p(c). The priori #
# is updated after every iteration, i.e. after every new sequence. if a sequence c1^N contains   #
# a classs that has not been seen before ? is used.
prior = np.zeros((dataset.n_classes,), dtype = np.float32)
for transcript in dataset.transcripts:
    for label in transcript:
        prior[label] += 1
prior = prior / np.sum(prior)
np.save('results/prior.npy', prior)

### generate length model for inference ####################################################
# In this work the authors focus on the problem of weakly supervised learning and propose  #
# two contributons. The first contribution addresses the modelling of p(c1^N,l1^N,x1^N).   #
# Instead of using a HMM they explicit model the lenght of each each action class.         #
# As a lenght model, the authors use a class-dependent Poisson distribution. After each    # 
# iteration the authors update the mean lenght of a segment for a class c. if the training # 
# sample (x1^T, c1^N)  contains a classs that has not seen before we set th lenght to be   #
# equals to N/T                                                                            #

length_model = np.zeros((dataset.n_classes,), dtype = np.float32)
length_count = np.zeros((dataset.n_classes,), dtype = np.uint32)
for labels in dataset.labels:
    for l in labels:
        length_model[l] += 1
for transcripts in dataset.transcripts:
    for l in transcripts:
        length_count[l] += 1
length_model /= length_count
np.save('results/length_model.npy', length_model)


### read segmented training data ###############################################
print('read data...')
with open('data/split1.train', 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = SegmentDataset('data', video_list, label2index, shuffle = True)
print('done')

### train log-linear modelfor segment classification ###########################
net = Net(dataset.input_dimension, dataset.n_classes)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)


for epoch in range(25):
    running_loss = 0.0
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = torch.nn.functional.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('epoch-%d: %.4f' % (epoch+1, running_loss / len(dataloader)))

net.cpu()
torch.save(net.state_dict(), 'results/nn.net')

### read segmented test data ###################################################
print('read data...')
with open('data/split1.test', 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = SegmentDataset('data', video_list, label2index, shuffle = True)
print('done')

### report segment accuracy on test set ########################################
net.cuda()
dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)

n_correct = 0
for data in dataloader:
    inputs, labels = data
    inputs = inputs.cuda()
    labels = labels.cuda()
    outputs = net(inputs)
    prediction = torch.max(outputs, dim = 1)[1]
    n_correct += torch.sum( prediction == labels )

print('Accuracy: %.4f' % (float(n_correct) / len(dataset)))




'''
### read segmented training data ###############################################
print('read data...')
with open('data/trainset', 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = SegmentDataset('data', video_list, label2index, shuffle = True)
print('done')

### train SVM for segment classification and convert to linear layer ###########
data = np.vstack(dataset.features)
labels = np.array(dataset.labels, dtype = np.uint32)
weights = np.zeros((dataset.input_dimension, dataset.n_classes), dtype = np.float64)
bias = np.zeros((dataset.n_classes,), dtype = np.float64)
# train n_classes one-vs-rest SVMs
for c in range(dataset.n_classes):
    print('train svm for class %d' % c)
    idx = np.argwhere(labels == c)[:, 0]
    Y = np.ones(labels.shape, dtype = np.float64)
    Y[idx] = 0
    svm = SVC(kernel = 'linear', probability = True)
    svm.fit(data, Y)
    coefs = svm.dual_coef_[0, :]
    weights[:, c] += np.sum( svm.support_vectors_ * coefs[:, None], axis = 0 ) * svm.probA_
    bias[c] = svm.intercept_ * svm.probA_ + svm.probB_
np.save('weights.npy', weights)
np.save('bias.npy', bias)
# save weights and bias as neural network
net = Net(dataset.input_dimension, dataset.n_classes)
weights = torch.Tensor(np.transpose(weights))
bias = torch.Tensor(bias)
net.fc.weight = weights
net.fc.bias = bias
torch.save(net.state_dict(), 'results/nn.net')
'''

