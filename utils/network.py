#!/usr/bin/python2.7

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# the neural network
class Net(nn.Module):

    def __init__(self, input_dim, n_classes):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(input_dim,2048)
     

    def _forward2(self, x):
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
    
        return x
    
    def forward(self, x):        
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True)
        normalized_data = (x - means) / (stds  +  1e-16)     
        x = self._forward2(normalized_data)
        return x

# scorer used for inference
class Scorer(object):

    def __init__(self, net, features, max_len):
        self.net = net
        features = np.transpose(features)
        features = np.cumsum(features, axis = 0)
        self.features = features
        self.max_len = max_len
        self.cache = dict()

    def get(self, t_start, t_end, label):
        # t_start and t_end are inclusive
        if t_end - t_start + 1 > self.max_len:
            return -np.inf
        elif (t_start, t_end) in self.cache.keys():
            return self.cache[(t_start, t_end)][label]
        else:
            data = self.features[t_end, :]
            if t_start > 0:
                data = (self.features[t_end, :] - self.features[t_start-1, :]) / (t_end - t_start + 1)
            else:
                data = self.features[t_end, :] / (t_end + 1)
            data = torch.FloatTensor(data)
            with torch.no_grad():
                scores = self.net._forward2(data.unsqueeze(0)).squeeze(0).numpy()
            scores = scores * (t_end - t_start + 1)
            self.cache[(t_start, t_end)] = scores
            return scores[label]

    def clear(self):
        self.cache.clear()

    def length(self):
        return self.features.shape[0]

