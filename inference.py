#!/usr/bin/python3

import numpy as np
import multiprocessing as mp
import queue
import torch
from utils.dataset import Dataset
from utils.network import Net, Scorer
from utils.grammar import NGram
from utils.length_model import MeanLengthModel
from utils.viterbi import Viterbi


### helper function for parallelized Viterbi decoding ##########################
def decode(q, decoder, index2label, dataset):
    while not q.empty():
        try:
            video = q.get(timeout = 3)
            net = Net(dataset.input_dimension, dataset.n_classes)
            net.load_state_dict( torch.load('results/nn.net') )
            scorer = Scorer(net, dataset.get_video(video)[0], decoder.length_model.max_length())
            score, labels, segments = decoder.decode(scorer)
            # save result
            with open('results/' + video, 'w') as f:
                f.write( '### Score: ###\n' + str(score) + '\n')
                f.write( '### Recognized sequence: ###\n' )
                f.write( ' '.join( [str(s.label) + ':' + str(s.length) + ':' + str(s.score)  for s in segments] ) + '\n' )
        except queue.Empty:
            pass


### read label2index mapping and index2label mapping ###########################
label2index = dict()
index2label = dict()
with open('data/mapping.txt', 'r') as f:
    content = f.read().split('\n')[0:-1]
    for line in content:
        label2index[line.split()[1]] = int(line.split()[0])
        index2label[int(line.split()[0])] = line.split()[1]

### read test data #############################################################
print('read data...')
with open('data/split1.test', 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = Dataset('data', video_list, label2index, shuffle = False)
print('done')

# length model, grammar, and network
grammar = NGram('results/grammar.txt', label2index, ngram_order = 3)
length_model = MeanLengthModel(dataset.n_classes, max_length = 500, threshold = 200.0)

# parallelization
n_threads = 8

# Viterbi decoder
viterbi_decoder = Viterbi(grammar, length_model, frame_sampling = 10, pruning_factor = 0.98, max_segment_start_hyp = 20)


# Viterbi decoding
q = mp.Queue()
for i, data in enumerate(dataset.features):
    video = list(dataset.features.keys())[i]
    q.put(video)
procs = []
for i in range(n_threads):
    p = mp.Process(target = decode, args = (q, viterbi_decoder, index2label, dataset) )
    procs.append(p)
    p.start()
for p in procs:
    p.join()

