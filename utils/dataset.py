#!/usr/bin/python3

import numpy as np
import random

# self.features[video]: the feature array of the given video (dimension x frames)
# self.transcrip[video]: the transcript (as label indices) for each video
# self.input_dimension: dimension of video features
# self.n_classes: number of classes
class Dataset(object):

    def __init__(self, base_path, video_list, label2index, shuffle = False):
        self.features = dict()
        self.transcript = dict()
        self.shuffle = shuffle
        self.idx = 0
        # read features for each video
        for video in video_list:
            # video features
            self.features[video] = np.load(base_path + '/features/' + video + '.npy')
            # transcript
            with open(base_path + '/transcripts/' + video + '.txt') as f:
                self.transcript[video] = [ label2index[line] for line in f.read().split('\n')[0:-1] ]
        # selectors for random shuffling
        self.selectors = list(self.features.keys())
        if self.shuffle:
            random.shuffle(self.selectors)
        # set input dimension and number of classes
        self.input_dimension = list(self.features.values())[0].shape[0]
        self.n_classes = len(label2index)

    def videos(self):
        return self.features.keys()

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.selectors)
            raise StopIteration
        else:
            video = self.selectors[self.idx]
            self.idx += 1
            return self.features[video], self.transcript[video]

    def __getitem__(self, idx):
        video = self.selectors[idx]
        return self.features[video], self.transcript[video]

    def get_video(self, video):
        return self.features[video], self.transcript[video]

    def get(self):
        try:
            return self.__next__()
        except StopIteration:
            return self.get()


class SegmentDataset(object):

    def __init__(self, base_path, video_list, label2index, shuffle = False):
        self.features = []
        self.labels = []
        self.shuffle = shuffle
        self.idx = 0
        # read features for each video
        for video in video_list:
            segments, labels = self._read(base_path, video, label2index)
            self.features += segments
            self.labels += labels
        # selectors for random shuffling
        self.selectors = list(range(len(self.features)))
        if self.shuffle:
            random.shuffle(self.selectors)
        # set input dimension and number of classes
        self.input_dimension = self.features[0].shape[0]
        self.n_classes = len(label2index)

    def _read(self, base_path, video, label2index):
        # read framewise features and labels
        features = np.load(base_path + '/features/' + video + '.npy')
        with open(base_path + '/groundTruth/' + video + '.txt', 'r') as f:
            labels = f.read().split('\n')[0:-1]
            labels = [ label2index[l] for l in labels ]
        # determine segment boundaries
        segment_indices = []
        prev_label = labels[0]
        for i, l in enumerate(labels):
            if not prev_label == l:
                segment_indices += [i]
                prev_label = l
        segment_indices += [len(labels)]
        # compute segment features and labels
        segments = []
        segment_labels = []
        start = 0
        for i in segment_indices:
            segments += [ np.mean(features[:, start:i], axis = 1) ]
            segment_labels += [ labels[start] ]
            start = i
        return segments, segment_labels

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.selectors)
            raise StopIteration
        else:
            video = self.selectors[self.idx]
            self.idx += 1
            return self.features[video], self.labels[video]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class TranscriptDataset(object):

    def __init__(self, base_path, video_list, label2index):
        self.labels = []
        self.transcripts = []
        self.idx = 0
        # read labels and transcripts for each video
        for video in video_list:
            # labels 
            with open(base_path + '/groundTruth/' + video + '.txt') as f:
                self.labels.append( [ label2index[line] for line in f.read().split('\n')[0:-1] ] )
            # transcript
            with open(base_path + '/transcripts/' + video + '.txt') as f:
                self.transcripts.append( [ label2index[line] for line in f.read().split('\n')[0:-1] ] )
        self.n_classes = len(label2index)

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self):
            self.idx = 0
            raise StopIteration
        else:
            labels = self.labels[self.idx]
            transcript = self.transcripts[self.idx]
            self.idx += 1
            return labels, transcript

    def __getitem__(self, idx):
        return self.labels[self.idx], self.transcripts[self.idx]

