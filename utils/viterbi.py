#!/usr/bin/python2.7

import numpy as np
from .grammar import PathGrammar
from .length_model import PoissonModel
import glob
import re

# Viterbi decoding
class Viterbi(object):

    ### helper structure ###
    class TracebackNode(object):
        def __init__(self, label, predecessor, boundary = False):
            self.label = label
            self.predecessor = predecessor
            self.boundary = boundary

    ### helper structure ###
    class HypDict(dict):
        class Hypothesis(object):
            def __init__(self, score, traceback):
                self.score = score
                self.traceback = traceback
        def update(self, key, score, traceback):
            if (not key in self) or (self[key].score <= score):
                self[key] = self.Hypothesis(score, traceback)

    # @grammar: the grammar to use, must inherit from class Grammar
    # @length_model: the length model to use, must inherit from class LengthModel
    # @frame_sampling: generate hypotheses every frame_sampling frames
    # @pruning_factor: value between 0 (no pruning) and 1 (out of all new segment start hypotheses prune all but the best hypothesis)
    # @max_segment_start_hyp: maximal number of new segment start hypotheses per time frame
    def __init__(self, grammar, length_model, frame_sampling = 1, pruning_factor = 0.0, max_segment_start_hyp = np.inf):
        self.grammar = grammar
        self.length_model = length_model
        self.frame_sampling = frame_sampling
        self.pruning_factor = pruning_factor
        self.max_segment_start_hyp = max_segment_start_hyp

    # Viterbi decoding of a sequence
    # @log_frame_probs: logarithmized frame probabilities
    #                   (usually log(network_output) - log(prior) - max_val, where max_val ensures negativity of all log scores)
    # @return: the score of the best sequence,
    #          the corresponding framewise labels (len(labels) = len(sequence))
    #          and the inferred segments in the form (label, length)
    def decode(self, scorer):
        # create initial hypotheses
        hyps = self.init_decoding()
        # decode each following time step
        for t in range(self.frame_sampling, scorer.length(), self.frame_sampling):
            hyps = self.decode_frame(t, hyps, scorer)
            scorer.clear()
        # transition to end symbol
        final_hyp = self.finalize_decoding(hyps, scorer)
        scorer.clear()
        labels, segments = self.traceback(final_hyp, scorer.length(), scorer)
        return final_hyp.score, labels, segments

    def prune(self, hyps, new_segment_keys):
        if self.pruning_factor > 0 and len(new_segment_keys) > 0:
            scores = sorted( [ hyps[key].score for key in new_segment_keys ] )
            reference_idx = min(int(self.pruning_factor * len(scores)), len(scores) - 1)
            reference_idx = max(reference_idx, len(scores) - self.max_segment_start_hyp)
            for key in new_segment_keys:
                if hyps[key].score < scores[reference_idx]:
                    del hyps[key]

    def init_decoding(self):
        hyps = self.HypDict()
        context = self.grammar.update_context((), self.grammar.start_symbol())
        for label in self.grammar.possible_successors(context):
            key = context + (label, self.frame_sampling)
            score = self.grammar.score(context, label)
            hyps.update(key, score, self.TracebackNode(label, None, boundary = True))
        return hyps

    def decode_frame(self, t, old_hyp, scorer):
        new_hyp = self.HypDict()
        new_segment_keys = set()
        for key, hyp in old_hyp.items():
            context, label, length = key[0:-2], key[-2], key[-1]
            # stay in the same label...
            if length + self.frame_sampling <= self.length_model.max_length():
                new_key = context + (label, length + self.frame_sampling)
                new_hyp.update(new_key, hyp.score, self.TracebackNode(label, hyp.traceback, boundary = False))
            # ... or go to the next label
            context = self.grammar.update_context(context, label)
            for new_label in self.grammar.possible_successors(context):
                if new_label == self.grammar.end_symbol():
                    continue
                new_key = context + (new_label, self.frame_sampling)
                score = hyp.score + scorer.get(t-length, t-1, label) + self.length_model.score(length, label) + self.grammar.score(context, new_label)
                new_hyp.update(new_key, score, self.TracebackNode(new_label, hyp.traceback, boundary = True))
                new_segment_keys.add(new_key)
        # prune among new hypothesized segments
        self.prune(new_hyp, new_segment_keys)
        # return new hypotheses
        return new_hyp

    def finalize_decoding(self, old_hyp, scorer):
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        T = ((scorer.length() - 1) // self.frame_sampling) * self.frame_sampling + 1
        for key, hyp in old_hyp.items():
            context, label, length = key[0:-2], key[-2], key[-1]
            context = self.grammar.update_context(context, label)
            score = hyp.score + scorer.get(T-length, T-1, label) + self.length_model.score(length, label) + self.grammar.score(context, self.grammar.end_symbol())
            if score >= final_hyp.score:
                final_hyp.score, final_hyp.traceback = score, hyp.traceback
        # return final hypothesis
        return final_hyp

    def traceback(self, hyp, n_frames, scorer):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length, self.score = label, 0, -np.inf
        traceback = hyp.traceback
        labels = []
        segments = [Segment(traceback.label)]
        while not traceback == None:
            segments[-1].length += self.frame_sampling
            labels += [traceback.label] * self.frame_sampling
            if traceback.boundary and not traceback.predecessor == None:
                segments.append( Segment(traceback.predecessor.label) )
            traceback = traceback.predecessor
        # the traceback node for the last frame contributes only length 1, not length self.frame_sampling!
        segments[0].length -= self.frame_sampling - 1
        labels = labels[:-self.frame_sampling + 1]
        # bring into forward order (traceback is a backward pass)
        labels, segments = list(reversed(labels)), list(reversed(segments))
        # score the segments
        offset = 0
        for s in segments:
            s.score = scorer.get(offset, offset + s.length - 1, s.label)
            offset += s.length
        # append labels/length for non processed frames (due to self.frame_sampling there might be a small unprocessed offset at the end)
        segments[-1].length += n_frames - len(labels)
        labels += [hyp.traceback.label] * (n_frames - len(labels))
        return labels, segments


