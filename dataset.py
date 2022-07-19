"""
    Training dataset of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/blob/master/s3prl/pretrain/bucket_dataset.py)
    Reference author: Andy T. Liu (https://github.com/andi611)
"""
import numpy as np
import torch
import os
import random
import pandas as pd
import re
import kaldiark
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset


class FeatLabelDataset(Dataset):
    def __init__(self, task_config, bucket_size, sets, max_timestep=0):
        super(FeatLabelDataset, self).__init__()

        self.task_config = task_config
        self.sample_length = task_config['sequence_length'] 
        if self.sample_length > 0:
            print('[Dataset] - Sampling random segments for training, sample length:', self.sample_length)

        # You can assign multiple .csv data files in sets
        tables = [pd.read_csv(s) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)
        print('[Dataset] - Training data from these sets:', str(sets))

        # Drop seqs that are too long
        if max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]
        # Drop seqs that are too short
        if max_timestep < 0:
            self.table = self.table[self.table.length > (-1 * max_timestep)]

        X = self.table['file_path'].tolist()
        Y = self.table['label_path'].tolist()
        X_lens = self.table['length'].tolist()
        self.num_samples = len(X)
        print('[Dataset] - Number of individual training instances:', self.num_samples)

        # Use bucketing to make utterances with closest length in a batch
        self.X = []
        self.Y = []
        batch_x, batch_y, batch_len = [], [], []

        for x, y, x_len in zip(X, Y, X_lens):
            batch_x.append(x)
            batch_y.append(y)
            batch_len.append(x_len)
            
            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                self.X.append(batch_x)
                self.Y.append(batch_y)
                batch_x, batch_y, batch_len = [], [], []
        
        # Gather the last batch
        if len(batch_x) > 1: 
            self.X.append(batch_x)
            self.Y.append(batch_y)

    def _sample(self, x, y):
        if self.sample_length <= 0: return x, y
        if len(x) < self.sample_length: return x, y
        idx = random.randint(0, len(x)-self.sample_length)
        return x[idx:idx+self.sample_length], y[idx:idx+self.sample_length]

    def __len__(self):
        return len(self.X)

    def collate_fn(self, items):
        # Hack bucketing
        items = items[0] 
        return items


class MelFeatDataset(FeatLabelDataset):
    
    def __init__(self, task_config, bucket_size, sets, max_timestep=0):
        super(MelFeatDataset, self).__init__(task_config, bucket_size, sets, max_timestep)

    def _load_feat(self, feat_path):
        return torch.FloatTensor(np.load(feat_path))

    def _load_label(self, label_path):
        return torch.LongTensor(np.load(label_path))

    def __getitem__(self, index):
        # Load acoustic feature, label and pad
        x_batch, y_batch = [], []
        for x_file, y_file in zip(self.X[index], self.Y[index]):
            feat = self._load_feat(x_file)
            label = self._load_label(y_file)
            x, y = self._sample(feat, label)
            x_batch.append(x)
            y_batch.append(y)

        x_len = [len(x_b) for x_b in x_batch]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        # Pad -100 for ignore index
        y_pad_batch = pad_sequence(y_batch, batch_first=True, padding_value=-100) 

        pad_mask = torch.ones(x_pad_batch.shape[:-1])  # (batch_size, seq_len)
        # Zero vectors for padding dimension
        for idx in range(x_pad_batch.shape[0]):
            pad_mask[idx, x_len[idx]:] = 0

        return x_pad_batch, y_pad_batch, pad_mask, x_len


def read_scp(scp_path):
    scp = []
    with open(scp_path) as f:
        for line in f:
            m = re.match(r'(.+) (.+):(.+)', line)
            key, path, shift = m.group(1), m.group(2), m.group(3)
            # key, path_shift = line.split()
            # path, shift = path_shift.split(':')
            scp.append((key, path, int(shift)))
    return scp


def read_feat(path, shift, mean = None, stddev = None):
    f = open(path, 'rb')
    f.seek(shift)
    mat = kaldiark.parse_feat_matrix(f)
    if mean is not None and stddev is not None:
        mat = (mat - mean) / stddev
    f.close()

    return mat


def read_assignment(path, shift):
    f = open(path)
    f.seek(shift)
    line = f.readline()
    assignment = [int(e) for e in line.split()]
    f.close()

    return assignment


def read_mean_var(path):
    f = open(path)
    s = np.array(eval(f.readline()))
    s2 = np.array(eval(f.readline()))
    nsamples = int(f.readline())
    mean = s / nsamples
    stddev = np.sqrt(s2 / nsamples - mean * mean)
    f.close()

    return mean, stddev


class LibriSpeechFbank:
    def __init__(self, fbank_scp_path, mean_var_path = None):
        self.fbank_scp = read_scp(fbank_scp_path)
        self.mean, self.stddev = read_mean_var(mean_var_path) if mean_var_path else (None, None)

    def __len__(self):
        return len(self.fbank_scp)

    def __iter__(self):
        for fbank_key, fbank_path, fbank_shift in self.fbank_scp:
            feat = read_feat(fbank_path, fbank_shift, self.mean, self.stddev)

            yield [torch.FloatTensor(feat)]


class LibriSpeechKmeans:
    def __init__(self, fbank_scp_path, bas_scp_path, mean_var_path = None):
        self.fbank_scp = read_scp(fbank_scp_path)
        self.bas_scp = read_scp(bas_scp_path)
        self.mean, self.stddev = read_mean_var(mean_var_path) if mean_var_path else (None, None)

    def __len__(self):
        return len(self.fbank_scp)

    def __iter__(self):
        indices = list(range(len(self.fbank_scp)))
        random.shuffle(indices)

        for i in indices:
            fbank_key, fbank_path, fbank_shift = self.fbank_scp[i]
            feat = read_feat(fbank_path, fbank_shift, self.mean, self.stddev)

            asg_key, asg_path, asg_shift = self.bas_scp[i]
            asg = read_assignment(asg_path, asg_shift)

            yield [torch.FloatTensor(feat)], [torch.LongTensor(asg)]

