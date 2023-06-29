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
    
    def __init__(self, frame_period, task_config, bucket_size, sets, max_timestep=0, multitask=False):
        super(MelFeatDataset, self).__init__(task_config, bucket_size, sets, max_timestep)
        self.frame_period = frame_period

    def _load_feat(self, feat_path):
        feat = np.load(feat_path)
        if self.frame_period == 20:
            odd_feat = feat[::2,:]
            even_feat = feat[1::2,:]
            if odd_feat.shape[0] != even_feat.shape[0]:
                even_feat = np.concatenate((even_feat, np.zeros((1,even_feat.shape[1]))), axis=0)
            feat = np.concatenate((odd_feat, even_feat), axis=1)
        return torch.FloatTensor(feat)

    def _load_label(self, label_path, feat_len):
        label = np.load(label_path)
        label_len = label.shape[0]
        if self.frame_period == 20 and feat_len != label_len:
            label = label[::2]
        return torch.LongTensor(label)

    def __getitem__(self, index):
        # Load acoustic feature, label and pad
        x_batch, y_batch = [], []
        for x_file, y_file in zip(self.X[index], self.Y[index]):
            feat = self._load_feat(x_file)
            label = self._load_label(y_file, feat.shape[0])
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

def get_feat_iterator(feat_dir, split, nshard, rank):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    def iterate():
        feat = np.load(feat_path, mmap_mode="r")
        assert feat.shape[0] == (offsets[-1] + lengs[-1])
        for offset, leng in zip(offsets, lengs):
            yield feat[offset: offset + leng]

    return iterate, len(lengs)

class FairseqFeatLabelDataset(Dataset):
    def __init__(self, task_config, bucket_size,feat_dir, label_dir, split):
        super(FairseqFeatLabelDataset, self).__init__()

        self.task_config = task_config
        self.sample_length = task_config['sequence_length'] 
        if self.sample_length > 0:
            print('[Dataset] - Sampling random segments for training, sample length:', self.sample_length)

        feat_path = f"{feat_dir}/{split}.npy"
        leng_path = f"{feat_dir}/{split}.len"
        with open(leng_path, "r") as f:
            self.lengs = [int(line.rstrip()) for line in f]
            self.offsets = [0] + np.cumsum(self.lengs[:-1]).tolist()

        self.feat = np.load(feat_path, mmap_mode="r")

        assert self.feat.shape[0] == (self.offsets[-1] + self.lengs[-1])
       
        print(f'[Dataset] - Load {len(self.lengs)} training data from directory {feat_dir}')

        label_path = f"{label_dir}/{split}.km"
        self.labels = []
        with open(label_path) as fp:
            for x in fp:
                l = list(map(int, x.strip().split(' ')))
                self.labels.append(l)

        # Sort data by length 
        idx = np.argsort(np.array(self.lengs))[::-1]
        self.lengs = np.array(self.lengs)[idx].tolist()
        self.offsets = np.array(self.offsets)[idx].tolist()
        self.labels = np.array(self.labels)[idx].tolist()

        # Use bucketing to make utterances with closest length in a batch
        self.LEN = []
        self.OFFSET = []
        self.Y = []

        batch_len, batch_offset, batch_y = [], [], []

        for l, o, y in zip(self.lengs, self.offsets, self.labels):
            batch_len.append(l)
            batch_offset.append(o)
            batch_y.append(y)
            
            # Fill in batch_x until batch is full
            if len(batch_len) == bucket_size:
                self.LEN.append(batch_len)
                self.OFFSET.append(batch_offset)
                self.Y.append(batch_y)
                batch_len, batch_offset, batch_y = [], [], []

        # Gather the last batch
        if len(batch_len) > 1: 
            self.LEN.append(batch_len)
            self.OFFSET.append(batch_offset)
            self.Y.append(batch_y)

    def _sample(self, x, y):
        if self.sample_length <= 0: return x, y
        if len(x) < self.sample_length: return x, y
        idx = random.randint(0, len(x)-self.sample_length)
        return x[idx:idx+self.sample_length], y[idx:idx+self.sample_length]
    
    def _sample_multitask(self, x, y1, y2):
        if self.sample_length <= 0: return x, y1, y2
        if len(x) < self.sample_length: return x, y1, y2
        idx = random.randint(0, len(x)-self.sample_length)
        return x[idx:idx+self.sample_length], y1[idx:idx+self.sample_length], y2[idx:idx+self.sample_length]

    def __len__(self):
        return len(self.LEN)

    def collate_fn(self, items):
        # Hack bucketing
        items = items[0] 
        return items

class LoadFairseqDataset(FairseqFeatLabelDataset):
    def __init__(self, frame_period, task_config, bucket_size, feat_dir, label_dir, split, mean_std_pth, multitask=False):
        super(LoadFairseqDataset, self).__init__(task_config, bucket_size, feat_dir, label_dir, split)
        mean_std = np.load(mean_std_pth)
        self.mean = mean_std[0].reshape(-1)
        self.std = mean_std[1].reshape(-1)
        self.frame_period = frame_period
        self.multitask = multitask

    def _load_feat(self, leng, offset):
        feat = self.feat[offset: offset + leng]
        feat = (feat-self.mean)/self.std
        if self.frame_period == 20:
            odd_feat = feat[::2,:]
            even_feat = feat[1::2,:]
            if odd_feat.shape[0] != even_feat.shape[0]:
                even_feat = np.concatenate((even_feat, np.zeros((1,even_feat.shape[1]))), axis=0)
            feat = np.concatenate((odd_feat, even_feat), axis=1)
        return torch.FloatTensor(feat)

    def _load_label(self, y, feat_len):
        label = np.array(y)
        label_len = label.shape[0]
        if self.frame_period == 20 and feat_len != label_len:
            if not self.multitask:
                label = label[::2]
                return torch.LongTensor(label)
            else:
                label_1 = label[::2]
                label_2 = label[1::2]
                if len(label_2) != len(label_1):
                    label_2 = np.append(label_2, label_1[-1])
                return torch.LongTensor(label_1), torch.LongTensor(label_2)

    def __getitem__(self, index):
        # Load acoustic feature, label and pad
        if not self.multitask:
            x_batch, y_batch = [], []
        else:
            x_batch, y1_batch, y2_batch = [], [], []

        for leng, offset, y in zip(self.LEN[index], self.OFFSET[index], self.Y[index]):
            feat = self._load_feat(leng, offset)
            label = self._load_label(y, feat.shape[0])
            if self.multitask:
                label_1 = label[0]
                label_2 = label[1]
                x, y1, y2 = self._sample_multitask(feat, label_1, label_2)
                x_batch.append(x)
                y1_batch.append(y1)
                y2_batch.append(y2)
            else:
                x, y = self._sample(feat, label)
                x_batch.append(x)
                y_batch.append(y)

        x_len = [len(x_b) for x_b in x_batch]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        pad_mask = torch.ones(x_pad_batch.shape[:-1]) 
        # Zero vectors for padding dimension
        for idx in range(x_pad_batch.shape[0]):
            pad_mask[idx, x_len[idx]:] = 0

        if not self.multitask:
            y_pad_batch = pad_sequence(y_batch, batch_first=True, padding_value=-100) 
            return x_pad_batch, y_pad_batch, pad_mask, x_len
        else: 
            y1_pad_batch = pad_sequence(y1_batch, batch_first=True, padding_value=-100) 
            y2_pad_batch = pad_sequence(y2_batch, batch_first=True, padding_value=-100) 
            return x_pad_batch, y1_pad_batch, y2_pad_batch, pad_mask, x_len

