"""
    Training interface of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
"""
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from melhubert import MelHuBERTModel, MelHuBERTConfig
import dataset

class MelHuBERTPretrainer(nn.Module):
    def __init__(self, datarc, upstream_config, initial_weight=None, device='cuda', multi_gpu=False):
        super(MelHuBERTPretrainer, self).__init__()

        self.datarc = datarc
        self.initial_weight = initial_weight
        self.device = device
        self.multi_gpu = multi_gpu

        # When doing resume training, the type upstream_config will be dictionary here
        if type(upstream_config) == str:
            self.upstream_config = yaml.load(open(upstream_config, 'r'), Loader=yaml.FullLoader)
            print('[MelHuBERTPretrainer] - Using upstream config from:', upstream_config)
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print('[MelHuBERTPretrainer] - Using upstream config from the previous experiment.')
        else:
            raise ValueError
        
        # Initialize the model 
        self._init_model()
        # Define pre-training loss
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        # Get dataloader
        self.dataloader = self._get_train_dataloader()
        # Make multiple GPU training possible
        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[MelHuBERTPretrainer] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[MelHuBERTPretrainer] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def _init_model(self):
        print('[MelHuBERTPretrainer] - Initializing model...')
        self.model_config = MelHuBERTConfig(self.upstream_config['hubert'])
        self.model = MelHuBERTModel(self.model_config)

        # Do initialization from a checkpoint if needed
        if self.initial_weight:
            all_states = torch.load(self.initial_weight, map_location="cpu")
            try:             
                self.model.load_state_dict(all_states["Model"])
                print(f'[MelHuBERTPretrainer] Load initilization model weight from {self.initial_weight}')
            except:
                raise NotImplementedError('Could not load the initilization weight')
        
        # Adjust the dimension of the last predicting feed forward layer and do re-initialize 
        if 'adjust_init_weight' in self.upstream_config.keys():
            self.model.adjust_arch(self.upstream_config['adjust_init_weight'])
            self.upstream_config['hubert']['num_cluster'] = self.upstream_config['adjust_init_weight']['num_cluster']
            print(f'[MelHuBERTPretrainer] Adjust last predicting feed forward layer\'s dimension according to the config.')

    def _get_train_dataloader(self):
        dataloader = dataset.LibriSpeechKmeans(self.datarc['fbank_scp'],
            self.datarc['bas_scp'], self.datarc['fbank_mean_var'])

        return dataloader

    def load_model(self, init_ckpt):
        assert 'Model' in init_ckpt
        if self.multi_gpu:
            self.model.module.load_state_dict(init_ckpt['Model'])
        else:
            self.model.load_state_dict(init_ckpt['Model'])

    def add_state_to_save(self, all_states):
        all_states['Model'] = self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict()
        all_states['Upstream_Config'] = self.upstream_config
        return all_states

    def forward(self, data, global_step=0, log_step=1000):
        """
        Args:
            data:
                [audio feature, cluster id, padding mask, audio length]
            
            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step
        Return:
            loss        
        """
        audio_feat, label, pad_mask = data[0], data[1], data[2]
        audio_feat = audio_feat.to(self.device)
        label = label.to(self.device)
        pad_mask = pad_mask.to(self.device)
  
        _, logit_m, logit_u, label_m, label_u, _, _ = self.model(audio_feat, pad_mask, label, mask=True)

        loss = 0.0 
        if logit_m != None and label_m != None and self.model_config.pred_masked_weight > 0: 
            loss += self.model_config.pred_masked_weight * self.loss(logit_m, label_m)
        if logit_u != None and label_u != None and self.model_config.pred_nomask_weight > 0: 
            loss += self.model_config.pred_nomask_weight * self.loss(logit_u, label_u)
        
        return loss

