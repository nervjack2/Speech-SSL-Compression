"""
MelHuBERT distillation interface 
"""

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MelHuBERTModel, MelHuBERTConfig

class MelHuBERTDistiller(nn.Module):
    def __init__(self, upstream_config, initial_weight=None, device='cuda', multi_gpu=False):
        super(MelHuBERTDistiller, self).__init__()

        self.initial_weight = initial_weight
        self.device = device
        self.multi_gpu = multi_gpu

        self.upstream_config = upstream_config
        # Initialize the model 
        self._init_model()
        # Define distillation loss 
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100, size_average=True)        
        self.loss_temp = self.upstream_config['loss_param']['T']
        self.loss_alpha = self.upstream_config['loss_param']['alpha']
        self.loss_type = self.upstream_config['loss_param']['type']
        
        if self.loss_type == 'masked':
            self.mask_or_not = True 
        elif self.loss_type == 'nomasked':
            self.mask_or_not = False 
        else:
            print(f'[Distiller] - No such loss type {self.loss_type}')
            exit(0)

        # Make multiple GPU training possible
        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[Distiller] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[Distiller] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def _init_model(self):    
        print('[Distiller] - Initializing model...')
        # Define student model architecture
        self.student_config = MelHuBERTConfig(self.upstream_config['melhubert'])
        self.model = MelHuBERTModel(self.student_config)
        # Define teacher model architecture
        self.teacher_config = MelHuBERTConfig(self.upstream_config['teacher'])
        self.teacher_model = MelHuBERTModel(self.teacher_config)
        # Load teacher model's weight
        assert self.initial_weight, 'Please specify teacher\'s weight by -i argument'
        all_states = torch.load(self.initial_weight, map_location="cpu")
        try:             
            self.teacher_model.load_state_dict(all_states["model"])
            print(f'[Distiller] - Load teacher model\'s weight from {self.initial_weight}')
        except:
            raise NotImplementedError('Could not load the teacher model\'s weight')

        # Initializing from teacher
        if self.upstream_config['melhubert']['initial_from_teacher']:
            print("[Distiller] - Initializing from teacher")
            self.model.encoder.pos_conv.load_state_dict(
                self.teacher_model.encoder.pos_conv.state_dict()
            )
            for l in range(self.student_config.encoder_layers):
                self.model.encoder.layers[l].load_state_dict(
                    self.teacher_model.encoder.layers[l].state_dict()
                )

    def load_model(self, init_ckpt):
        assert 'model' in init_ckpt
        if self.multi_gpu:
            self.model.module.load_state_dict(init_ckpt['model'])
        else:
            self.model.load_state_dict(init_ckpt['model'])

    def add_state_to_save(self, all_states):
        all_states['model'] = self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict()
        all_states['Upstream_Config'] = self.upstream_config
        return all_states
    
    def loss_fn_kd(self, outputs, labels, teacher_outputs, T=1, alpha=0.5):
        # Teacher's cross entropy loss
        teacher_loss = self.loss(teacher_outputs, labels)
        # Student's cross entropy loss
        hard_loss = self.loss(outputs, labels)
        # Student-Teacher KL divergence loss
        soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                                F.softmax(teacher_outputs/T, dim=1))
        total_loss = (hard_loss * (1. - alpha)) + (soft_loss * alpha)
        return total_loss, hard_loss, soft_loss, teacher_loss
    
    def acc(self, outputs, labels):
        acc = torch.sum(torch.argmax(outputs, dim=1) == labels).item()
        total = len(labels)
        return acc, total

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
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
        audio_feat, label, pad_mask, audio_len = data[0], data[1], data[2], data[3]
        audio_feat = audio_feat.to(self.device)
        label = label.to(self.device)
        pad_mask = pad_mask.to(self.device)
        
        with torch.no_grad():
            _, t_logit_m, t_logit_u, t_label_m, t_label_u, _, _, mask_indices = self.teacher_model(audio_feat, pad_mask, label, mask=self.mask_or_not)

        _, logit_m, logit_u, label_m, label_u, _, _, _ = self.model(audio_feat, pad_mask, label, mask=self.mask_or_not, teacher_mask_indices=mask_indices)

        loss = 0.0 
        hard_loss, soft_loss = 0.0, 0.0
        teacher_loss = 0.0

        if self.loss_type == 'masked':
            all_loss, h_loss, s_loss, t_loss = self.loss_fn_kd(logit_m, label_m, t_logit_m, T=self.loss_temp, alpha=self.loss_alpha)
            loss += all_loss
            hard_loss += h_loss
            soft_loss += s_loss
            teacher_loss += t_loss

        elif self.loss_type == 'nomasked':
            all_loss, h_loss, s_loss, t_loss = self.loss_fn_kd(logit_u, label_u, t_logit_u, T=self.loss_temp, alpha=self.loss_alpha)
            loss += all_loss
            hard_loss += h_loss
            soft_loss += s_loss
            teacher_loss += t_loss
        
        return loss
    