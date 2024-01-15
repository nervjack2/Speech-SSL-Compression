"""
    UpstreamExpert of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/tree/master/s3prl/upstream/distiller)
    Reference author: Heng-Jui Chang (https://github.com/vectominist)
"""

import yaml
import torch
import torchaudio
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from ..interfaces import UpstreamBase
from .model import MelHuBERTModel, MelHuBERTConfig

def load_mean_std(mean_std_npy_path):
    mean_std = np.load(mean_std_npy_path)
    mean = torch.Tensor(mean_std[0].reshape(-1))
    std = torch.Tensor(mean_std[1].reshape(-1))
    return mean, std  

def extract_fbank(wavform, mean, std, fp=20):
    waveform = wavform.unsqueeze(0)*(2**15)
    y = torchaudio.compliance.kaldi.fbank(
                        waveform,
                        num_mel_bins=40,
                        sample_frequency=16000,
                        window_type='hamming',
                        frame_length=25,
                        frame_shift=10)
    # Normalize by the mean and std of Librispeech
    mean = mean.to(y.device, dtype=torch.float32)
    std = std.to(y.device, dtype=torch.float32)
    y = (y-mean)/std
    if fp == 20:
        # Downsampling by twice 
        odd_y = y[::2,:]
        even_y = y[1::2,:]
        if odd_y.shape[0] != even_y.shape[0]:
            even_y = torch.cat((even_y, torch.zeros(1,even_y.shape[1]).to(y.device)), dim=0)
        y = torch.cat((odd_y, even_y), dim=1)
    return y

class UpstreamExpert(UpstreamBase):
    """
    The Mel Hubert wrapper
    """
    def __init__(self, ckpt, mode, fp, mean_std_npy_path, model_config=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode 
        self.fp = fp 
        # Load upstream model 
        all_states = torch.load(ckpt, map_location="cpu")
        if "melhubert" in all_states["Upstream_Config"]:
            upstream_config = all_states["Upstream_Config"]["melhubert"] 
        else:
            upstream_config = all_states["Upstream_Config"]["hubert"] 
        upstream_config = MelHuBERTConfig(upstream_config)
        self.upstream_model = MelHuBERTModel(upstream_config)
        state_dict = all_states["model"]
        # if self.mode == 'melhubert' or self.mode == 'distillation' or self.mode == 'row-pruning':
        #     self.upstream_model.load_state_dict(state_dict)
    
        if 'Pruned_heads' in all_states:
            # If head-pruned
            pruned_heads = all_states["Pruned_heads"]
            summarized = {}
            for layer_heads in pruned_heads:
                for layer in layer_heads:
                    summarized[layer] = summarized.get(layer, 0) + len(layer_heads[layer])
            pruned_heads = summarized

            for idx, layer in enumerate(self.upstream_model.encoder.layers):
                if idx in pruned_heads:
                    layer.self_attn.num_heads -= pruned_heads[idx]
                    orig_embed_dim = layer.self_attn.embed_dim
                    embed_dim = layer.self_attn.head_dim * layer.self_attn.num_heads
                    bias = True
                    layer.self_attn.embed_dim = embed_dim
                    layer.self_attn.k_proj = nn.Linear(orig_embed_dim, embed_dim, bias=bias)
                    layer.self_attn.v_proj = nn.Linear(orig_embed_dim, embed_dim, bias=bias)
                    layer.self_attn.q_proj = nn.Linear(orig_embed_dim, embed_dim, bias=bias)
                    layer.self_attn.out_proj = nn.Linear(embed_dim, orig_embed_dim, bias=bias)
                    layer.self_attn.skip_embed_dim_check = True
                    layer.self_attn.reset_parameters()   

        if 'Pruning' in all_states:
            # If weight-pruned
            from .pytorch_code import prune
            from .wp_utils import get_params_to_prune
            params_to_prune, _ = get_params_to_prune(self.upstream_model)
            prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.Identity,
            )
        
        self.upstream_model.load_state_dict(state_dict)
        
        if 'Pruning' in all_states:
            for module, name in params_to_prune:
                prune.remove(module, name)
        
        # Load the mean and std of LibriSpeech 360 hours 
        self.mean, self.std = load_mean_std(mean_std_npy_path)
        
    def get_downsample_rates(self, key: str) -> int:
        if self.fp == 20:
            return 320
        elif self.fp == 10:
            return 160 

    def forward(self, wavs, no_pred=True, norm=True):
        # Extract fbank feature for model's input
        mel_input = [extract_fbank(p, self.mean, self.std, fp=self.fp) for p in wavs]
        mel_len = [len(mel) for mel in mel_input]
        mel_input = pad_sequence(mel_input, batch_first=True) # (B x S x D)
        # Prepare padding mask
        pad_mask = torch.ones(mel_input.shape[:-1])  # (B x S)
        # Zero vectors for padding dimension
        for idx in range(mel_input.shape[0]):
            pad_mask[idx, mel_len[idx]:] = 0
        mel_input = mel_input.to(
            device=wavs[0].device, dtype=torch.float32
        )  
        pad_mask = torch.FloatTensor(pad_mask).to( 
            device=wavs[0].device, dtype=torch.float32
        )  

        hidden, _, _, _, _, layer_hiddens, pre_feat, _ = self.upstream_model(mel_input, pad_mask, mask=False, no_pred=True, get_hidden=True)

        hidden_states = [pre_feat] + layer_hiddens

        states = {
            "hidden_states": hidden_states,
            "last_hidden_state": hidden,
        }

        return states
