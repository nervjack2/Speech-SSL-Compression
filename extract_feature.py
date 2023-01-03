"""
This is a simple example of how to extract feature.
Please use -m option to specify the mode.
"""

import argparse
import torch
import torch.nn as nn
import torchaudio
import numpy as np 
from torch.nn.utils.rnn import pad_sequence
from model import MelHuBERTConfig, MelHuBERTModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['melhubert', 'weight-pruning', 'head-pruning', 'row-pruning', 'distillation']
                                                , help='Different mode of inference')
    parser.add_argument('-c', '--checkpoint', help='Path to model checkpoint')
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    args = parser.parse_args()

    return args

def load_mean_std(mean_std_npy_path):
    mean_std = np.load(mean_std_npy_path)
    mean = torch.Tensor(mean_std[0].reshape(-1))
    std = torch.Tensor(mean_std[1].reshape(-1))
    return mean, std  

def extract_fbank(waveform_path, mean, std):
    waveform, sr = torchaudio.load(waveform_path)
    waveform = waveform*(2**15)
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
    # Downsampling by twice 
    odd_y = y[::2,:]
    even_y = y[1::2,:]
    if odd_y.shape[0] != even_y.shape[0]:
        even_y = torch.cat((even_y, torch.zeros(1,even_y.shape[1]).to(y.device)), dim=0)
    new_y = torch.cat((odd_y, even_y), dim=1)
    return new_y

def prepare_data(wav_path):
    # Load the mean and std of LibriSpeech 360 hours 
    mean_std_npy_path = './example/libri-360-mean-std.npy'
    mean, std = load_mean_std(mean_std_npy_path)
    # Extract fbank feature for model's input
    mel_input = [extract_fbank(p, mean, std) for p in wav_path]
    mel_len = [len(mel) for mel in mel_input]
    mel_input = pad_sequence(mel_input, batch_first=True) # (B x S x D)
    # Prepare padding mask
    pad_mask = torch.ones(mel_input.shape[:-1])  # (B x S)
    # Zero vectors for padding dimension
    for idx in range(mel_input.shape[0]):
        pad_mask[idx, mel_len[idx]:] = 0

    return mel_input, pad_mask

def load_cluster_label(cluster_path):
    cluster = []
    for p in cluster_path:
        c = torch.LongTensor(np.load(p)[::2])
        cluster.append(c)
    cluster = pad_sequence(cluster, batch_first=True, padding_value=-100) 
    return cluster

def main():
    args = get_args()
    print(f'[Extractor] - Extracting feature with {args.mode} mode')
    # Preparing model's input
    wav_path = [
        './example/100-121669-0000.flac',
        './example/1001-134707-0000.flac'
    ]
    print(f'[Extractor] - Extracting feature from these files: {wav_path}')
    mel_input, pad_mask = prepare_data(wav_path)
    # Put data on device 
    mel_input = mel_input.to(
        device=args.device, dtype=torch.float32
    )  
    pad_mask = torch.FloatTensor(pad_mask).to( 
        device=args.device, dtype=torch.float32
    )  

    # Load cluster label 
    cluster_path = [
        './100-121669-0000.npy',
        './1001-134707-0000.npy'
    ]
    cluster = load_cluster_label(cluster_path).to(
        device=args.device, dtype=torch.long
    )
    
    # Load upstream model 
    all_states = torch.load(args.checkpoint, map_location="cpu")
    upstream_config = all_states["Upstream_Config"]["melhubert"]  
    upstream_config = MelHuBERTConfig(upstream_config)
    upstream_model = MelHuBERTModel(upstream_config).to(args.device)
    state_dict = all_states["model"]
    if args.mode == 'melhubert' or args.mode == 'distillation':
        upstream_model.load_state_dict(state_dict)
        upstream_model.eval() 
    elif args.mode == 'weight-pruning':
        from pytorch_code import prune
        from weight_pruning.wp_utils import get_params_to_prune
        params_to_prune, _ = get_params_to_prune(upstream_model)
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.Identity,
        )
        upstream_model.load_state_dict(state_dict)
        for module, name in params_to_prune:
            prune.remove(module, name)
    elif args.mode == 'head-pruning':
        pruned_heads = all_states["Pruned_heads"]
        summarized = {}
        for layer_heads in pruned_heads:
            for layer in layer_heads:
                summarized[layer] = summarized.get(layer, 0) + len(layer_heads[layer])
        pruned_heads = summarized

        for idx, layer in enumerate(upstream_model.encoder.layers):
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
        upstream_model.load_state_dict(state_dict)
    else:
        print(f'Currently not support {args.mode} mode')
    
    total_params = sum(p.numel() for p in upstream_model.parameters())
    print(f'[Extractor] - Successfully load model with {total_params} parameters')

    with torch.no_grad():
        # out = upstream_model(mel_input, pad_mask, get_hidden=True, no_pred=True)
        out = upstream_model(mel_input, pad_mask, cluster, mask=True, no_pred=False)
    loss = torch.nn.functional.cross_entropy(out[1], out[3])
    print(loss)
    exit(0)
    last_layer_feat, hidden_states = out[0], out[5]
    print(f'[Extractor] - Feature with shape of {last_layer_feat.shape} is extracted')
    

if __name__ == '__main__':
    main()