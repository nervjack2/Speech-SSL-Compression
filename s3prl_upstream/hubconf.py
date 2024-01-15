"""
    Hubconf for Mel HuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
"""

import os
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def compression_20ms_weight_pruning_960hours_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    mean_std_npy_path = '/home/nervjack2/libri-with-cluster/np/libri-960-np-normalize/mean-std.npy'
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, mode='weight-pruning', fp=20, mean_std_npy_path=mean_std_npy_path, *args, **kwargs)

def compression_10ms_weight_pruning_960hours_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    mean_std_npy_path = '/home/nervjack2/libri-with-cluster/np/libri-960-np-normalize/mean-std.npy'
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, mode='weight-pruning', fp=10, mean_std_npy_path=mean_std_npy_path, *args, **kwargs)

def compression_20ms_head_pruning_960hours_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    mean_std_npy_path = '/home/nervjack2/libri-with-cluster/np/libri-960-np-normalize/mean-std.npy'
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, mode='head-pruning', fp=20, mean_std_npy_path=mean_std_npy_path, *args, **kwargs)

def compression_20ms_row_pruning_960hours_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    mean_std_npy_path = '/home/nervjack2/libri-with-cluster/np/libri-960-np-normalize/mean-std.npy'
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, mode='row-pruning', fp=20, mean_std_npy_path=mean_std_npy_path, *args, **kwargs)

def compression_10ms_row_pruning_960hours_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    mean_std_npy_path = '/home/nervjack2/libri-with-cluster/np/libri-960-np-normalize/mean-std.npy'
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, mode='row-pruning', fp=10, mean_std_npy_path=mean_std_npy_path, *args, **kwargs)


def compression_20ms_distillation_960hours_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    mean_std_npy_path = '/home/nervjack2/libri-with-cluster/np/libri-960-np-normalize/mean-std.npy'
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, mode='distillation', fp=20, mean_std_npy_path=mean_std_npy_path, *args, **kwargs)


def compression_20ms_row_pruning_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    mean_std_npy_path = '/home/nervjack2/libri-with-cluster/np/libri-360-np-normalize2/mean-std.npy'
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, mode='row-pruning', fp=20, mean_std_npy_path=mean_std_npy_path, *args, **kwargs)

def compression_10ms_row_pruning_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    mean_std_npy_path = '/home/nervjack2/libri-with-cluster/np/libri-360-np-normalize2/mean-std.npy'
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, mode='row-pruning', fp=10, mean_std_npy_path=mean_std_npy_path, *args, **kwargs)

