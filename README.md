This is the official implementation of this two papers:
- [MelHuBERT: A simplified HuBERT on Mel spectrogram](https://arxiv.org/abs/2211.09944)
- [Compressing Transformer-based self-supervised models for speech processing](https://arxiv.org/abs/2211.09949)

Our implementation of pre-training interface is based on [S3PRL toolkit](https://github.com/s3prl/s3prl)

## Data Preparing
1. Download fbank features and clustering labels of libri-360 by the following command:
    ```
    wget http://140.112.30.56:9999/dataset/libri-360.tar.gz
    ```

2. Run the following script to prepare numpy data for training:
    ```
    python3 preprocess/tidy_libri360_kaldi_data.py [KALDI_DATA_DIR] [OUT_NUMPY_DIR] [DATA_CSV_FILE]
    ```
    Where KALDI_DATA_DIR is the directory you will get after decompressing in step 1. 
    OUT_NUMPY_DIR is the output directory of numpy data.
    DATA_CSV_FILE is the file recording data path and its corresponding label path which will be used later. 
    
    Note-1: Please use absolute path when running the command. \
    Note-2: The mean and standard variance of LibriSpeech 360 hours will be saved at OUT_NUMPY_DIR/mean-std.npy. This will be useful if you are going to test the downstream performance \
    Note-3: This script will do normalization for you.

3. Adjust your config file:
    - config_runner.yaml: Adjust **pretrain_expert.datarc.sets** to  [ DATA_CSV_FILE ]. 

## Command 
### Pre-training MelHuBERT from scratch
Execute the following command to pretrain MelHuBERT from scratch with default configuration
```
python3 train.py -g ./melhubert/config/config_model.yaml -c ./melhubert/config/config_runner.yaml -n EXP_DIR_PATH -m melhubert
```
-g: Model config \
-c: Runner config \
-n: The model checkpoints, log file, and the pre-training config you used will be saved at this directory \

### Head-Pruning on MelHuBERT
Execute the following command to do head-pruning on a pre-trained MelHuBERT. 
There are two metric for head-pruning, l1 and data-driven. 

For l1 metric, please execute the following command
```
python3 train.py -i Path/to/CkptFile -g ./head_pruning/config/l1/config_model.yaml -c ./head_pruning/config/l1/config_runner.yaml -n EXP_DIR_PATH -m head-pruning
```
For data-driven metric, please execute the following command 
```
python3 train.py -i Path/to/CkptFile -g ./head_pruning/config/data_driven/config_model.yaml -c ./head_pruning/config/data_driven/config_runner.yaml -n EXP_DIR_PATH -m head-pruning
```

-i: Pre-trained MelHuBERT will be loaded from this .ckpt file \
-g: model config \
-c: runner config \
-n: The model checkpoints, log file, and pre-training config you used will be saved at this directory

## Pretrained Models 
Pretrained models are saved at [here](http://140.112.30.56:9999/pretrained_model/) \
You can use wget to download the models \
Their pretraining config are saved at pretraining-config/
### Load models
```
import torch
from melhubert import MelHuBERTConfig, MelHuBERTModel
    
all_states = torch.load(model_ckpt, map_location="cpu")
upstream_config = all_states["Upstream_Config"]["hubert"]  
upstream_config = MelHuBERTConfig(upstream_config)
upstream_model = MelHuBERTModel(upstream_config).to(device)
state_dict = all_states["model"]
upstream_model.load_state_dict(state_dict)
upstream_model.eval() # If you are only used to extract representation
last_layer_feat, _, _, _, _, hidden_states, _ = upstream_model(mel_input, input_pad_mask, get_hidden=True, no_pred=True)
```
### Pretrained models performance
|                  Model name                  | Params | Phone Classification(PER%) | Phone Recognition(PER%) | Speaker Identificaiton(ACC%) |
|:--------------------------------------------:|:------:|:--------------------------:|:-----------------------:|:----------------------------:|
| melhubert-10ms-stage1-libri360-200epochs.ckpt     | ~90M   |            13.61           |          15.10          |             64.75      |
| melhubert-20ms-stage1-libri360-200epochs.ckpt     | ~90M   |            13.61           |          12.96          |             66.34      |

