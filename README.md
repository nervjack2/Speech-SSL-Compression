# Speech Compression
This is the official implementation of this two papers:
- [MelHuBERT: A simplified HuBERT on Mel spectrogram](https://arxiv.org/abs/2211.09944)
- [Compressing Transformer-based self-supervised models for speech processing](https://arxiv.org/abs/2211.09949)

## Data Preparing
1. Download fbank features and clustering labels of libri-360 from the [link](https://drive.google.com/drive/u/1/folders/1fplM2ocPK7KcjobWFQOs4HSR_o0gW8NI):

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
python3 train.py -m melhubert -g ./melhubert/config/config_model.yaml -c ./melhubert/config/config_runner.yaml -n EXP_DIR_PATH 
```
-g: Model config \
-c: Runner config \
-n: The model checkpoints, log file, and the pre-training config you used will be saved at this directory \

### Head-Pruning on MelHuBERT
Execute the following command to do head-pruning on a pre-trained MelHuBERT. 
There are two metric for head-pruning, l1 and data-driven. 

For l1 metric, please execute
```
python3 train.py  -m head-pruning -i Path/to/CkptFile -g ./head_pruning/config/l1/config_model.yaml -c ./head_pruning/config/l1/config_runner.yaml -n EXP_DIR_PATH
```
For data-driven metric, please execute
```
python3 train.py -m head-pruning -i Path/to/CkptFile -g ./head_pruning/config/data_driven/config_model.yaml -c ./head_pruning/config/data_driven/config_runner.yaml -n EXP_DIR_PATH 
```

-i: Pre-trained MelHuBERT will be loaded from this .ckpt file \
-g: model config \
-c: runner config \
-n: The model checkpoints, log file, and pre-training config you used will be saved at this directory

### Distillation on MelHuBERT
Execute the following command to do knowledge distillation on a pre-trained MelHuBERT teacher. 

Please execute
```
python3 train.py  -m distillation -i Path/to/CkptFile -g ./distillation/config/config_model.yaml -c ./distillation/config/config_runner.yaml -n EXP_DIR_PATH
```

-i: Pre-trained MelHuBERT will be loaded from this .ckpt file \
-g: model config \
-c: runner config \
-n: The model checkpoints, log file, and pre-training config you used will be saved at this directory

Choosing between "masked" and "nomasked" for **loss_param.type** in config_model.yaml. \
This parameter controls whether the input would be randomly masked.

## Pretrained Models 
Pretrained models are saved at [here](https://drive.google.com/drive/u/1/folders/1DHmpyQ3aekB6YFtNq2_du2HydR2rZHpM) \
You can use wget to download the models \
Their pretraining config are saved at pretraining-config/
### Load models
```
import torch
from model import MelHuBERTConfig, MelHuBERTModel
    
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
| melhubert-10ms-stage1-libri360.ckpt     | ~90M   |            13.61           |          15.10          |             64.75      |
| melhubert-20ms-stage1-libri360.ckpt     | ~90M   |            13.61           |          12.96          |             66.34      |

## Acknowledgement 
Our implementation of pre-training interface is based on [S3PRL toolkit](https://github.com/s3prl/s3prl)
