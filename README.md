# Speech Self-Supervised Model Compression
This is the official implementation of this two papers:
- [MelHuBERT: A simplified HuBERT on Mel spectrogram](https://arxiv.org/abs/2211.09944)
- [Compressing Transformer-based self-supervised models for speech processing](https://arxiv.org/abs/2211.09949)

We support four diffrent type of compression on a transformer-based speech SSL model, including weight pruning, head pruning, low-rank approximation, and knowledge distillation.

## Data Preparing
First, please execute the following command to prepare LibriSpeech 360 horus and paired cluster labels (K-means on log Mel feature)
```
bash preprocess.sh [DATA_DIR]
```

Then, please adjust **datarc.sets** in config_runner.yaml to [ DATA_DIR/libri-360-data-cluster-pair.csv ]

The mean and std of LibriSpeech 360 hours is saved at DATA_DIR/mean-std.npy

## Training Command 
### Pre-training MelHuBERT from scratch
Execute the following command to pretrain MelHuBERT from scratch with default configuration
```
python3 train.py -m melhubert -g ./melhubert/config/config_model.yaml -c ./melhubert/config/config_runner.yaml -n EXP_DIR_PATH 
```
-g: Model config \
-c: Runner config \
-n: The model checkpoints, log file, and the pre-training config you used will be saved at this directory 

### Weight Pruning
Execute the following command to do weight pruning on a pre-trained MelHuBERT. 

```
python3 train.py  -m weight-pruning -i Path/to/CkptFile -g ./weight_pruning/config/config_model.yaml -c ./weight_pruning/config/config_runner.yaml -n EXP_DIR_PATH
```

-i: Pre-trained MelHuBERT will be loaded from this .ckpt file \
-g: model config \
-c: runner config \
-n: The model checkpoints, log file, and pre-training config you used will be saved at this directory

### Head Pruning
Execute the following command to do head pruning on a pre-trained MelHuBERT. 
There are two metric for head pruning, l1 and data-driven. 

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

### Distillation 
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
Still need to fix the link

Please execute the following command to download the pretrained MelHuBERT 
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G9ri5QPc2e8dvafe8bSUNjbP9hTrJtb2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1G9ri5QPc2e8dvafe8bSUNjbP9hTrJtb2" -O melhubert-libri360-20fp.ckpt && rm -rf /tmp/cookies.txt
```
## Extracting feature 
Please execute the following command to extract feature from two example waveforms
```
python3 extract_feature.py -m [MODE] -c [CHECKPOINT]
```

-m: Choose from melhubert, weight-pruning, head-pruning, row-pruning, and distillation \
-c: Model checkpoint path

## Acknowledgement 
Our implementation of pre-training interface is based on [S3PRL toolkit](https://github.com/s3prl/s3prl)
