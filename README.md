# Speech Self-Supervised Model Compression
This is the official implementation of:
- [Compressing Transformer-based self-supervised models for speech processing](https://arxiv.org/abs/2211.09949)

We support four diffrent type of compression on a transformer-based speech SSL model (MelHuBERT), including weight pruning, head pruning, low-rank approximation, and knowledge distillation.

## Data Preparing
1. Download [dataset](https://drive.google.com/file/d/1hMvXA7VxtcIa6gd0-nPWYCho5qLmeKFL/view?usp=sharing).
2. Please execute the following command to prepare LibriSpeech 960 horus and paired cluster labels
```
bash preprocess.sh [DATA_ZIP_FILE] [DATA_OUT_DIR]
```
Note: Please use absolute path here.

Then, please adjust **datarc.sets** in config_runner.yaml to [ DATA_OUT_DIR/libri960-stg2-{FRAME_PERIOD}.csv ]

The mean and std of LibriSpeech 960 hours is saved at DATA_OUT_DIR/mean-std.npy

## Training Command 
<!-- ### Pre-training MelHuBERT from scratch
Execute the following command to pretrain MelHuBERT from scratch with default configuration
```
python3 train.py -m melhubert -g ./melhubert/config/config_model.yaml -c ./melhubert/config/config_runner.yaml -n EXP_DIR_PATH 
```
-g: Model config \
-c: Runner config \
-n: The model checkpoints, log file, and the pre-training config you used will be saved at this directory  -->

### Weight Pruning
Execute the following command to do weight pruning on a pre-trained MelHuBERT. 

```
python3 train.py  -m weight-pruning -i Path/to/CkptFile -g ./weight_pruning/config/config_model_{FRAME_PERIOD}.yaml -c ./weight_pruning/config/config_runner_{FRAME_PERIOD}.yaml -n EXP_DIR_PATH -f FRAME_PERIOD
```

-i: Pre-trained MelHuBERT will be loaded from this .ckpt file \
-g: model config \
-c: runner config \
-n: The model checkpoints, log file, and pre-training config you used will be saved at this directory \
-f: Frame period

### Head Pruning
Execute the following command to do head pruning on a pre-trained MelHuBERT. 
There are two metric for head pruning, l1 and data-driven. 

For l1 metric, please execute
```
python3 train.py  -m head-pruning -i Path/to/CkptFile -g ./head_pruning/config/l1/config_model_{FRAME_PERIOD}.yaml -c ./head_pruning/config/l1/config_runner_{FRAME_PERIOD}.yaml -n EXP_DIR_PATH -f FRAME_PERIOD
```
For data-driven metric, please execute
```
python3 train.py -m head-pruning -i Path/to/CkptFile -g ./head_pruning/config/data_driven/config_model_{FRAME_PERIOD}.yaml -c ./head_pruning/config/data_driven/config_runner_{FRAME_PERIOD}.yaml -n EXP_DIR_PATH -f FRAME_PERIOD
```
<!-- 
-i: Pre-trained MelHuBERT will be loaded from this .ckpt file \
-g: model config \
-c: runner config \
-n: The model checkpoints, log file, and pre-training config you used will be saved at this directory -->

### Row Pruning 
Execute the following command to do row pruning on a pre-trained MelHuBERT.

Please execute
```
python3 train.py  -m row-pruning -i Path/to/CkptFile -g ./row_pruning/config/config_model_{FRAME_PERIOD}.yaml -c ./row_pruning/config/config_runner_{FRAME_PERIOD}.yaml -n EXP_DIR_PATH -f FRAME_PERIOD
```

### Distillation 
Execute the following command to do knowledge distillation on a pre-trained MelHuBERT teacher. 

Please execute
```
python3 train.py  -m distillation -i Path/to/CkptFile -g ./distillation/config/config_model_{FRAME_PERIOD}.yaml -c ./distillation/config/config_runner_{FRAME_PERIOD}.yaml -n EXP_DIR_PATH -f FRAME_PERIOD
```

<!-- -i: Pre-trained MelHuBERT will be loaded from this .ckpt file \
-g: model config \
-c: runner config \
-n: The model checkpoints, log file, and pre-training config you used will be saved at this directory -->
<!-- 
Choosing between "masked" and "nomasked" for **loss_param.type** in config_model.yaml. \
This parameter controls whether the input would be randomly masked. -->

## Pretrained Models 
- [MelHuBERT-960h-10ms](https://drive.google.com/file/d/18u2u-528uDh5T7R1bp1wvWJ2ygcrNlzx/view?usp=sharing)
- [MelHuBERT-960h-20ms](https://drive.google.com/file/d/1Fn_C5VoH5iV3LdvBEjvfAsbMPhWFFPdd/view?usp=sharing)

## Extracting feature 
Please execute the following command to extract feature from two example waveforms
```
python3 extract_feature.py -m MODE -c CHECKPOINT -f FRAME_PERIOD -d DATASET_SIZE
```

-m: Choose from melhubert, weight-pruning, head-pruning, row-pruning, and distillation \
-c: Model checkpoint path \
-f: Frame period \
-d: Pre-training size of dataset (360 or 960)

## Acknowledgement 
Our implementation of pre-training interface is based on [S3PRL toolkit](https://github.com/s3prl/s3prl) (Shu-wen Yang, Andy T. Liu)
