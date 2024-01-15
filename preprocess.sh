#!/bin/bash

# Preparing data
rm -rf libri-960-kaldi-data
mkdir libri-960-kaldi-data
tar -xvf $1 -C libri-960-kaldi-data/
mv libri-960-kaldi-data/stage2-cluster-20ms/split200/* libri-960-kaldi-data/stage2-cluster-20ms/
rm -rf libri-960-kaldi-data/stage2-cluster-20ms/split200/

# Extracting the data and the cluster 
python3 preprocess/tidy_libri960_kaldi_data.py libri-960-kaldi-data $2
