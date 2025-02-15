#!/bin/bash
python 04.preprocess.py
python 05.tokenization.py

# train
DATA_DIR="data"
MODEL_SAVE_DIR="/result"
VOCAB_FILE="vocab.txt"

export CUDA_VISIBLE_DEVICES=FIX
python3 train.py \
    --data_dir $DATA_DIR \
    --model_save_dir $MODEL_SAVE_DIR \
    --vocab_file $VOCAB_FILE

