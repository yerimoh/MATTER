#!/bin/sh

model_save_dir=model/
preds_save_dir=preds/
cache_dir=../.cache/
export CUDA_VISIBLE_DEVICES=1

for model_name in PickyBPE; do
    echo $model_name
    python -u relation_classification2.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir
done
