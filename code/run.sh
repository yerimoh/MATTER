export CUDA_VISIBLE_DEVICES=FIX

python 00.material_weighting.py
python 01.makefrequent.py
python 02.DMM.py --lambda_value 1.0
