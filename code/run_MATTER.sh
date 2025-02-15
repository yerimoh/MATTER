export CUDA_VISIBLE_DEVICES=FIX

python 01.material_weighting.py
python 02.makefrequent.py
python 03.DMM.py --lambda_value 1.0
