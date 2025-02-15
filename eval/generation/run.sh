export CUDA_VISIBLE_DEVICES=2

basemodel="MATTER"
echo "Running model: $basemodel"
# Execute the Python script with the specified basemodel and seed list
python main.py --basemodel $basemodel --seed_list 1 11 111


