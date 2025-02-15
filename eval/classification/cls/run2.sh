model_save_dir=model2/
preds_save_dir=preds2/
cache_dir=../.cache/
export CUDA_VISIBLE_DEVICES=5


# Define the seed list
seed_list="1 2 3"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done



# Define the seed list
seed_list="4 5 6"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done


# Define the seed list
seed_list="444 555 666"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done



# Define the seed list
seed_list="111 222 333"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done



# Define the seed list
seed_list="444 555 666"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done


# Define the seed list
seed_list="777 888 999"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done
