model_save_dir=model3/
preds_save_dir=preds3/
cache_dir=../.cache/
export CUDA_VISIBLE_DEVICES=6


# Define the seed list
seed_list="15 20 25"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done



# Define the seed list
seed_list="5 55 555"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done


# Define the seed list
seed_list="1 11 111"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done



# Define the seed list
seed_list="2 22 222"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done



# Define the seed list
seed_list="3 33 333"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done



# Define the seed list
seed_list="4 44 444"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done

# Define the seed list
seed_list="6 66 666"

for model_name in sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done