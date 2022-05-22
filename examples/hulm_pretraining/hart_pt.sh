echo $@
CUDA_VISIBLE_DEVICES=$1 \
python -O HaRT/run_hulm_hart.py \
    --learning_rate 0.00024447089107483056 \
    --model_name_or_path gpt2 \
    --instantiate_hart \
    --add_history \
    --initial_history HaRT/initial_history/initialized_history_tensor.pt \
    --extract_layer 11 \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir $2 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 15 \
    --block_size 1024 \
    --max_train_blocks 8 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --train_table $3 \
    --dev_table $4 \
    --test_table $5 \
    # --overwrite_output_dir \

    
    
