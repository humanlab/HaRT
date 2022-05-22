echo $@
CUDA_VISIBLE_DEVICES=$1 \
python HaRT/run_ft_hart.py \
    --learning_rate 0.00024447089107483056 \
    --model_name_or_path $2 \
    --task_type user \
    --num_labels $3 \
    --use_history_output \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model $4 \
    --greater_is_better True \
    --metric_for_early_stopping eval_loss \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 25 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --max_train_blocks 4 \
    --output_dir $5 \
    --add_history \
    --initial_history HaRT/initial_history/initialized_history_tensor.pt \
    --train_table $6 \
    --dev_table $7 \
    --test_table $8 \
    # --overwrite_output_dir \

    
    