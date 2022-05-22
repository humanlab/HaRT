echo $@
CUDA_VISIBLE_DEVICES=$1 \
python -O HaRT/optuna_trials/run_hulm_hart_trials.py \
    --search_params \
    --use_optuna \
    --num_trials 5 \
    --num_users_for_optuna $2 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --instantiate_hart \
    --model_name_or_path gpt2 \
    --add_history \
    --initial_history HaRT/initial_history/initialized_history_tensor.pt \
    --hostname localhost \
    --db HuLM \
    --output_dir $3 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 15 \
    --block_size 1024 \
    --max_train_blocks 8 \
    --train_table $4 \
    --dev_table $5 \
    # --overwrite_output_dir \

    
    
