echo $@
CUDA_VISIBLE_DEVICES=$1 \
python -O HaRT/optuna_trials/run_ft_hart_trials.py \
    --search_params \
    --use_optuna \
    --num_trials 10 \
    --early_stopping_patience 6 \
    --model_name_or_path $2 \
    --task_type document \
    --task_name stance \
    --num_labels 3 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 \
    --greater_is_better True \
    --metric_for_early_stopping eval_loss \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 15 \
    --per_device_train_batch_size  1 \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --max_train_blocks 8 \
    --output_dir $3 \
    --add_history \
    --initial_history HaRT/initial_history/initialized_history_tensor.pt \
    --train_table HaRT/data/datasets/stance/ath_train_all.pkl \
    --dev_table HaRT/data/datasets/stance/ath_dev_all.pkl \
    --test_table HaRT/data/datasets/stance/ath_test_all.pkl \
    # --overwrite_output_dir \

    
    