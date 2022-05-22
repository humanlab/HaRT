# using the default hyperparams (of GPT2)
# block size of 1024
# batch size of 1 per device
# 3 train epochs
# no user history
echo $@
CUDA_VISIBLE_DEVICES=$1,$2,$3 \
python -O HULM_AR/run_clm_trials.py \
    --model_name_or_path gpt2 \
    --instantiate_hart \
    --search_params \
    --use_optuna \
    --hostname 130.245.162.235 \
    --db HuLM \
    --train_table fb20lbp_upt50_en_train_10pc \
    --dev_table fb20lbp_upt50_en_oosmsgs \
    --output_dir HULM_AR/experiments/outputs/WithoutOptuna/dgxCuda_trials_10pc/dev/oosmsgs/0WithoutHistory_10pc \
    --num_train_epochs 5 \
    --per_device_train_batch_size  1 \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --max_train_blocks 8 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy steps \
    --save_steps 705 \
    # --overwrite_output_dir \
    # --max_val_blocks 20 \
      