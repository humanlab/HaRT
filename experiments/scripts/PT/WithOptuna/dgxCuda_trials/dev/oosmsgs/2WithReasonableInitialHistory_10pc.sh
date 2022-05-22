# using the default hyperparams (of GPT2) (layer_ins=2, extract_layer=11)
# block size of 1024
# batch size of 1 per device
# 3 train epochs
# Reasonable embeds as initial user history (created using dlatk as an average of average of word embeds derived from GPT2)
echo $@
CUDA_VISIBLE_DEVICES=$1,$2,$3 \
python -O HULM_AR/run_clm_trials.py \
    --model_name_or_path gpt2 \
    --instantiate_hart \
    --search_params \
    --use_optuna \
    --add_history \
    --initial_history HULM_AR/initial_history/initialized_history_tensor.pt \
    --extract_layer $4 \
    --hostname 130.245.162.235 \
    --db HuLM \
    --train_table fb20lbp_upt50_en_train_10pc \
    --dev_table fb20lbp_upt50_en_oosmsgs \
    --output_dir HULM_AR/experiments/outputs/WithoutOptuna/dgxCuda_trials_10pc/dev/oosmsgs/WithReasonableHistory_5e4_5e6_7epochs \
    --num_train_epochs 7 \
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
    # for ray tune, must give absolute path for initial_history and output_dir

    
    