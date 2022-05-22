# using the default hyperparams (of GPT2) (layer_ins=2, extract_layer=11)
# block size of 1024
# batch size of 1 per device
# 3 train epochs
# Reasonable embeds as initial user history (created using dlatk as an average of average of word embeds derived from GPT2)
echo $@
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=$1 \
python HULM_AR/run_clm.py \
    --learning_rate 0.00024447089107483056 \
    --model_name_or_path gpt2 \
    --instantiate_hart \
    --add_history \
    --initial_history HULM_AR/initial_history/initialized_history_tensor.pt \
    --extract_layer $2 \
    --do_train \
    --do_eval \
    --hostname 130.245.162.235 \
    --db HuLM \
    --train_table fb20lbp_upt50_en_non_oosmsgs \
    --dev_table fb20lbp_upt50_en_oosmsgs \
    --output_dir dummy \
    --num_train_epochs 1 \
    --per_device_train_batch_size  1 \
    --per_device_eval_batch_size 20 \
    --block_size 50 \
    --max_train_blocks 1 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    # --overwrite_output_dir \
    # --max_val_blocks 20 \

    
    