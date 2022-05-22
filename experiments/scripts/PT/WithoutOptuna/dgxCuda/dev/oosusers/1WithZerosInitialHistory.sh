# using the default hyperparams (of GPT2) (layer_ins=2, extract_layer=11)
# block size of 1024
# batch size of 1 per device
# 3 train epochs
# zeros as initial user history

echo $@
CUDA_VISIBLE_DEVICES=$1,$2,$3,$4 \
python -O HULM_AR/run_clm.py \
    --model_name_or_path gpt2 \
    --instantiate_hart \
    --add_history \
    --extract_layer $5 \
    --do_train \
    --do_eval \
    --hostname 130.245.162.235 \
    --db HuLM \
    --train_table fb20lbp_upt50_en_train_1pc_temp \
    --dev_table fb20lbp_upt50_en_non_oosmsgs_temp \
    --output_dir HULM_AR/experiments/outputs/WithoutOptuna/dgxCuda/dev/oosusers/1WithZerosInitialHistory \
    --num_train_epochs 1 \
    --per_device_train_batch_size  1 \
    --per_device_eval_batch_size 27 \
    --block_size 1024 \
    --max_train_blocks 8 \
    # --max_val_blocks 20 \
    # --overwrite_output_dir \
