# using the default hyperparams (of GPT2)
# block size of 1024
# batch size of 1 per device
# 3 train epochs
# no user history
echo $@
CUDA_VISIBLE_DEVICES=$1,$2,$3,$4 \
python -O HULM_AR/run_clm.py \
    --model_name_or_path gpt2 \
    --do_train \
    --do_eval \
    --hostname 130.245.162.235 \
    --db HuLM \
    --train_table fb20lbp_upt50_en_train_1pc_temp \
    --dev_table fb20lbp_upt50_en_oosmsgs \
    --output_dir HULM_AR/experiments/outputs/WithoutOptuna/dgxCuda/dev/oosmsgs/FlatList25eval \
    --num_train_epochs 5 \
    --per_device_train_batch_size  6 \
    --per_device_eval_batch_size 25 \
    --block_size 1024 \
    --max_train_blocks 8 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    # --overwrite_output_dir \
    # --max_val_blocks 20 \
      