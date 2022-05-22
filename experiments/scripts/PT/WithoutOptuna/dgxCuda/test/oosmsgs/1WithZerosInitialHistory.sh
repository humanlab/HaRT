# using the default hyperparams (of GPT2) (layer_ins=2, extract_layer=11)
# block size of 1024
# batch size of 1 per device
# 1 train epoch
# zeros as initial user history

CUDA_VISIBLE_DEVICES=0 \
python -O hulm_ar/HULM_AR/run_clm.py \
    --model_name_or_path gpt2 \
    --instantiate_hart \
    --add_history \
    --do_train \
    --do_eval \
    --hostname localhost \
    --db HuLM \
    --train_table fb20lbp_upt50_en_train_10pc \
    --test_table fb20lbp_upt50_en_oosmsgs \
    --output_dir hulm_ar/HULM_AR/experiments/outputs/WithoutOptuna/dgxCuda/test/oosmsgs/1WithZerosInitialHistory \
    --num_train_epochs 1 \
    --per_device_train_batch_size  1 \
    --per_device_eval_batch_size 1 \
    --block_size 1024 \
    --max_train_blocks 2 \
    --max_val_blocks 2 \
    # --overwrite_output_dir \
