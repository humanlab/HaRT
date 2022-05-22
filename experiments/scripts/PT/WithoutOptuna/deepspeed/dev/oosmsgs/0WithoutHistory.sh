# using the default hyperparams (of GPT2)
# block size of 1024
# batch size of 1 per device
# 3 train epochs
# no user history

deepspeed --include localhost:0,1,2,3,4,5,6,7 \
HULM_AR/run_clm.py \
    --model_name_or_path gpt2 \
    --instantiate_hart \
    --do_train \
    --do_eval \
    --hostname 130.245.162.235 \
    --db HuLM \
    --train_table fb20lbp_upt50_en_train_1pc \
    --dev_table fb20lbp_upt50_en_oosmsgs \
    --output_dir HULM_AR/experiments/outputs/WithoutOptuna/deepspeed/dev/oosmsgs/0WithoutHistoryMG \
    --num_train_epochs 1 \
    --per_device_train_batch_size  1 \
    --per_device_eval_batch_size 1 \
    --block_size 1024 \
    --max_train_blocks 14 \
    --max_val_blocks 14 \
    --deepspeed HULM_AR/config/ds_config.json \
      --overwrite_output_dir \