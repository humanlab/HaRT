echo $@
CUDA_VISIBLE_DEVICES=$1,$2,$3 \
python -O HULM_AR/run_clm_gpt2.py \
    --model_name_or_path gpt2 \
    --do_train \
    --do_eval \
    --do_predict \
    --hostname 130.245.162.235 \
    --db HuLM \
    --train_table fb20lbp_upt50_en_train_10pc \
    --dev_table fb20lbp_upt50_en_oosmsgs \
    --output_dir HULM_AR/experiments/outputs/PT_GPT2_LM_FT \
    --num_train_epochs 5 \
    --per_device_train_batch_size 164 \
    --per_device_eval_batch_size 164 \
    --block_size 50 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
   
    
    