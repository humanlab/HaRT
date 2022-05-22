echo $@
CUDA_VISIBLE_DEVICES=$1,$2,$3 \
python -O HULM_AR/run_clm_gpt2.py \
    --model_name_or_path /home/nisoni/new_Hulm/HULM_AR/experiments/outputs/PT_GPT2_LM_FT/checkpoint-10525 \
    --do_predict \
    --hostname 130.245.162.235 \
    --db HuLM \
    --dev_table fb20lbp_upt50_en_oosmsgs \
    --output_dir HULM_AR/experiments/outputs/PT_test_ppl/eval_dev/gpt2_LM_FT \
    --per_device_eval_batch_size 684 \
    --block_size 50 \