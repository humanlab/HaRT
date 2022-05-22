echo $@
CUDA_VISIBLE_DEVICES=$1,$2,$3 \
python -O HULM_AR/run_clm.py \
    --model_name_or_path $4 \
    --add_history \
    --initial_history HULM_AR/initial_history/initialized_history_tensor.pt \
    --do_predict \
    --hostname 130.245.162.235 \
    --db HuLM \
    --test_table fb20lbp_upt50_en_oosmsgs \
    --output_dir HULM_AR/experiments/outputs/PT_test_ppl/scoped \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    # --max_val_blocks 8 \