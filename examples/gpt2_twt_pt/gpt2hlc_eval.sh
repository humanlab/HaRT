python HaRT/run_continue_pt_gpt_twt.py \
    --model_name_or_path hlab/hart-gpt2sml-twt-v1 \
    --do_eval \
    --output_dir outputs/gpt2_twt_wassa_eval \
    --per_device_eval_batch_size 30 \
    --block_size 200 \
    --validation_file $1 \
