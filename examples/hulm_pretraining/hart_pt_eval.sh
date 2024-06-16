python -O HaRT/run_continue_pt_hart.py \
    --model_name_or_path hlab/hart-gpt2sml-twt-v1 \
    --add_history \
    --initial_history HaRT/initial_history/initialized_history_tensor.pt \
    --extract_layer 11 \
    --do_eval \
    --output_dir outputs/hart_wassa_eval \
    --per_device_eval_batch_size 15 \
    --block_size 1024 \
    --validation_file $1 \
    --overwrite_output_dir \

    
    
