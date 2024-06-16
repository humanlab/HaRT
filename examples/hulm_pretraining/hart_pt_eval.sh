CUDA_VISIBLE_DEVICES=2 \
python -O HaRT/run_continue_pt_hart.py \
    --model_name_or_path /chronos_data/nisoni/HaRT_Twt_model \
    --add_history \
    --initial_history HaRT/initial_history/initialized_history_tensor.pt \
    --extract_layer 11 \
    --do_eval \
    --output_dir /chronos_data/nisoni/NAACL_Tutorial/outputs/hart_wassa_eval \
    --per_device_eval_batch_size 15 \
    --block_size 1024 \
    --validation_file /chronos_data/nisoni/NAACL_Tutorial/data/essay_train_table_5_essays.pkl \
    # --overwrite_output_dir \

    
    
