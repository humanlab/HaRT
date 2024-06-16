CUDA_VISIBLE_DEVICES=2 \
python -O HaRT/run_continue_pt_hart.py \
    --learning_rate 5e-5 \
    --model_name_or_path hlab/hart-gpt2sml-twt-v1 \
    --add_history \
    --initial_history HaRT/initial_history/initialized_history_tensor.pt \
    --extract_layer 11 \
    --do_train \
    --do_eval \
    --output_dir /chronos_data/nisoni/NAACL_Tutorial/outputs/HF_testrun \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 15 \
    --block_size 1024 \
    --max_train_blocks 8 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --train_file /chronos_data/nisoni/NAACL_Tutorial/data/essay_train_table_5_essays.pkl \
    --validation_file /chronos_data/nisoni/NAACL_Tutorial/data/essay_train_table_5_essays.pkl \
    # --overwrite_output_dir \

    
    
