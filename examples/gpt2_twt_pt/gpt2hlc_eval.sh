CUDA_VISIBLE_DEVICES=2 \
python /chronos_data/nisoni/NAACL_Tutorial/HaRT/run_continue_pt_gpt_twt.py \
    --model_name_or_path /chronos_data/nisoni/analysing_hart_user_states/GPT2_TWT_PT_3June2024/outputs/gpt2_twt_pt_60bs \
    --do_eval \
    --output_dir /chronos_data/nisoni/NAACL_Tutorial/outputs/gpt2_twt_wassa_eval \
    --per_device_eval_batch_size 60 \
    --block_size 200 \
    --validation_file /chronos_data/nisoni/NAACL_Tutorial/data/essay_train_table_5_essays.pkl \