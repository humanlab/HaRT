CUDA_VISIBLE_DEVICES=2 \
python /chronos_data/nisoni/NAACL_Tutorial/HaRT/run_continue_pt_gpt_twt.py \
    --model_name_or_path /chronos_data/nisoni/analysing_hart_user_states/GPT2_TWT_PT_3June2024/outputs/gpt2_twt_pt_60bs \
    --do_train \
    --do_eval \
    --output_dir /chronos_data/nisoni/NAACL_Tutorial/outputs/gpt2_twt_wassa \
    --num_train_epochs 5 \
    --per_device_train_batch_size 60 \
    --per_device_eval_batch_size 60 \
    --block_size 200 \
    --max_train_blocks 8 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --train_file /chronos_data/nisoni/NAACL_Tutorial/data/essay_train_table_5_essays.pkl \
    --validation_file /chronos_data/nisoni/NAACL_Tutorial/data/essay_train_table_5_essays.pkl \