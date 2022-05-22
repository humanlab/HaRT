echo $@
CUDA_VISIBLE_DEVICES=$1,$2,$3 \
python -O HULM_AR/run_clm.py \
    --learning_rate 0.00024447089107483056 \
    --model_name_or_path gpt2 \
    --instantiate_hart \
    --add_history \
    --initial_history HULM_AR/initial_history/initialized_history_tensor.pt \
    --extract_layer $4 \
    --hostname 130.245.162.235 \
    --db HuLM \
    --train_table fb20lbp_upt50_en_train_10pc \
    --dev_table fb20lbp_upt50_en_oosmsgs \
    --output_dir HULM_AR/experiments/outputs/hist_ppl_fig/4blocks \
    --num_train_epochs 5 \
    --per_device_train_batch_size  1 \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --max_train_blocks 4 \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    