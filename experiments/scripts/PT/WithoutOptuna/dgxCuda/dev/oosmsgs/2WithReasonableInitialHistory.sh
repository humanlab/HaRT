# using the default hyperparams (of GPT2) (layer_ins=2, extract_layer=11)
# block size of 1024
# batch size of 1 per device
# 3 train epochs
# Reasonable embeds as initial user history (created using dlatk as an average of average of word embeds derived from GPT2)
echo $@
CUDA_VISIBLE_DEVICES=$1,$2,$3 \
python -O HULM_AR/run_clm.py \
    --model_name_or_path gpt2 \
    --learning_rate 0.00021535876141510403 \
    --instantiate_hart \
    --add_history \
    --initial_history HULM_AR/initial_history/initialized_history_tensor.pt \
    --extract_layer $4 \
    --do_train \
    --do_eval \
    --do_predict \
    --hostname 130.245.162.235 \
    --db HuLM \
    --train_table fb20lbp_upt50_en_train_1pc \
    --dev_table fb20lbp_upt50_en_oosmsgs \
    --output_dir HULM_AR/experiments/outputs/PT/scoped_1pc_comparable \
    --num_train_epochs 5 \
    --per_device_train_batch_size  1 \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --max_train_blocks 8 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    # --overwrite_output_dir \
    # --max_val_blocks 20 \

    
    