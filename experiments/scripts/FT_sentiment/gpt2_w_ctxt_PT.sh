# using the default hyperparams (of GPT2) (layer_ins=2, extract_layer=11)
# block size of 1024
# batch size of 1 per device
# 3 train epochs
# Reasonable embeds as initial user history (created using dlatk as an average of average of word embeds derived from GPT2)
echo $@
CUDA_VISIBLE_DEVICES=$1,$2 \
python -O HULM_AR/run_ft.py \
    --model_name_or_path $3 \
    --task_type document \
    --task_name sentiment \
    --num_labels 3 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 \
    --greater_is_better True \
    --metric_for_early_stopping eval_loss \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 15 \
    --per_device_train_batch_size  1 \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --max_train_blocks 8 \
    --output_dir HULM_AR/experiments/outputs/FT_sentiment_nodups/gpt2_w_ctxt_PT \
    --hostname 130.245.162.235 \
    --db HuLM \
    --train_table sentiment_train_with_history_nodups \
    --dev_table sentiment_train_with_history_nodups \
    --test_table sentiment_test_with_history_nodups \
    --overwrite_output_dir \
    # --max_val_blocks 20 \

    
    