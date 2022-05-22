# using the default hyperparams (of GPT2) (layer_ins=2, extract_layer=11)
# block size of 1024
# batch size of 1 per device
# 3 train epochs
# Reasonable embeds as initial user history (created using dlatk as an average of average of word embeds derived from GPT2)
echo $@
CUDA_VISIBLE_DEVICES=$1,$2 \
python -O HULM_AR/run_ft_gpt2.py \
    --learning_rate 0.00018840181749761617 \
    --model_name_or_path $3 \
    --task_type document \
    --task_name stance \
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
    --per_device_train_batch_size 100 \
    --per_device_eval_batch_size 100 \
    --block_size 50 \
    --output_dir /home/nisoni/new_models/FT/gpt2LMFT_ctlb_fb/stance/fem \
    --hostname 130.245.162.235 \
    --db stance_hulm \
    --train_table stance_fem_train \
    --dev_table stance_fem_dev \
    --test_table stance_fem_test \
    --overwrite_output_dir \
    # --max_val_blocks 20 \

    
    