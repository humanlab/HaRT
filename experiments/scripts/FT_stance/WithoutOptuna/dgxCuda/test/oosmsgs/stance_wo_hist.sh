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
    --num_labels 3 \
    --do_eval \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --output_dir HULM_AR/experiments/outputs/FT_nodups/WithoutOptuna/test/oosmsgs/stance_wo_hist_best_FT \
    --instantiate_hart \
    --hostname 130.245.162.235 \
    --db stance_hulm \
    --test_table stance_abo_test_with_history_nodups \
    --overwrite_output_dir \
    # --max_val_blocks 20 \

    
    
