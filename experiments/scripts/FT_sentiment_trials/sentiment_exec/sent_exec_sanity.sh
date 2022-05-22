HULM_AR/experiments/scripts/FT_sentiment_trials/gpt2_w_ctxt_PT_seed50.sh $@ /home/nisoni/Fourth_expmts_hulm/HULM_AR/experiments/outputs_pre_early_stopping/WithoutOptuna/dgxCuda_trials_10pc/dev/oosmsgs/0WithoutHistory_10pc/run-2/checkpoint-9400
HULM_AR/experiments/scripts/FT_sentiment_trials/gpt2_w_ctxt_PT_seed32.sh $@ /home/nisoni/Fourth_expmts_hulm/HULM_AR/experiments/outputs_pre_early_stopping/WithoutOptuna/dgxCuda_trials_10pc/dev/oosmsgs/0WithoutHistory_10pc/run-2/checkpoint-9400

HULM_AR/experiments/scripts/FT_sentiment_trials/scoped_random.sh $@ gpt2
HULM_AR/experiments/scripts/FT_sentiment_trials/scoped_1pc.sh $@ /home/nisoni/Fourth_expmts_hulm/HULM_AR/experiments/outputs_pre_early_stopping/WithoutOptuna/dgxCuda_trials/dev/oosmsgs/2WithReasonableInitialHistory/run-0/checkpoint-399

# 1pc comparable:
HULM_AR/experiments/scripts/FT_sentiment_trials/scoped_1pc.sh $@ /home/nisoni/new_Hulm/HULM_AR/experiments/outputs/PT/scoped_1pc_comparable/checkpoint-712