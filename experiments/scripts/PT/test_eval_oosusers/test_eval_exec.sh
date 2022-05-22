/home/nisoni/new_Hulm/HULM_AR/experiments/scripts/PT/test_eval_oosusers/eval_test_scoped.sh $@ /home/nisoni/Fourth_expmts_hulm/HULM_AR/experiments/outputs_pre_early_stopping/WithoutOptuna/dgxCuda_trials_10pc/dev/oosmsgs/WithReasonableHistory_5e4_5e6_7epochs/run-0/checkpoint-9400

/home/nisoni/new_Hulm/HULM_AR/experiments/scripts/PT/test_eval_oosusers/eval_test_hulm_wo_recurrence.sh $@ /home/nisoni/Fourth_expmts_hulm/HULM_AR/experiments/outputs_pre_early_stopping/WithoutOptuna/dgxCuda_trials_10pc/dev/oosmsgs/0WithoutHistory_10pc/run-2/checkpoint-9400

# /home/nisoni/new_Hulm/HULM_AR/experiments/scripts/PT/test_eval_oosusers/eval_test_1pc.sh $@ /home/nisoni/Fourth_expmts_hulm/HULM_AR/experiments/outputs_pre_early_stopping/WithoutOptuna/dgxCuda_trials/dev/oosmsgs/2WithReasonableInitialHistory/run-0/checkpoint-399

#commentine above and using 1pc comparable:

HULM_AR/experiments/scripts/PT/test_eval_oosusers/eval_test_1pc.sh $@ /home/nisoni/new_Hulm/HULM_AR/experiments/outputs/PT/scoped_1pc_comparable/checkpoint-712


/home/nisoni/new_Hulm/HULM_AR/experiments/scripts/PT/test_eval_oosusers/eval_test_6blocks.sh $@ /home/nisoni/new_Hulm/HULM_AR/experiments/outputs/hist_ppl_fig/6blocks/checkpoint-9400

/home/nisoni/new_Hulm/HULM_AR/experiments/scripts/PT/test_eval_oosusers/eval_test_4blocks.sh $@ /home/nisoni/new_Hulm/HULM_AR/experiments/outputs/hist_ppl_fig/4blocks/checkpoint-9400

/home/nisoni/new_Hulm/HULM_AR/experiments/scripts/PT/test_eval_oosusers/eval_test_2blocks.sh $@ /home/nisoni/new_Hulm/HULM_AR/experiments/outputs/hist_ppl_fig/2blocks/checkpoint-9400

/home/nisoni/new_Hulm/HULM_AR/experiments/scripts/PT/test_eval_oosusers/eval_test_1block.sh $@ /home/nisoni/new_Hulm/HULM_AR/experiments/outputs/hist_ppl_fig/1block/checkpoint-9400

