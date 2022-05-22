# Fine-tune using datasets and hyperparameters from the paper

For Sentiment Analysis:
```
HaRT/examples/finetuning/hart/sentiment/sent.sh <gpu_comma_separated_list_in_quotes> <path_to_pretrained_hart> <path_to_output>
```
Example:
```
HaRT/examples/finetuning/hart/sentiment/sent.sh "0,1" HaRT/model/hart_pt HaRT/outputs/sentiment
```

Similarly, for all Stance Detection Topics:
```
HaRT/examples/finetuning/hart/stance/abo.sh "0,1" HaRT/model/hart_pt HaRT/outputs/stance/abo
HaRT/examples/finetuning/hart/stance/ath.sh "0,1" HaRT/model/hart_pt HaRT/outputs/stance/ath
HaRT/examples/finetuning/hart/stance/clim.sh "0,1" HaRT/model/hart_pt HaRT/outputs/stance/clim
HaRT/examples/finetuning/hart/stance/clin.sh "0,1" HaRT/model/hart_pt HaRT/outputs/stance/clin
HaRT/examples/finetuning/hart/stance/fem.sh "0,1" HaRT/model/hart_pt HaRT/outputs/stance/fem
```




## (Optional) Hyperparamaters search using Optuna Trials

For Sentiment Analysis:
```
HaRT/examples/finetuning_optuna_trials/hart/sentiment/sent.sh <gpu_comma_separated_list_in_quotes> <path_to_pretrained_hart> <path_to_output>
```
Example:
```
HaRT/examples/finetuning_optuna_trials/hart/sentiment/sent.sh "0,1" HaRT/model/hart_pt HaRT/outputs/sentiment_trials
```

Similarly, for all Stance Detection Topics:
```
HaRT/examples/finetuning_optuna_trials/hart/stance/abo.sh "0,1" HaRT/model/hart_pt HaRT/outputs/stance_trials/abo
HaRT/examples/finetuning_optuna_trials/hart/stance/ath.sh "0,1" HaRT/model/hart_pt HaRT/outputs/stance_trials/ath
HaRT/examples/finetuning_optuna_trials/hart/stance/clim.sh "0,1" HaRT/model/hart_pt HaRT/outputs/stance_trials/clim
HaRT/examples/finetuning_optuna_trials/hart/stance/clin.sh "0,1" HaRT/model/hart_pt HaRT/outputs/stance_trials/clin
HaRT/examples/finetuning_optuna_trials/hart/stance/fem.sh "0,1" HaRT/model/hart_pt HaRT/outputs/stance_trials/fem

```
