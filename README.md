# HaRT: Human-aware Recurrent Transformer <br/> [<img src=https://img.shields.io/badge/Download%20Model-green>](https://drive.google.com/file/d/1MGJN1Fp21Q7lPbICNx2_D5qZg8gG0Qla/view?usp=sharing) [<img src=https://img.shields.io/badge/Datasets-yellow>](dataset.md) [<img src=https://img.shields.io/badge/Read%20Paper-blue>](https://aclanthology.org/2022.findings-acl.52) [<img src=https://img.shields.io/badge/Website-purple>](https://nikita-soni-nlp.netlify.app/)


This repository contains the code-base for HaRT, a solution for Human Language Modeling (HuLM).  <br/>
You can read more about the HuLM task and the HaRT model in our paper [Human Language Modeling](https://arxiv.org/pdf/2205.05128.pdf). <br/>
We have also released the [datasets](dataset.md) used for the released version of HaRT model. <br/>
You can also check out our [website](https://nikita-soni-nlp.netlify.app/) for more details.<br/>
Code in this repository is built upon [HuggingFace's codebase](https://github.com/huggingface/transformers).



# Setup

### Requires Python 3.x (tested with Python 3.8)
```
Clone this repository.
pip install -r HaRT/requirements.txt
```
Optional steps for using a conda environment before installing the requirements above:
```
conda create -n hart python==3.8
conda activate hart
```

### [Download](https://drive.google.com/file/d/1MGJN1Fp21Q7lPbICNx2_D5qZg8gG0Qla/view?usp=sharing) HaRT-Twt model
For model information, see our [model_card.md](model_card.md)

# Data input formats

The current code supports csv and pickle files as data inputs for the following usage for fine-tuning and pre-training and evaluation.
Please refer [input_formats.md](input_formats.md) for details on input data formats for pre-training and evaluation, and fine-tuning document- and user-level tasks.

# Downstream Tasks Fine-tuning

## Fine-tuning for document classification with history
### Runs hyperparameter search as well:
Searches for learning rate in the range of 5e-6 to 5e-4. [[run_ft_hart_trials.py](HaRT/optuna_trials/run_ft_hart_trials.py)]
```
HaRT/examples/finetuning_optuna_trials/document_classification.sh \
<gpu_comma_separated_list_in_quotes> \
<num_trials> \
<path_to_pretrained_hart> \
<num_labels> \
<path_to_output> \
<path_to_train_data> <path_to_dev_data> <path_to_test_data>
```
Example Document Sentiment: <br/>
The num_trials tells how many Optuna trials to run for a hyperparameter search sweep.
```
HaRT/examples/finetuning_optuna_trials/document_classification.sh \
"0,1" \
5 \
HaRT/model/hart_pt \
3 \
HaRT/outputs/sentiment \
HaRT/data/datasets/sentiment/sent_train.pkl \
HaRT/data/datasets/sentiment/sent_dev.pkl \
HaRT/data/datasets/sentiment/sent_test.pkl
```

### To fine-tune without hyperparam search (faster but less accurate):
```
HaRT/examples/finetuning/hart/document_classification.sh \
<gpu_comma_separated_list_in_quotes> \
<path_to_pretrained_hart> \
<num_labels> \
<path_to_output> \
<path_to_train_data> \
<path_to_dev_data> \
<path_to_test_data>
```
Example Document Sentiment
```
HaRT/examples/finetuning/hart/document_classification.sh \
"0,1" \
HaRT/model/hart_pt \
3 \
HaRT/outputs/sentiment \
HaRT/data/datasets/sentiment/sent_train.pkl \
HaRT/data/datasets/sentiment/sent_dev.pkl \
HaRT/data/datasets/sentiment/sent_test.pkl
```

### To fine-tune using datasets and hyperparameters from the paper:

For Sentiment Analysis:
```
HaRT/examples/finetuning/hart/sentiment/sent.sh \
<gpu_comma_separated_list_in_quotes> \
<path_to_pretrained_hart> \
<path_to_output>
```
Example
```
HaRT/examples/finetuning/hart/sentiment/sent.sh \
"0,1" \
HaRT/model/hart_pt \
HaRT/outputs/sentiment
```

Please refer [finetune.md](finetune.md) for commands for all the other document-level finetuning tasks from the paper.

## Fine-tuning for user-level tasks
```
HaRT/examples/finetuning/hart/user_level_task.sh \
<gpu_comma_separated_list_in_quotes> \
<path_to_pretrained_hart> \
<num_labels> \
<metric> \
<path_to_output> \
<path_to_train_data> \
<path_to_dev_data> \
<path_to_test_data>
```
Example for a regression task: <br/>
The num_labels will be 1. <br/>
The metric can be either 'eval_r' for pearson r correlation or eval_r_dis for dis-attenuated pearson r correlation.
```
HaRT/examples/finetuning/hart/user_level_task.sh \
"0,1" \
HaRT/model/hart_pt \
1 \
eval_r \
HaRT/outputs/user_regression \
HaRT/data/datasets/user_train.pkl \
HaRT/data/datasets/user_dev.pkl \
HaRT/data/datasets/user_test.pkl
```

Example for a classification task: <br/>
The num_labels will be >1. <br/>
The metric will be 'eval_f1' for weighted F1.
```
HaRT/examples/finetuning/hart/user_level_task.sh \
"0,1" \
HaRT/model/hart_pt \
3 \
eval_f1 \
HaRT/outputs/user_classification \
HaRT/data/datasets/user_train.pkl \
HaRT/data/datasets/user_dev.pkl \
HaRT/data/datasets/user_test.pkl
```


# HuLM Task

## Pre-training
```
HaRT/examples/hulm_pretraining/hart_pt.sh \
<gpu_comma_separated_list_in_quotes> \
<path_to_output> \
<path_to_train_data> \
<path_to_dev_data> \
<path_to_test_data>

```
Example:
```
HaRT/examples/hulm_pretraining/hart_pt.sh \
"0,1" \
HaRT/model/hart_pt \
HaRT/data/datasets/pt_train.pkl \
HaRT/data/datasets/pt_dev.pkl \
HaRT/data/datasets/pt_test.pkl
```

## (Optional) HuLM Hyperparameter search using Optuna trials
The num_sampled_users_for_optuna (default value 5000) tells how many users to sample for Optuna trials to run for a hyperparameter search sweep. <br/>
Searches for learning rate in the range of 5e-6 to 5e-2. [[run_hulm_hart_trials.py](HaRT/optuna_trials/run_hulm_hart_trials.py)]
```
HaRT/examples/hulm_optuna_trials/hart_trials.sh \
<gpu_comma_separated_list_in_quotes> \
<num_sampled_users_for_optuna> \
<path_to_output> \
<path_to_train_data> \
<path_to_dev_data> \

```

Example:
```
HaRT/examples/hulm_optuna_trials/hart_trials.sh \
"0,1" \
5000 \
HaRT/outputs/hart_pt_trials_5000users \
HaRT/data/datasets/pt_train.pkl \
HaRT/data/datasets/pt_dev.pkl
```

> Defaults to 5000 users for running trials. <br/>
[hart_trials.sh](examples/hulm_optuna_trials/hart_trials.sh) can be modified for desired number of users by editing the value for <br/>
--num_users_for_optuna


