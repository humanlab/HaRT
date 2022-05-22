#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning GPT-2HLC for sequence classification."""

import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch.nn as nn

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process, IntervalStrategy

from src.user_ft_trainer import Trainer as user_Trainer
from args.ft_args import DataTrainingArguments, ModelArguments
from src.model_gpt2hlc.finetune_gpt2hlc import GPT2hlcForSequenceClassification
from data.utils_hart.ft_doc_disable_hulm_batching_data_utils import load_dataset as load_no_hulm_dataset 
from data.utils_gpt2hlc.ft_user_data_utils_gpt2hlc import load_dataset as load_user_dataset
from data.data_collator import user_default_data_collator

logger = logging.getLogger(__name__)
    
class EvalLogsCallback(TrainerCallback):

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if 'epoch' in metrics.keys() and control.should_save:
            self.save_metrics('eval_{}'.format(metrics['epoch']), metrics, args)
        elif not control.should_save and state.best_model_checkpoint:
            metrics['best_model_checkpoint'] = state.best_model_checkpoint
            metrics['best_model_metric'] = state.best_metric
        elif 'epoch' not in metrics.keys():
            self.save_metrics('eval_wo_epoch', metrics, args)

    def save_metrics(self, split, metrics, args):
        import json
        
        path = os.path.join(args.output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

class EarlyStoppingCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that handles early stopping.

    Args:
       early_stopping_patience (:obj:`int`):
            Use with :obj:`metric_for_best_model` to stop training when the specified metric worsens for
            :obj:`early_stopping_patience` evaluation calls.
       early_stopping_threshold(:obj:`float`, `optional`):
            Use with TrainingArguments :obj:`metric_for_best_model` and :obj:`early_stopping_patience` to denote how
            much the specified metric must improve to satisfy early stopping conditions. `

    This callback depends on :class:`~transformers.TrainingArguments` argument `load_best_model_at_end` functionality
    to set best_metric in :class:`~transformers.TrainerState`.
    """

    def __init__(self, metric_for_early_stopping, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_for_early_stopping = metric_for_early_stopping
        self.prev_metric_value = None
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        #TODO: use args.greater_is_better which is w.r.t. early stopping metric
        # operator = np.greater if args.greater_is_better else np.less 
        operator = np.less
        if self.prev_metric_value is None or (
            operator(metric_value, self.prev_metric_value)
            and abs(metric_value - self.prev_metric_value) >= self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
        self.prev_metric_value = metric_value 

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
            self.metric_for_early_stopping is not None
        ), "EarlyStoppingCallback requires metric_for_early_stopping to be defined"
        assert (
            args.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = self.metric_for_early_stopping
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_early_stopping, but did not find {metric_to_check} so early stopping is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Labels
    # It can be a classification or a regression task
    num_labels = data_args.num_labels
    if num_labels > 1:
        is_regression = False # classification task
    else:
        is_regression = True # regression task

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name if data_args.task_name is not None else data_args.task_type,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False if 'bertweet' in model_args.model_name_or_path else  model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=None,
    )
    tokenizer.pad_token = tokenizer.pad_token if 'bertweet' in model_args.model_name_or_path else tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    config.freeze_model = model_args.freeze_model
    
    if data_args.task_type=='document':
        if 'bertweet' in model_args.model_name_or_path:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            model = GPT2hlcForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            
    elif data_args.task_type=='user':
        config.add_history = None
        
        def add_insep_token(tokenizer):
            special_tokens_dict = {'sep_token': str('<|insep|>')}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            assert num_added_toks == 1
            assert tokenizer.sep_token == '<|insep|>'
        if tokenizer.sep_token is None:
            add_insep_token(tokenizer)

        model = GPT2hlcForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        def freeze_params(model: nn.Module):
            for par in model.parameters():
                par.requires_grad = False
    
        modules = [
            model.transformer.wte,
            model.transformer.wpe,
            model.transformer.drop,
            model.transformer.h[:10],
            model.transformer.ln_f
        ]
        
        for x in modules:
            freeze_params(x)
    
    def freeze_params(model: nn.Module):
        for par in model.parameters():
            par.requires_grad = False

    if model_args.freeze_model:        
        freeze_params(model.transformer)

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warn(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Preprocessed and padded datasets with labels
    def load_dataset(args):
        if data_args.task_type=='document':
            return load_no_hulm_dataset(*args)
        elif data_args.task_type=='user':
            return load_user_dataset(*args)

    if data_args.train_table is not None or data_args.dev_table is not None or data_args.test_table is not None:
        if data_args.train_table is not None:
            args = [logger, tokenizer, data_args.train_table, block_size, data_args.max_train_blocks, data_args, 'train', True]
            train_dataset, train_uncut_blocks = load_dataset(args) 
        if data_args.dev_table is not None:
            args = [logger, tokenizer, data_args.dev_table, block_size, data_args.max_val_blocks, data_args, 'dev', True]
            eval_dataset, eval_uncut_blocks = load_dataset(args)
        elif data_args.test_table is not None:
            args = [logger, tokenizer, data_args.test_table, block_size, data_args.max_val_blocks, data_args, 'test', True]
            eval_dataset, eval_uncut_blocks = load_dataset(args)
    else:
        raise ValueError("This FT runner requires mysql database tables as train/dev/test data sources currently!")

    def compute_metrics(p: EvalPrediction):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        import scipy

        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=-1)
        
        if hasattr(p, 'user_ids'):
            user_mapper = pd.DataFrame(p.user_ids, columns=['user_id'])
            user_mapper['preds'] = preds
            user_mapper['labels'] = p.label_ids
            assert len(preds) == len(p.label_ids) == len(p.user_ids), "Mismatch in the number of user_ids, predictions and labels!"
            user_mapper = user_mapper.groupby('user_id').agg({'preds':'mean', 'labels':'mean'}).reset_index()
            if data_args.save_preds_labels:
                np.savetxt(training_args.output_dir +'/preds.txt', user_mapper['preds'].to_numpy())
                np.savetxt(training_args.output_dir + '/labels.txt', user_mapper['labels'].to_numpy())

        if is_regression:
            mse = ((user_mapper.preds - user_mapper.labels) ** 2).mean().item()
            r_pear, p_value = scipy.stats.pearsonr(user_mapper.preds, user_mapper.labels)
            # from https://www.aclweb.org/anthology/W18-0604.pdf 
            r_meas1 = 0.77
            r_meas2 = 0.70
            r_dis = r_pear/((r_meas1*r_meas2)**0.5)

            return {
                'mse': mse,
                'r_dis': r_dis,
                'r_pear':r_pear,
                'p_value': p_value
                }
        else:
            indices = p.label_ids!=-100 # make sure to ignore the labels marked as -100
            labels = p.label_ids[indices]
            preds = preds[indices]
            if data_args.save_preds_labels:
                np.savetxt(training_args.output_dir +'/preds.txt', preds, fmt='%d')
                np.savetxt(training_args.output_dir + '/labels.txt', labels, fmt='%d')
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }

    # Data collator
    if data_args.task_type=='user':
        # This one will take care of collating batches with user_ids, labels, input_ids and attention_mask
        data_collator = user_default_data_collator
    else:
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator = default_data_collator


    # Initialize our Trainer
    if data_args.task_type=='document':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[
                    EvalLogsCallback, 
                    EarlyStoppingCallback(
                        model_args.metric_for_early_stopping, 
                        model_args.early_stopping_patience,
                        model_args.early_stopping_threshold
                    )
                    ] if training_args.do_train else None
        )
    elif data_args.task_type=='user':
        trainer = user_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[
                    EvalLogsCallback, 
                    EarlyStoppingCallback(
                        model_args.metric_for_early_stopping, 
                        model_args.early_stopping_patience,
                        model_args.early_stopping_threshold
                    )
                    ] if training_args.do_train else None
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        else: 
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics['seed'] = training_args.seed
        metrics['pretrained_model_loc'] = model_args.model_name_or_path
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        metrics["train_blocks_per_sample"] = train_uncut_blocks if data_args.max_train_blocks is None else min(data_args.max_train_blocks, train_uncut_blocks)
        metrics["block_size"] = block_size
        metrics["gpus"] = training_args.n_gpu
        metrics["total_epochs"] = training_args.num_train_epochs
        metrics["per_device_train_batch_size"] = training_args.per_device_train_batch_size
        metrics["train_table"] = data_args.train_table
        metrics["dev_table"] = data_args.dev_table

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval and not training_args.do_train:
        if data_args.dev_table is not None:
            logger.info("*** Evaluate Dev set ***")
            eval_test('dev', data_args, training_args, eval_dataset, eval_uncut_blocks, trainer)
        elif data_args.test_table is not None:
            args = [logger, tokenizer, data_args.test_table, block_size, data_args.max_val_blocks, data_args, 'test', "True"]
            eval_dataset, eval_uncut_blocks = load_dataset(args)
            logger.info("*** Evaluate Test set ***")
            eval_test('test', data_args, training_args, eval_dataset, eval_uncut_blocks, trainer)
        else:
            raise ValueError("Expecting dev or test data to run eval.")
    
    # Evaluation
    if training_args.do_predict:
        if data_args.test_table is None:
            raise ValueError("You are trying to predict on test data but forgot to provide a test data source path!")

        if data_args.dev_table is not None:
            args = [logger, tokenizer, data_args.test_table, block_size, data_args.max_val_blocks, data_args, 'test', True]
            eval_dataset, eval_uncut_blocks = load_dataset(args)

        logger.info("*** Evaluate Test set ***")
        eval_test('test', data_args, training_args, eval_dataset, eval_uncut_blocks, trainer)
        
        
def eval_test(test_type, data_args, training_args, eval_dataset, eval_uncut_blocks, trainer):
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
    metrics["test_samples"] = min(max_eval_samples, len(eval_dataset))
    metrics["test_blocks_per_sample"] = eval_uncut_blocks if data_args.max_val_blocks is None else min(data_args.max_val_blocks, eval_uncut_blocks)
    metrics["per_device_test_batch_size"] = training_args.per_device_eval_batch_size
    metrics["test_table"] = data_args.test_table
    trainer.log_metrics("eval_{}".format(test_type), metrics)
    trainer.save_metrics("eval_{}".format(test_type), metrics)
      
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
