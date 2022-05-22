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
""" Finetuning HaRT for sequence classification."""

import logging
import os
import sys
from typing import Optional

import numpy as np
import torch.nn as nn

import transformers
from transformers import (
    AutoConfig,
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

from args.ft_args import DataTrainingArguments, ModelArguments
from src.model.configuration_hart import HaRTConfig
from src.model.modeling_hart import HaRTBaseLMHeadModel
from src.model.hart import HaRTPreTrainedModel
from src.model.finetune_hart import HaRTForSequenceClassification
from data.utils_hart.ft_doc_disable_hulm_batching_data_utils import load_dataset as load_no_hulm_dataset
from data.utils_hart.ft_doc_data_utils import load_dataset as load_doc_dataset
from data.utils_hart.ft_user_data_utils import load_dataset as load_user_dataset
from data.data_collator import DataCollatorWithPaddingForHaRT

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
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
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
    set_seed(model_args.init_seed)

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
    if model_args.load_non_PT_hulm_model:
        config = HaRTConfig(
            num_labels=num_labels,
            finetuning_task=data_args.task_name if data_args.task_name is not None else data_args.task_type,
            use_history_output=data_args.use_history_output
            )
        if model_args.add_history:
            config.add_history = True
        if model_args.use_qh05_wts:
            config.use_qh05_wts = True
        else:
            config.use_qh05_wts = False
        
        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        def add_insep_token(tokenizer):
            special_tokens_dict = {'sep_token': str('<|insep|>')}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            assert num_added_toks == 1
            assert tokenizer.sep_token == '<|insep|>'
        add_insep_token(tokenizer)
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name if data_args.task_name is not None else data_args.task_type,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config.use_history_output=data_args.use_history_output
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    
    config.pad_token_id = tokenizer.eos_token_id
    config.freeze_model = model_args.freeze_model
    config.use_hart_no_hist = model_args.use_hart_no_hist
    
    if training_args.do_train and not model_args.load_non_PT_hulm_model:
        model = HaRTForSequenceClassification(config, model_args.model_name_or_path)
    elif training_args.do_train and model_args.load_non_PT_hulm_model:
        hartbaseLMModel = HaRTBaseLMHeadModel.from_pretrained(model_args.model_name_or_path, config=config)
        hartbaseLMModel.resize_token_embeddings(len(tokenizer))
        hart = HaRTPreTrainedModel(config, hartbaseLMModel)
        model = HaRTForSequenceClassification(config, pt_model=hart)
    elif training_args.do_eval and not training_args.do_train:
        model = HaRTForSequenceClassification.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError("You're neither training nor evaluating. Can't pick a model because I don't know what do you want to do.")

    def freeze_params(model: nn.Module):
        for par in model.parameters():
            par.requires_grad = False

    if model_args.freeze_model:        
        freeze_params(model.transformer)
    
    if data_args.task_type=='user':
        freeze_params(model.transformer.transformer)
    
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
            if model_args.use_hart_no_hist:
                return load_no_hulm_dataset(*args)
            else:
                return load_doc_dataset(*args)
        elif data_args.task_type=='user':
            return load_user_dataset(*args)

    if data_args.train_table is not None or data_args.dev_table is not None or data_args.test_table is not None:
        if data_args.train_table is not None:
            args = [logger, tokenizer, data_args.train_table, block_size, data_args.max_train_blocks, data_args, 'train', data_args.disable_hulm_batching]
            train_dataset, train_uncut_blocks = load_dataset(args) 
        if data_args.dev_table is not None:
            args = [logger, tokenizer, data_args.dev_table, block_size, data_args.max_val_blocks, data_args, 'dev', data_args.disable_hulm_batching]
            eval_dataset, eval_uncut_blocks = load_dataset(args)
        elif data_args.test_table is not None:
            args = [logger, tokenizer, data_args.test_table, block_size, data_args.max_val_blocks, data_args, 'test', data_args.disable_hulm_batching]
            eval_dataset, eval_uncut_blocks = load_dataset(args)
    else:
        raise ValueError("This FT runner requires train/dev/test data source paths currently!")

    def compute_metrics(p: EvalPrediction):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        import scipy

        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=-1)

        if is_regression:
            if data_args.save_preds_labels:
                np.savetxt(training_args.output_dir +'/preds.txt', preds)
                np.savetxt(training_args.output_dir + '/labels.txt', p.label_ids)
            mse = ((preds - p.label_ids) ** 2).mean().item()
            r_pear, p_value = scipy.stats.pearsonr(preds, p.label_ids)
            # from https://www.aclweb.org/anthology/W18-0604.pdf 
            r_meas1 = 0.77
            r_meas2 = 0.70
            r_dis = r_pear/((r_meas1*r_meas2)**0.5)

            return {
                'mse': mse,
                'r_dis': r_dis,
                'r_pear': r_pear,
                'p_value': p_value
                }
        else:
            indices = p.label_ids!=-100 # make sure to ignore the labels marked as -100
            labels = p.label_ids[indices]
            if not model_args.use_hart_no_hist:
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
    if not data_args.disable_hulm_batching:
        # This one will take care of collating batches of type [users, blocks, block_size]
        data_collator = DataCollatorWithPaddingForHaRT(model_args, config, tokenizer, is_ft=True, is_user_level_ft=data_args.task_type=='user')
    else:
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator = default_data_collator


    # Initialize our Trainer
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
        metrics['model_seed'] = model_args.init_seed
        metrics['train_seed'] = training_args.seed
        metrics['lr'] = training_args.learning_rate
        metrics['pretrained_model_loc'] = model_args.model_name_or_path
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        metrics["train_blocks_per_sample"] = train_uncut_blocks if data_args.max_train_blocks is None else min(data_args.max_train_blocks, train_uncut_blocks)
        metrics["block_size"] = block_size
        metrics["gpus"] = training_args.n_gpu
        metrics["total_epochs"] = training_args.num_train_epochs
        metrics["per_device_train_batch_size"] = training_args.per_device_train_batch_size
        metrics["train_table"] = data_args.train_table
        metrics["dev_table"] = data_args.dev_table
        if config.add_history:
            metrics["history"] = model_args.add_history
            metrics["extract_layer"] = config.extract_layer if config.extract_layer else None
            metrics["layer_ins"] = config.layer_ins if config.layer_ins else None
            if model_args.add_history:
                metrics["0s_initial_history"] = False if model_args.initial_history else True
        

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval and data_args.dev_table is not None and not training_args.do_train:
        if data_args.dev_table is not None:
            logger.info("*** Evaluate Dev set ***")
            eval_test('dev', data_args, training_args, eval_dataset, eval_uncut_blocks, trainer)
        elif data_args.test_table is not None:
            args = [logger, tokenizer, data_args.test_table, block_size, data_args.max_val_blocks, data_args, 'test', data_args.disable_hulm_batching]
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
            args = [logger, tokenizer, data_args.test_table, block_size, data_args.max_val_blocks, data_args, 'test', data_args.disable_hulm_batching]
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
