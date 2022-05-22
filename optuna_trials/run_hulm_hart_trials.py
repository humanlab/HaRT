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
"""
Running Optuna trials for fine-tuning HaRT for human language modeling.
"""

import logging
import copy
import json
import math
import os
import sys
sys.path.insert(1, '/home/nisoni/HaRT/HaRT')
from typing import Dict

from args.clm_args import DataTrainingArguments, ModelArguments
from src.model.hart import HaRTPreTrainedModel
from src.model.modeling_hart import HaRTBaseLMHeadModel
from src.model.configuration_hart import HaRTConfig
from data.utils_optuna_trials.hulm_sample_data_utils import load_dataset
from data.data_collator import DataCollatorWithPaddingForHaRT

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class evalLogsCallback(TrainerCallback):

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs['metrics']
        if control.should_save:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
            metrics['trial_params'] = json.dumps(state.trial_params)
        self.metrics = metrics.copy()
        logger.info(json.dumps(metrics))      
    
    def on_save(self, args, state, control, **kwargs):
        output_dir = state.best_model_checkpoint.split('/checkpoint')[0]
        self.save_metrics('eval_{}'.format(self.metrics['epoch']), self.metrics, output_dir)
        logger.info("Saving eval metrics after epoch {} into {}".format(self.metrics['epoch'], output_dir))
    
    def on_train_end(self, args, state, control, **kwargs):
        output_dir = state.best_model_checkpoint.split('/checkpoint')[0]
        metrics = state.trial_params.copy()
        metrics["number_of_gpus"] = args.n_gpu
        metrics["best_loss"] = state.best_metric
        metrics["best_perplexity"] = math.exp(state.best_metric)
        metrics["best_model_checkpoint"] = state.best_model_checkpoint
        self.metrics = metrics
        self.save_metrics('final', self.metrics, output_dir)
    
    def save_metrics(self, split, metrics, output_dir, combined=True):
        path = os.path.join(output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)
     
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
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.instantiate_hart:
        config = HaRTConfig()
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    
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
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    def add_insep_token(tokenizer):
        special_tokens_dict = {'sep_token': str('<|insep|>')}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        assert num_added_toks == 1
        assert tokenizer.sep_token == '<|insep|>'
    
    add_insep_token(tokenizer)

    hartbaseLMmodel = HaRTBaseLMHeadModel.from_pretrained(model_args.model_name_or_path, config=config)
    hartbaseLMmodel.resize_token_embeddings(len(tokenizer))

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

    if data_args.train_table is not None or data_args.dev_table is not None or data_args.test_table is not None:
        if data_args.train_table is not None:
            train_dataset, train_uncut_blocks = load_dataset(logger, tokenizer, data_args.train_table, block_size, data_args.max_train_blocks, data_args, 'train', data_args.disable_hulm_batching)
        if data_args.dev_table is not None:
            eval_dataset, eval_uncut_blocks = load_dataset(logger, tokenizer, data_args.dev_table, block_size, data_args.max_val_blocks, data_args, 'dev', data_args.disable_hulm_batching)
        elif data_args.test_table is not None:
            eval_dataset, eval_uncut_blocks = load_dataset(logger, tokenizer, data_args.test_table, block_size, data_args.max_val_blocks, data_args, 'test', data_args.disable_hulm_batching)
    else:
        raise ValueError("This CLM runner requires mysql database tables as train/dev/test data sources currently!")

    # Data collator
    if model_args.instantiate_hart:
        # This one will take care of collating batches of type [users, windows, num_tokens]
        data_collator = DataCollatorWithPaddingForHaRT(model_args, config, tokenizer, training_args.deepspeed)
    else:
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator = default_data_collator

    if model_args.search_params:

        def model_init():
            # Set seed before initializing model.
            set_seed(training_args.seed)
            model = HaRTPreTrainedModel(config, hartbaseLMmodel)
            model.transformer.resize_token_embeddings(len(tokenizer))
            model.resize_token_embeddings(len(tokenizer))
            return model
        
        ####### NOTE: Ray Hyperparameter search is not extensively tested in this project!!  #########
        def ray_hp_space(trial):
            from ray import tune

            return {
                "learning_rate": tune.loguniform(1e-6, 1e-4),
                "seed": tune.uniform(1, 50),
            }
        
        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-2, log=True),
                # "weight_decay": trial.suggest_float("weight_decay", 0, 1, log=True),
            }
        
        def compute_objective(metrics: Dict[str, float]) -> float:
            """
            The objective to minimize eval loss.

            Args:
                metrics (:obj:`Dict[str, float]`): The metrics returned by the evaluate method.

            Return:
                :obj:`float`: The objective to minimize or maximize
            """
            metrics = copy.deepcopy(metrics)
            loss = metrics.pop("eval_loss", None)
            return loss 
                    
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[evalLogsCallback]
        )
        backend = 'ray' if model_args.use_ray else 'optuna' if model_args.use_optuna else None
        hp_space = ray_hp_space if model_args.use_ray else optuna_hp_space if model_args.use_optuna else None
        best_trial = trainer.hyperparameter_search(
                                        backend=backend,
                                        hp_space=hp_space,
                                        n_trials=model_args.num_trials,
                                        compute_objective=compute_objective)
    else:
       raise ValueError("This runner is only for hyperparams search trials!")

    def log_and_save_metrics():
        metrics = {}
        metrics["best_trial_details"] = json.dumps(best_trial) # run_id, loss, hyperparams
        metrics["best_trial_perplexity"] = math.exp(best_trial[1])
        max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        metrics["train_blocks_per_sample"] = train_uncut_blocks if data_args.max_train_blocks is None else min(data_args.max_train_blocks, train_uncut_blocks)
        metrics["block_size"] = block_size
        metrics["gpus"] = training_args.n_gpu
        metrics["total_epochs"] = training_args.num_train_epochs
        metrics["per_device_train_batch_size"] = training_args.per_device_train_batch_size
        
        metrics["train_table"] = data_args.train_table
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        metrics["eval_blocks_per_sample"] = eval_uncut_blocks if data_args.max_val_blocks is None else min(data_args.max_val_blocks, eval_uncut_blocks) 
        metrics["per_device_eval_batch_size"] = training_args.per_device_eval_batch_size
        metrics["is_dev"] = True if data_args.dev_table else False
        metrics["eval_table"] = data_args.dev_table if data_args.dev_table else data_args.test_table
        if model_args.instantiate_hart:
            metrics["history"] = model_args.add_history
            metrics["extract_layer"] = model_args.extract_layer if model_args.extract_layer else config.extract_layer
            metrics["layer_ins"] = model_args.layer_ins if model_args.layer_ins else config.layer_ins
            if model_args.add_history:
                metrics["0s_initial_history"] = False if model_args.initial_history else True

        trainer.log_metrics("trial", metrics)
        trainer.save_metrics("trial", metrics, combined=False)

    log_and_save_metrics()

    if training_args.do_predict:
        logger.info("*** Evaluate Test set ***")
        eval_dataset, eval_uncut_blocks = load_dataset(logger, tokenizer, data_args.test_table, block_size, data_args.max_val_blocks, data_args, 'test', data_args.disable_hulm_batching)
        
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity
        metrics["eval_blocks_per_sample"] = eval_uncut_blocks if data_args.max_val_blocks is None else min(data_args.max_val_blocks, eval_uncut_blocks) 
        metrics["per_device_eval_batch_size"] = training_args.per_device_eval_batch_size
        metrics["eval_table"] = data_args.test_table
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
