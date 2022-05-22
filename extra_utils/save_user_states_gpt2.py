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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import sys


import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from args.ft_args import DataTrainingArguments, ModelArguments
from extra_utils.finetune_user_gpt2_save_states import GPT2ForSequenceClassification
from data.utils_hart.ft_doc_data_utils import load_dataset as load_doc_dataset
from data.utils_hart.ft_user_data_utils import load_dataset as load_user_dataset
from data.data_collator import DataCollatorWithPaddingForHaRT

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.6.0.dev0")

logger = logging.getLogger(__name__)
    
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
    # For document level tasks like stance detection, provide num_labels for classification
    # For user level tasks, num_labels is set to 1, making it a regression task
    if data_args.task_type=='document':
        is_regression = False
        num_labels = data_args.num_labels
    else:
        is_regression = True
        num_labels = 1  #regression task

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
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    config.add_history = None
    model = GPT2ForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        args=training_args,
        tokenizer=tokenizer,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        save_user_states=True
    )
  
    def add_insep_token(tokenizer):
        special_tokens_dict = {'sep_token': str('<|insep|>')}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        assert num_added_toks == 1
        assert tokenizer.sep_token == '<|insep|>'
    add_insep_token(tokenizer)
    
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
            return load_doc_dataset(*args)
        elif data_args.task_type=='user':
            return load_user_dataset(*args)

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
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    table = data_args.train_table if data_args.train_table else data_args.dev_table if data_args.dev_table else data_args.test_table 
    data_type = 'train' if data_args.train_table else 'dev' if data_args.dev_table else 'test' 
    args = [logger, tokenizer, table, block_size, data_args.max_val_blocks, data_args, data_type, data_args.disable_hulm_batching]
    eval_dataset, eval_uncut_blocks = load_dataset(args)
    logger.info("*** Evaluate all test set ***")
    metrics = trainer.evaluate(eval_dataset=eval_dataset)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
