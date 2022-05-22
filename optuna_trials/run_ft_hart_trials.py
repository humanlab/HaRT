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
""" Runnig Optuna trials for fine-tuning HaRT for sequence classification."""

import logging
import copy
import json
import os
import sys
dirname = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dirname,'..'))
from typing import Optional, Dict, Callable, Union

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

from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    is_optuna_available,
    is_ray_tune_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    init_deepspeed,
)

from transformers.trainer_utils import BestRun, get_last_checkpoint, is_main_process, IntervalStrategy, HPSearchBackend

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

class CustomTrainer(Trainer):

    def hyperparameter_search(
        self,
        hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
        run_hp_search: Optional[Callable[["optuna.Trial"], BestRun]] = None,
        compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: str = "minimize",
        backend: Optional[Union["str", HPSearchBackend]] = None,
        hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
        **kwargs,
    ) -> BestRun:
        """
        Launch an hyperparameter search using ``optuna`` or ``Ray Tune``. The optimized quantity is determined by
        :obj:`compute_objective`, which defaults to a function returning the evaluation loss when no metric is
        provided, the sum of all metrics otherwise.

        .. warning::

            To use this method, you need to have provided a ``model_init`` when initializing your
            :class:`~transformers.Trainer`: we need to reinitialize the model at each new run. This is incompatible
            with the ``optimizers`` argument, so you need to subclass :class:`~transformers.Trainer` and override the
            method :meth:`~transformers.Trainer.create_optimizer_and_scheduler` for custom optimizer/scheduler.

        Args:
            hp_space (:obj:`Callable[["optuna.Trial"], Dict[str, float]]`, `optional`):
                A function that defines the hyperparameter search space. Will default to
                :func:`~transformers.trainer_utils.default_hp_space_optuna` or
                :func:`~transformers.trainer_utils.default_hp_space_ray` depending on your backend.
            compute_objective (:obj:`Callable[[Dict[str, float]], float]`, `optional`):
                A function computing the objective to minimize or maximize from the metrics returned by the
                :obj:`evaluate` method. Will default to :func:`~transformers.trainer_utils.default_compute_objective`.
            n_trials (:obj:`int`, `optional`, defaults to 100):
                The number of trial runs to test.
            direction(:obj:`str`, `optional`, defaults to :obj:`"minimize"`):
                Whether to optimize greater or lower objects. Can be :obj:`"minimize"` or :obj:`"maximize"`, you should
                pick :obj:`"minimize"` when optimizing the validation loss, :obj:`"maximize"` when optimizing one or
                several metrics.
            backend(:obj:`str` or :class:`~transformers.training_utils.HPSearchBackend`, `optional`):
                The backend to use for hyperparameter search. Will default to optuna or Ray Tune, depending on which
                one is installed. If both are installed, will default to optuna.
            kwargs:
                Additional keyword arguments passed along to :obj:`optuna.create_study` or :obj:`ray.tune.run`. For
                more information see:

                - the documentation of `optuna.create_study
                  <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html>`__
                - the documentation of `tune.run
                  <https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run>`__

        Returns:
            :class:`transformers.trainer_utils.BestRun`: All the information about the best run.
        """
        if backend is None:
            backend = default_hp_search_backend()
            if backend is None:
                raise RuntimeError(
                    "At least one of optuna or ray should be installed. "
                    "To install optuna run `pip install optuna`."
                    "To install ray run `pip install ray[tune]`."
                )
        backend = HPSearchBackend(backend)
        if backend == HPSearchBackend.OPTUNA and not is_optuna_available():
            raise RuntimeError("You picked the optuna backend, but it is not installed. Use `pip install optuna`.")
        if backend == HPSearchBackend.RAY and not is_ray_tune_available():
            raise RuntimeError(
                "You picked the Ray Tune backend, but it is not installed. Use `pip install 'ray[tune]'`."
            )
        self.hp_search_backend = backend
        if self.model_init is None:
            raise RuntimeError(
                "To use hyperparameter search, you need to pass your model through a model_init function."
            )

        self.hp_space = default_hp_space[backend] if hp_space is None else hp_space
        self.hp_name = hp_name
        self.compute_objective = default_compute_objective if compute_objective is None else compute_objective

        run_hp_search = run_hp_search if run_hp_search is not None else run_hp_search_optuna if backend == HPSearchBackend.OPTUNA else run_hp_search_ray
        best_run = run_hp_search(self, n_trials, direction, **kwargs)

        self.hp_search_backend = None
        return best_run
 
class EvalLogsCallback(TrainerCallback):

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs['metrics']
        if control.should_save:
            metrics['trial_params'] = json.dumps(state.trial_params)
        self.metrics = metrics.copy() 
        print(json.dumps(metrics))      
    
    def on_save(self, args, state, control, **kwargs):
        output_dir = state.best_model_checkpoint.split('/checkpoint')[0]
        self.save_metrics('eval_{}'.format(self.metrics['epoch']), self.metrics, output_dir)
        print("Saving eval metrics after epoch {} into {}".format(self.metrics['epoch'], output_dir))
    
    def on_train_end(self, args, state, control, **kwargs):
        output_dir = state.best_model_checkpoint.split('/checkpoint')[0]
        metrics = state.trial_params.copy()
        metrics["number_of_gpus"] = args.n_gpu
        metrics["best_metric"] = state.best_metric
        metrics["best_model_checkpoint"] = state.best_model_checkpoint
        self.metrics = metrics
        self.save_metrics('final', self.metrics, output_dir)
    
    def save_metrics(self, split, metrics, output_dir, combined=True):
        path = os.path.join(output_dir, f"{split}_results.json")
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
    # model initialization seed maybe different from training seed -- to get a stable averaged result.
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
        hart = HaRTPreTrainedModel.from_pretrained(model_args.model_name_or_path)
    elif training_args.do_train and model_args.load_non_PT_hulm_model:
        hartbaseLMModel = HaRTBaseLMHeadModel.from_pretrained(model_args.model_name_or_path, config=config)
        hartbaseLMModel.resize_token_embeddings(len(tokenizer))
        hart = HaRTPreTrainedModel(config, hartbaseLMModel)
    else:
        raise ValueError("You're neither training nor evaluating. Can't pick a model because I don't know what do you want to do.")


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

    if model_args.search_params:
        def model_init():
            set_seed(model_args.init_seed)
            if training_args.do_train and not model_args.load_non_PT_hulm_model:
                model = HaRTForSequenceClassification(config, model_args.model_name_or_path)
            else:
                model = HaRTForSequenceClassification(config, pt_model=hart)
            set_seed(training_args.seed)

            def freeze_params(model: nn.Module):
                for par in model.parameters():
                    par.requires_grad = False

            if model_args.freeze_model:        
                freeze_params(model.transformer)
            
            if data_args.task_type=='user':
                freeze_params(model.transformer.transformer)
    
            return model
        
        ####### NOTE: Ray Hyperparameter search is not extensively tested in this project!!  #########
        def ray_hp_space(trial):
            from ray import tune

            return {
                "learning_rate": tune.loguniform(5e-6, 5e-4),
            }
        
        def stance_optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
                # "weight_decay": trial.suggest_float("weight_decay", 0.0, 1.0, log=False),
            }
        
        def sent_optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-5, log=True),
                # "weight_decay": trial.suggest_float("weight_decay", 0.0, 1.0, log=False),
            }

        def doc_optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
                # "weight_decay": trial.suggest_float("weight_decay", 0.0, 1.0, log=False),
            }
        
        def compute_objective(metrics: Dict[str, float]) -> float:
            """
            The objective to minimize/maximize.

            Args:
                metrics (:obj:`Dict[str, float]`): The metrics returned by the evaluate method.

            Return:
                :obj:`float`: The objective to minimize or maximize
            """
        
            metrics = copy.deepcopy(metrics)
            f1 = metrics.pop(training_args.metric_for_best_model, None)

            return f1 
        
        def run_hp_search(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
            import optuna

            def _objective(trial, checkpoint_dir=None):
                checkpoint = None
                if checkpoint_dir:
                    for subdir in os.listdir(checkpoint_dir):
                        if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                            checkpoint = os.path.join(checkpoint_dir, subdir)
                trainer.objective = None
                trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
                # If there hasn't been any evaluation during the training loop.
                if getattr(trainer, "objective", None) is None:
                    metrics = trainer.evaluate()
                    trainer.objective = trainer.compute_objective(metrics)
                output_dir = trainer.state.best_model_checkpoint.split('/checkpoint')[0]
                # TODO: see if can get the best model from best model checkpoint instead of saving
                # if yes, use HF trainer instead of CustomTrainer and remove run_hp_search code.
                trainer.save_model(output_dir=output_dir)
                return trainer.objective

            timeout = kwargs.pop("timeout", None)
            n_jobs = kwargs.pop("n_jobs", 1)
            study = optuna.create_study(direction=direction, **kwargs)
            study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
            best_trial = study.best_trial
            return BestRun(str(best_trial.number), best_trial.value, best_trial.params)

        # Initialize our Trainer
        trainer = CustomTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
                ]
        )

        backend = 'ray' if model_args.use_ray else 'optuna' if model_args.use_optuna else None
        optuna_hp_space = stance_optuna_hp_space if data_args.task_name=='stance' else sent_optuna_hp_space if data_args.task_name=='sentiment' else doc_optuna_hp_space
        hp_space = ray_hp_space if model_args.use_ray else optuna_hp_space if model_args.use_optuna else None
        
        best_trial = trainer.hyperparameter_search(
                                        backend=backend,
                                        hp_space=hp_space,
                                        run_hp_search=run_hp_search,
                                        n_trials=model_args.num_trials,
                                        compute_objective=compute_objective,
                                        direction='maximize')
    else:
       raise ValueError("This runner is only for hyperparams search trials!")

    def log_and_save_metrics():
        metrics = {}
        metrics['pretrained_model_loc'] = model_args.model_name_or_path
        metrics["best_trial_details"] = json.dumps(best_trial) # run_id, f1, hyperparams
        metrics["best_trial_f1"] = best_trial.objective
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
        max_val_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        metrics["eval_blocks_per_sample"] = eval_uncut_blocks if data_args.max_val_blocks is None else min(data_args.max_val_blocks, eval_uncut_blocks) 
        metrics["per_device_eval_batch_size"] = training_args.per_device_eval_batch_size
        metrics["is_dev"] = True if data_args.dev_table else False
        metrics["eval_table"] = data_args.dev_table if data_args.dev_table else data_args.test_table
        if config.add_history:
            metrics["history"] = model_args.add_history
            metrics["extract_layer"] = config.extract_layer
            metrics["layer_ins"] = config.layer_ins
            if model_args.add_history:
                metrics["0s_initial_history"] = False if model_args.initial_history else True

        trainer.log_metrics("trial", metrics)
        trainer.save_metrics("trial", metrics, combined=False)

    log_and_save_metrics()

    # Evaluation
    if training_args.do_predict:
        trainer.model = trainer.model.from_pretrained(training_args.output_dir + '/run-' + best_trial.run_id)
        trainer.model = trainer.model.to(training_args.device)
        trainer.save_model()
        
        args = [logger, tokenizer, data_args.test_table, block_size, data_args.max_val_blocks, data_args, 'test', data_args.disable_hulm_batching]
        eval_dataset, eval_uncut_blocks = load_dataset(args)
        logger.info("*** Evaluate all test set ***")
        eval_test(best_trial, 'test', data_args, training_args, eval_dataset, eval_uncut_blocks, trainer)

def eval_test(best_trial, test_type, data_args, training_args, eval_dataset, eval_uncut_blocks, trainer):
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    metrics['best_trial_objective'] = best_trial.objective
    metrics['best_trial_run_id'] = best_trial.run_id
    metrics['best_trial_hyperparams'] = json.dumps(best_trial.hyperparameters)
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
