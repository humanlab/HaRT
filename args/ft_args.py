from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "The type of task to train on: 'document' or 'user' -level"},
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: 'stance', 'sentiment', 'age', 'ope', or 'ner'"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False, metadata={"help": "NER return entity level metrics or not"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    
    )
    use_history_output: bool = field(
        default=False, metadata={"help": "Should use the history output from Ar_HuLM for FT tasks predictions (regression/user-level tasks mainly) or not."}
    )
    save_preds_labels: bool = field(
        default=False, metadata={"help": "Should save the predictions and labels into text files or not."}
    )
    num_labels: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of classification labels when fine tuning a 'document' type task."
        },
    )
    train_table: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data table in a csv or pickle file (path to the file)."})
    dev_table: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data table in a csv or pickle file (path to the file) to validate the model during training."},
    )
    test_table: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data table in a csv or pickle file (path to the file) to evaluate the trained model for perplexity."},
    )
    db: Optional[str] = field(
        default=None,
        metadata={"help": "The database where input training data table resides. (a mysql database)."}
    )
    hostname: Optional[str] = field(
        default=None,
        metadata={"help": "The host name or IP where the (mysql) database resides."}
    )
    max_train_blocks: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training blocks to this "
            "value if set."
        },
    )
    max_val_blocks: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation blocks to this "
            "value if set."
        },
    )
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Optional input block sequence length after tokenization "
            "(batched into instances of max_train_blocks/max_val_blocks , each of size block_size"
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    disable_hulm_batching: bool = field(
        default=False, metadata={"help": "Batch the dataset as a flat list ([users, blocks * block_size]) instead of hulm style batching, i.e., [users, blocks, block_size] dimensions."}
    )
    agg_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "One of 'last', 'sum', 'avg', 'masked_last', 'masked_avg', 'masked_sum'"
            "When using user_states/history for downstream tasks, what kind of "
            "user_states/history aggregation to use. Currently, used only when saving states for users."
        }
    )
    train_pkl: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data pickle file."})
    train_hist_pkl: Optional[str] = field(
        default=None,
        metadata={"help": "The input training users' historical data pickle file."})
    dev_pkl: Optional[str] = field(
        default=None,
        metadata={"help": "The input dev data pickle file."})
    dev_hist_pkl: Optional[str] = field(
        default=None,
        metadata={"help": "The input dev users' historical data pickle file."})
    test_pkl: Optional[str] = field(
        default=None,
        metadata={"help": "The input test data pickle file."})
    test_hist_pkl: Optional[str] = field(
        default=None,
        metadata={"help": "The input test users' historical data pickle file."})


    def __post_init__(self):
        if self.task_type is None or (self.task_type != 'user' and self.task_type != 'document'):
            raise ValueError("Need to define task type as one of 'document' or 'user'")
        if self.num_labels is None:
            raise ValueError('num_labels required to fine-tune downstream tasks!')        
        if self.train_table is None and (self.dev_table is None and self.test_table is None):
            raise ValueError("Need a training/validation (dev or test) table.")

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    init_seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of model initialization."})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_qh05_wts: bool = field(
        default=False, 
        metadata={
            "help": "Whether to use (at 'layer_ins') pretrained query, key, value weights followed by" 
            "query weights (for concatenated query and history) initialized with 0.5 mean, instead of,"
            "newly initialized query (for concatenated hidden states and history) and key weights"
            }
    )
    use_hart_no_hist: bool = field(
        default=False,
        metadata={"help": "Whether to use HaRT model with no available historcal context."},
    )
    freeze_model: bool = field(
        default=False, metadata={"help": "Freeze the transformer module of the model. Train only classification layer."}
    )
    load_non_PT_hulm_model: bool = field(
        default=False, metadata={"help": "Whether to use a non-pretrained hulm model or not"}
    )
    add_history: bool = field(
        default=False, metadata={"help": "Whether to use history (and history recurrence) or not."}
    )
    initial_history: Optional[str] = field(
        default=None, metadata={"help": "A .pt file containing a reasonable initial history embedding as a pytorch tensor."}
    )
    #TODO: following args should ideally be a part of training_args
    metric_for_early_stopping: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={
            "help": "To be used with `metric_for_early_stopping`."
             "To stop training when the specified `metric_for_early_stopping` worsens for"
            "`early_stopping_patience` evaluation calls."
        }
    )
    early_stopping_threshold: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Use with `metric_for_early_stopping` and `early_stopping_patience` to denote how"
            "much the specified metric must improve to satisfy early stopping conditions."
            }
    )
    search_params: bool = field(
        default=False, metadata={"help": "To enable Hyperparameters search using ``optuna`` or ``Ray Tune``"}
    )
    use_ray: bool = field(
        default=False, metadata={"help": "To enable Hyperparameters search using ``Ray Tune``"}
    )
    use_optuna: bool = field(
        default=False, metadata={"help": "To enable Hyperparameters search using ``optuna``"}
    )
    num_trials: Optional[int] = field(
        default=10,
        metadata={
            "help": "Number of trials to run when 'search_params' is true."
        },
    )
