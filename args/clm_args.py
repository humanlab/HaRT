from dataclasses import dataclass, field
from typing import Optional

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
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
    use_qh05_wts: bool = field(
        default=False, 
        metadata={
            "help": "Whether to use (at 'layer_ins') pretrained query, key, value weights followed by" 
            "query weights (for concatenated query and history) initialized with 0.5 mean, instead of,"
            "newly initialized query (for concatenated hidden states and history) and key weights"}
    )
    instantiate_hart: bool = field(
        default=False, metadata={"help": "Whether to use a local instance of model config or not."}
    )
    add_history: bool = field(
        default=False, metadata={"help": "Whether to use history (and history recurrence) or not."}
    )
    initial_history: Optional[str] = field(default=None, metadata={"help": "A .pt file containing a reasonable initial history embedding as a pytorch tensor."}) 
    layer_ins: Optional[int] = field(
        default=None,
        metadata={
            "help": "If add_history is True, layer_ins tells at which layer the history should be addded (inserted)."
        },
    )
    extract_layer: Optional[int] = field(
        default=11,
        metadata={
            "help": "If add_history is True, extract_layer tells which layer's output should be used for updating history."
        },
    )
    output_block_last_hidden_states: bool = field(
        default=False, metadata={"help": "Whether to output last hidden-states of the model's blocks at the output of last layer for each block or not."}
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
        default=5,
        metadata={
            "help": "Number of trials to run when 'search_params' is true."
        },
    )
    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    train_table: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data table (a mysql database table)."})
    dev_table: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data table to evaluate the perplexity on. (a mysql database table)."},
    )
    test_table: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data table to evaluate the perplexity on. (a mysql database table)."},
    )
    db: Optional[str] = field(
        default=None,
        metadata={"help": "The database where input training data table resides. (a mysql database)."})
    hostname: Optional[str] = field(
        default=None,
        metadata={"help": "The host name or IP where the (mysql) database resides."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    num_users_for_optuna: Optional[int] = field(
        default=5000,
        metadata={
            "help": "For hyperparameter search, truncate the number of training users to this "
            "value if set."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
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
        default=None,
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.train_table is None and self.dev_table is None and self.test_table is None and self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a train/validation(dev or test) database table, dataset name or a training/validation file.")
        else:
            if (self.train_table is not None or self.dev_table is not None or self.test_table is not None) and (self.db is None or self.hostname is None):
                raise ValueError("Need database and hostname/IP if providing a train/val(dev or test) mysql tables.")
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
