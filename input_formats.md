# Pre-training and language modeling evaluation (perplexity) input format:

user_id   |            message            | updated_time  |
--------- | ----------------------------- | ------------- |
user_id1  | this is a message from user1  | timestamp1    |
user_id2  | this is a message from user2  | timestamp2    |

# Fine-tuning document-level task input format:

user_id   |            message            | updated_time  | label  |
--------- | ----------------------------- | ------------- | ------ |
user_id1  | this is a message from user1  | timestamp1    | class1 |
user_id2  | this is a message from user2  | timestamp2    | class2 |

# Fine-tuning user-level task input format:

user_id   |            message            | updated_time  |   label  |
--------- | ----------------------------- | ------------- | -------- |
user_id1  | this is a message from user1  | timestamp1    | _class1_ |
user_id1  | this is a message from user2  | timestamp2    | _class1_ |
user_id2  | this is a message from user2  | timestamp2    | class2   |

>For user-level tasks, inputs will replicate the same respective user-label for each record for a user.

>Note1: If updated_time is unavailable, you can use message identifier or any other value you'd like to use to 
temporally order user's messages (keeping the same column names). <br/>
Note2: user_id should be of integer type ([Pytorch tensor requirement](https://discuss.pytorch.org/t/how-to-convert-strings-to-tensors/107507))


## Useful input arguments specific to using HaRT:
HaRT takes the input text sequences with user identifiers and automatically creates blocks of user text sequences from the inputs. 
A block is a temporally ordered sequence of messages (text document) of a user separated by a special token
```
--max_train_blocks <insert_number> : restricts the number of blocks per user to this value when training. By default, None. *HaRT is pre-trained and fine-tuned for document-level tasks with 8 max_train blocks. For user-level tasks, we use 4 max_train blocks.*
--max_val_blocks <insert_num> : restricts the number of blocks per user to this value when evaluating. By default, None.
--block_size <insert_num> : the number of tokens in each block. By default, 1024.
```

Arguments related to initial_history, that should be included (by default included in relevant [example scripts](examples)) for using HaRT with recurrent user-states:
```
--add_history: required to use the recurrent user-state module of HaRT.
--initial_history HaRT/initial_history/initialized_history_tensor.pt : uses this as the initial user-state (U0)
```
> Refer [paper](https://arxiv.org/pdf/2205.05128.pdf) and [website](https://nikita-soni-nlp.netlify.app/) for more details on initial history and recurrent user-states.

Useful arguments related to hidden-states (works by default in the code, no changes required; useful to know for custom usage):
```
--output_block_last_hidden_states : outputs last hidden states for all user-blocks (i.e., for all input tokens for a user).
--output_block_extract_layer_hs :  outputs hidden states from 11th layer (i.e., the default extract layer) for all user-blocks (i.e., for all input tokens for a user).
```


## Argument that uses user states for downstream tasks predictions:
```
--use_history_output : By default, uses the average of the output user-states for all non-padded blocks of inputs for a user.
```

## Argument for document-level downstream tasks that have no historical context:
```
--use_hart_no_hist
```

## Useful arguments for fine-tuning:
```
--save_preds_labels: to save the predictions and labels in text files in the output directory^. *Please note this will save in a sorted order (ordered by user_id and updated_time).*
--freeze_model: to freeze HaRT's parameters and only train the classfication head.
```
> ^ If running evaluation and prediction using --do_eval for dev set and --do_predict for test set together, the predictions and labels for the _test_ set will get saved in a _sorted_ order in the output directory.
