import time
import logging
import pandas as pd
from transformers import BatchEncoding

from tqdm import tqdm
tqdm.pandas()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def format_data(user_id_column, text_id_column, order_by_column, data):
    if text_id_column is None:
        data['text_id'] = range(len(data))
        text_id_column = 'text_id'
    if user_id_column is not None and order_by_column is not None:
        data = data.sort_values(by=[user_id_column, order_by_column])
    elif order_by_column is not None:
        data = data.sort_values(by=[order_by_column])
    elif user_id_column is None:
        data['user_id'] = range(len(data))
        user_id_column = 'user_id'
    data.reset_index(drop=True, inplace=True)
    return data, user_id_column, text_id_column

def get_data_from_csv(logger, csv_file, user_id_column, text_id_column, order_by_column):
    logger.info("Getting data from pickle file:{}".format(csv_file))
    data = pd.read_csv(csv_file)
    return format_data(user_id_column, text_id_column, order_by_column, data)

def get_data_from_pkl(logger, pkl_file, user_id_column, text_id_column, order_by_column):
    logger.info("Getting data from pickle file:{}".format(pkl_file))
    data = pd.read_pickle(pkl_file)
    return format_data(user_id_column, text_id_column, order_by_column, data)

def get_data_from_dataframe(logger, data, user_id_column, text_id_column, order_by_column):
    logger.info("Getting data from dataframe:{}".format(data))
    return format_data(user_id_column, text_id_column, order_by_column, data)

def append_insep(data, tokenizer, text_column):
    data[text_column] = data[text_column] + tokenizer.sep_token

def concat(data, user_id_column, text_column, text_id_column):
    return data.groupby(user_id_column).agg({text_column: ' '.join, text_id_column: list}).reset_index()
    # return data.groupby(user_id_column)[text_column].apply(' '.join).reset_index()

def process_data(data, tokenizer, text_column, block_size):

    def tokenize(data):
        return tokenizer(data)
    
    def pad(data, pad_value):
        multiplier = (block_size - len(data))%block_size
        data.extend([pad_value]*multiplier)
        return data
    
    def chunks(data):
        i_values = data['input_ids']
        a_values = data['attention_mask']
        l_values = data['labels']
        return [BatchEncoding(dict(input_ids = i_values[x:x+block_size], 
                            attention_mask=a_values[x:x+block_size],
                            labels = l_values[x:x+block_size])) 
                            for x in range(0, len(i_values), block_size)]

    def process(data):
        tokenized = tokenize(data)
        tokenized['labels'] = tokenized['input_ids'].copy()
        tokenized['input_ids'] = pad(tokenized['input_ids'], tokenizer.eos_token_id)
        tokenized['attention_mask'] = pad(tokenized['attention_mask'], 0)
        tokenized['labels'] = pad(tokenized['labels'], -100)
        return chunks(tokenized)

    data['batch_encodings'] = data[text_column].progress_apply(process)

def transform_data(logger, tokenizer, data, block_size, text_column, user_id_column, text_id_column):
    start_time = time.time()
    data_new = data[[user_id_column, text_column, text_id_column]].copy()
    # TODO: Place in correct position
    data_new = data_new.dropna()
    append_insep(data_new, tokenizer, text_column)
    data_new = concat(data_new, user_id_column, text_column, text_id_column)
    process_data(data_new, tokenizer, text_column, block_size)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return data_new 
    
def group_data(logger, data, user_id_column, text_id_column, max_blocks):
    batch = pd.DataFrame(data.batch_encodings.tolist())
    actual_blocks = len(batch.columns)
    logger.info('************** Total Number of blocks = {} *************'.format(len(batch.columns)))
    if max_blocks is not None and len(batch.columns) > max_blocks:
        batch = batch[range(max_blocks)]
        logger.info('************ Trimmed Number of blocks = {} *************'.format(len(batch.columns)))
    assert len(data)==len(batch)
    #TODO: add a toggle for inference/pre-training
    data = batch # pd.concat((data[[user_id_column, text_id_column]], batch), axis=1)
    return data.to_numpy().tolist(), actual_blocks

def load_tokenized_dataset(tokenizer, data, block_size=1024, max_blocks=8, text_column=None, user_id_column=None, text_id_column=None, order_by_column=None, retain_original_order=False):
    if isinstance(data, pd.DataFrame):
        data, user_id_column, text_id_column = get_data_from_dataframe(logger, data, user_id_column, text_id_column, order_by_column)
    elif 'pkl' in data:
        data, user_id_column, text_id_column = get_data_from_pkl(logger, data, user_id_column, text_id_column, order_by_column)
    elif 'csv' in data:
        data, user_id_column, text_id_column = get_data_from_csv(logger, data, user_id_column, text_id_column, order_by_column)
    else:
        raise ValueError("Invalid data file format. Please provide a pandas dataframe, or csv, or pkl file")
    if retain_original_order:
        original_data_order = data.copy()
    data = transform_data(logger, tokenizer, data, block_size, text_column, user_id_column, text_id_column)
    logger.info('************** Block size = {} *************'.format(block_size))
    if retain_original_order:
        return group_data(logger, data, user_id_column, text_id_column, max_blocks), original_data_order
    return group_data(logger, data, user_id_column, text_id_column, max_blocks)