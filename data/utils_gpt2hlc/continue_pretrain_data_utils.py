import time
import copy
import pandas as pd
from more_itertools import split_at
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from transformers import BatchEncoding

from tqdm import tqdm
tqdm.pandas()

def add_insep_token(tokenizer):
    special_tokens_dict = {'sep_token': str('<|insep|>')}
    tokenizer.add_special_tokens(special_tokens_dict)

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

def process_data(data, tokenizer, block_size, max_blocks,  text_column):

    def tokenize(data):
        return tokenizer(data)
    
    def pad(data, pad_value):
        multiplier = (block_size - len(data))%block_size
        data.extend([pad_value]*multiplier)
        return data
    
    def pad_message(data, pad_value, counter=0):
        for i,x in enumerate(data):
            if len(x) > block_size:
                x = x[0:block_size]
                data[i] = x
                counter+=1
            else:
                x.extend([pad_value]*(block_size-len(x)))
        return data, counter
    
    def limit_and_split_messages(data):
        i_values = data['input_ids'][0:1024*max_blocks] if max_blocks is not None else data['input_ids']
        i_values = list(split_at(i_values, lambda x:x==tokenizer.eos_token_id))[0]
        i_values = i_values[:-1] if i_values[-1]==tokenizer.sep_token_id else i_values
        i_values = list(split_at(i_values, lambda x:x==tokenizer.sep_token_id))
        return i_values
 
    def pad_and_collate_data(data, counter):
        i_values = data
        a_values = [[1]*len(x) for x in i_values]
        l_values = copy.deepcopy(i_values)
        
        i_values, counter = pad_message(i_values, tokenizer.eos_token_id, counter)
        a_values, _ = pad_message(a_values, 0)
        l_values, _ = pad_message(l_values, -100)

        return [BatchEncoding(dict(input_ids = i_values[x], 
                            attention_mask=a_values[x], labels = l_values[x])) 
                            for x in range(len(i_values))], counter

    def process(data):
        counter = 0
        tokenized = tokenize(data)
        tokenized['input_ids'] = pad(tokenized['input_ids'], tokenizer.eos_token_id)
        input_ids = limit_and_split_messages(tokenized)
        ret_data, counter = pad_and_collate_data(input_ids, counter)
        return ret_data

    data['batch_encodings'] = data[text_column].progress_apply(process)

def transform_data(logger, tokenizer, data, block_size, max_blocks, text_column, user_id_column, text_id_column):
    start_time = time.time()
    data_new = data[[user_id_column, text_column, text_id_column]].copy()
    data_new = data_new.dropna()
    append_insep(data_new, tokenizer, text_column)
    data_new = concat(data_new, user_id_column, text_column, text_id_column)
    print(block_size, max_blocks, text_column)
    process_data(data_new, tokenizer, block_size, max_blocks, text_column)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return data_new 
    
def group_data(data, logger):
    batch = pd.DataFrame(data.batch_encodings.tolist())
    batch = batch.stack()
    logger.info('************** Total Number of instances = {} *************'.format(len(batch)))
    return batch.to_numpy().tolist()

def load_dataset(logger, tokenizer, data, block_size=1024, max_blocks=8, text_column=None, user_id_column=None, text_id_column=None, order_by_column=None):
    add_insep_token(tokenizer)
    if isinstance(data, pd.DataFrame):
        data, user_id_column, text_id_column = get_data_from_dataframe(logger, data, user_id_column, text_id_column, order_by_column)
    elif 'pkl' in data:
        data, user_id_column, text_id_column = get_data_from_pkl(logger, data, user_id_column, text_id_column, order_by_column)
    elif 'csv' in data:
        data, user_id_column, text_id_column = get_data_from_csv(logger, data, user_id_column, text_id_column, order_by_column)
    else:
        raise ValueError("Invalid data file format. Please provide a pandas dataframe, or csv, or pkl file")
    data = transform_data(logger, tokenizer, data, block_size, max_blocks, text_column, user_id_column, text_id_column)
    logger.info('************** Block size = {} *************'.format(block_size))
    instances = group_data(data, logger)
    return instances, len(instances)

