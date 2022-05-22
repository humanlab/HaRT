import time
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from transformers import BatchEncoding

user_id_column = 'user_id'
message_column = 'message'
order_by_fields = [user_id_column, 'updated_time']

def get_conn(data_args):
    myDB = URL(drivername='mysql', host=data_args.hostname,
                database=data_args.db, query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})
    engine = create_engine(myDB, encoding='latin1')
    conn = engine.connect()
    return conn

def get_data_from_db(logger, table, data_args, data_type):

    logger.info("Getting data from table:{} in {} database".format(table, data_args.db))
    conn = get_conn(data_args)   
    
    select_clause = 'select user_dataset_id' + ', ' + message_column + ', message_id, updated_time from ' + table
    order_clause = ' order by ' + ', '.join(order_by_fields)
    limit_clause =  '' if not __debug__ else ' limit 10'
    source_filter_column = 'dataset '
    source_not_included = "'fb'"

    if data_type=='train': 
        if "en_non_oosmsgs" in table:      
            dev_filter_column = 'is_oosusr_dev'
            test_filter_column = 'is_oosusr_test'
            where_clause = ' where ' + source_filter_column + 'not in (' + source_not_included + ') and ' + dev_filter_column + '=0' + ' and ' + test_filter_column + '=0'
            stmt = select_clause + where_clause + order_clause + limit_clause
        else:
            where_clause = ' where ' + source_filter_column + 'not in (' + source_not_included + ')'
            stmt = select_clause + where_clause + order_clause + limit_clause       
        results = conn.execute(stmt)
    elif data_type=='dev':
        if 'en_non_oosmsgs' in table:      
            filter_column = 'is_oosusr_dev'
            where_clause = ' where ' + source_filter_column + 'not in (' + source_not_included + ') and ' + filter_column + '=1'
            stmt = select_clause + where_clause + order_clause + limit_clause
        elif 'en_oosmsgs' in table:      
            filter_column = 'is_oosmsg_dev'
            where_clause = ' where ' + source_filter_column + 'not in (' + source_not_included + ') and ' + filter_column + '=1'
            stmt = select_clause + where_clause + order_clause + limit_clause
        else:
            where_clause = ' where ' + source_filter_column + 'not in (' + source_not_included + ')'
            stmt = select_clause + where_clause + order_clause + limit_clause
        results = conn.execute(stmt)
    elif data_type=='test':
        if 'en_non_oosmsgs' in table:      
            filter_column = 'is_oosusr_test'
            where_clause = ' where ' + source_filter_column + 'not in (' + source_not_included + ') and ' + filter_column + '=1'
            stmt = select_clause + where_clause + order_clause + limit_clause
        elif 'en_oosmsgs' in table:      
            filter_column = 'is_oosmsg_test'
            where_clause = ' where ' + source_filter_column + 'not in (' + source_not_included + ') and ' + filter_column + '=1'
            stmt = select_clause + where_clause + order_clause + limit_clause
        results = conn.execute(stmt)
    
    data = pd.DataFrame(results.fetchall()) 
    data.columns = results.keys()
    data[user_id_column] = data['user_dataset_id']
    data = data[data.message.notnull()]

    conn.close()
    return data

def get_data_from_csv(logger, csv_file, data_type):
    logger.info("Getting data from {} data pickle file:{}".format(data_type, csv_file))
    data = pd.read_csv(csv_file)
    data.sort_values(by=[', '.join(order_by_fields)], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def get_data_from_pkl(logger, pkl_file, data_type):
    logger.info("Getting data from {} data pickle file:{}".format(data_type, pkl_file))
    data = pd.read_pickle(pkl_file)
    data.sort_values(by=[', '.join(order_by_fields)], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def append_insep(data, tokenizer):
    data[message_column] = data[message_column] + tokenizer.sep_token

def concat(data):
    return data.groupby(user_id_column)[message_column].apply(''.join).reset_index()

def process_data(data, tokenizer, block_size):

    def tokenize(data):
        return tokenizer(data)
    
    def pad(data, pad_value):
        multiplier = (block_size - len(data))%block_size
        data.extend([pad_value]*multiplier)
        return data
    
    def convert_to_int(data):
        data['input_ids'] = list(map(int,data['input_ids']))
        data['attention_mask'] = list(map(int,data['attention_mask']))
        data['labels'] = list(map(int,data['labels']))
    
    def chunks(data):
        i_values = data['input_ids']
        a_values = data['attention_mask']
        l_values = data['labels']
        return [BatchEncoding(dict(input_ids = i_values[x:x+block_size], 
                            attention_mask=a_values[x:x+block_size], labels = l_values[x:x+block_size])) 
                            for x in range(0, len(i_values), block_size)]

    def process(data):
        tokenized = tokenize(data)
        tokenized['labels'] = tokenized['input_ids'].copy()
        tokenized['input_ids'] = pad(tokenized['input_ids'], tokenizer.eos_token_id)
        tokenized['attention_mask'] = pad(tokenized['attention_mask'], 0)
        tokenized['labels'] = pad(tokenized['labels'], -100)
        convert_to_int(tokenized)
        return chunks(tokenized)

    data['batch_encodings'] = data[message_column].apply(process)

def transform_data(logger, tokenizer, data, block_size):
    start_time = time.time()
    data_new = data[[user_id_column, message_column]].copy()
    append_insep(data_new, tokenizer)
    data_new = concat(data_new)
    process_data(data_new, tokenizer, block_size)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return data_new 
    
def group_data(data, max_blocks, logger):
    batch = pd.DataFrame(data.batch_encodings.tolist())
    actual_blocks = len(batch.columns)
    logger.info('************** Total Number of blocks = {} *************'.format(len(batch.columns)))
    if max_blocks is not None and len(batch.columns) > max_blocks:
        batch = batch[range(max_blocks)]
        logger.info('************ Trimmed Number of blocks = {} *************'.format(len(batch.columns)))
    return batch.to_numpy().tolist(), actual_blocks

def load_dataset(logger, tokenizer, table, block_size, max_blocks, data_args, data_type, disable_hulm_batching):
    if 'pkl' in table:
        data = get_data_from_pkl(logger, table, data_type)
    elif 'csv' in table:
        data = get_data_from_csv(logger, table, data_type)
    else:
        data = get_data_from_db(logger, table, data_args, data_type)
    data = transform_data(logger, tokenizer, data, block_size)
    logger.info('************** Block size = {} *************'.format(block_size))
    if not disable_hulm_batching:
        return group_data(data, max_blocks, logger) 
    else:
        instances, uncut_num_blocks = group_data(data, max_blocks, logger)
        flat_list = [item for sublist in instances for item in sublist if item is not None]
        return flat_list, uncut_num_blocks