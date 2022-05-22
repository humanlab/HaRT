import time
import copy
import pandas as pd
from more_itertools import split_at
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from transformers import BatchEncoding


def add_insep_token(tokenizer):
    special_tokens_dict = {'sep_token': str('<|insep|>')}
    tokenizer.add_special_tokens(special_tokens_dict)
            
def get_conn(data_args):
    myDB = URL(drivername='mysql', host=data_args.hostname,
                database=data_args.db, query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})
    engine = create_engine(myDB, encoding='latin1')
    conn = engine.connect()
    return conn

def get_data(logger, table, data_args, data_type):

    logger.info("Getting data from table:{} in {} database".format(table, data_args.db))
    conn = get_conn(data_args)   
    
    select_clause = 'select user_dataset_id, message_id, message, updated_time from ' + table
    order_clause = ' order by user_dataset_id, updated_time'
    limit_clause =  '' #if not __debug__ else ' limit 10'

    if data_type=='train': 
        if "en_non_oosmsgs" in table:      
            dev_filter_column = 'is_oosusr_dev'
            test_filter_column = 'is_oosusr_test'
            where_clause = ' where ' + dev_filter_column + '=0' + ' and ' + test_filter_column + '=0'
            stmt = select_clause + where_clause + order_clause + limit_clause
        else:
            stmt = select_clause + order_clause + limit_clause       
        results = conn.execute(stmt)
    elif data_type=='dev':
        if 'en_non_oosmsgs' in table:      
            filter_column = 'is_oosusr_dev'
            where_clause = ' where ' + filter_column + '=1'
            stmt = select_clause + where_clause + order_clause + limit_clause
        elif 'en_oosmsgs' in table:      
            filter_column = 'is_oosmsg_dev'
            where_clause = ' where ' + filter_column + '=1'
            stmt = select_clause + where_clause + order_clause + limit_clause
        else:
            stmt = select_clause + order_clause + limit_clause
            
        results = conn.execute(stmt)
    elif data_type=='test':
        if 'en_non_oosmsgs' in table:      
            filter_column = 'is_oosusr_test'
            where_clause = ' where ' + filter_column + '=1'
            stmt = select_clause + where_clause + order_clause + limit_clause
        elif 'en_oosmsgs' in table:      
            filter_column = 'is_oosmsg_test'
            where_clause = ' where ' + filter_column + '=1'
            stmt = select_clause + where_clause + order_clause + limit_clause
        results = conn.execute(stmt)
    
    data = pd.DataFrame(results.fetchall()) 
    data.columns = results.keys()
    data = data[data.message.notnull()]

    conn.close()
    return data

def append_insep(data, tokenizer):
    data['message'] = data['message'] + tokenizer.sep_token

def concat(data):
    return data.groupby('user_dataset_id')['message'].apply(''.join).reset_index()

def process_data(data, tokenizer, block_size, max_blocks):

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

    data['batch_encodings'] = data['message'].apply(process)

def transform_data(logger, tokenizer, data, block_size, max_blocks):
    start_time = time.time()
    data_new = data[['user_dataset_id', 'message']].copy()
    append_insep(data_new, tokenizer)
    data_new = concat(data_new)
    process_data(data_new, tokenizer, block_size, max_blocks)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return data_new 
    
def group_data(data, logger):
    batch = pd.DataFrame(data.batch_encodings.tolist())
    batch = batch.stack()
    logger.info('************** Total Number of instances = {} *************'.format(len(batch)))
    return batch.to_numpy().tolist()

def load_dataset(logger, tokenizer, table, block_size, max_blocks, data_args, data_type):
    add_insep_token(tokenizer)
    data = get_data(logger, table, data_args, data_type)
    data = transform_data(logger, tokenizer, data, block_size, max_blocks)
    logger.info('************** Block size = {} *************'.format(block_size))
    instances = group_data(data, logger)
    return instances, len(instances)

