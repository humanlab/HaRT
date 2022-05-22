import time
import copy
import pandas as pd
from more_itertools import split_at
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from transformers import BatchEncoding


def add_insep_token(tokenizer):
    special_tokens_dict = {'sep_token': str('<|insep|>')}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            

def get_conn(data_args):
    myDB = URL(drivername='mysql', host=data_args.hostname,
                database=data_args.db, query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})
    engine = create_engine(myDB, encoding='latin1')
    conn = engine.connect()
    return conn

def get_data(logger, table, label_field, data_args, data_type):

    logger.info("Getting data from table:{} in {} database".format(table, data_args.db))
    conn = get_conn(data_args)   
    
    select_clause = 'select user_id, message_id, message, updated_time from ' + table
    where_clause = ''
    order_clause = ' order by user_id, updated_time'
    limit_clause =  '' #if not __debug__ else ' limit 10'

    if data_type=='train': 
        if "en_non_oosmsgs" in table:      
            filter_table = 'masterstats_lbp_upt50_en_train'
            where_clause = ' where user_id in (select user_id from ' + filter_table + ')'
            stmt = select_clause + where_clause + order_clause + limit_clause
        else:
            filter_table = '20_outcomes_all'
            filter_column = 'r10pct_test_fold'
            where_clause = ' where user_id in (select user_id from ' + filter_table + ' where '+ filter_column +'=0)'
            stmt = select_clause + where_clause + order_clause + limit_clause
            
    elif data_type=='dev':
        if 'en_non_oosmsgs' in table:      
            filter_column = 'is_oosuser_dev'
            where_clause = ' where ' + filter_column + '=1'
        elif 'en_oosmsgs' in table:      
            filter_column = 'is_oosmsg_dev'
            where_clause = ' where ' + filter_column + '=1'
        else:
            filter_table = '20_outcomes_all'
            filter_column = 'r10pct_test_fold'
            where_clause = ' where user_id in (select user_id from ' + filter_table + ' where '+ filter_column +'=1)'
    elif data_type=='test':
        if 'en_non_oosmsgs' in table:      
            filter_column = 'is_oosuser_test'
            where_clause = ' where ' + filter_column + '=1'
        elif 'en_oosmsgs' in table:      
            filter_column = 'is_oosmsg_test'
            where_clause = ' where ' + filter_column + '=1'
    elif data_type=='test_qlength':
        if 'en_non_oosmsgs' in table:      
            filter_table = 'masterstats_lbp_upt50_en_test'
            where_clause = ' where user_id in (select user_id from ' + filter_table + ' where qlength >= 100)'  
        
    stmt = select_clause + where_clause + order_clause + limit_clause    
    results = conn.execute(stmt)
    
    data = pd.DataFrame(results.fetchall()) 
    data.columns = results.keys()
    data = data[data.message.notnull()]

    logger.info("Getting labels for table:{} in {} database".format(table, data_args.db))
    labels = get_labels(conn, data_type, table, label_field)

    conn.close()
    return data, labels

def get_labels(conn, data_type, data_table, label_field):

    select_clause = 'select user_id, ' + label_field + ' from '
    where_clause = ''
    order_clause = ' order by user_id'
    limit_clause =  '' #if not __debug__ else ' limit 10'

    if data_type=='train':
        if 'en_non_oosmsgs' in data_table: 
            table = 'masterstats_lbp_upt50_en_train'
            where_clause = ' where user_id in (select distinct user_id from ' + data_table + ')'
        else:
            table = 'masterstats_lbp_trainingset'
            filter_table_where = '20_outcomes_all where r10pct_test_fold=0' 
            where_clause = ' where user_id in (select distinct user_id from ' + filter_table_where + ')'
    elif data_type=='dev':
        if 'en_non_oosmsgs' in data_table:      
            table = 'masterstats_lbp_upt50_en_dev'
        elif 'en_oosmsgs' in data_table:      
            table = 'masterstats_lbp_upt50_en_dev_seen'  
        else:
            table = 'masterstats_lbp_trainingset'
            filter_table_where = '20_outcomes_all where r10pct_test_fold=1' 
            where_clause = ' where user_id in (select distinct user_id from ' + filter_table_where + ')'
    elif data_type=='test':
        if 'en_non_oosmsgs' in data_table:      
            table = 'masterstats_lbp_upt50_en_test'
        elif 'en_oosmsgs' in data_table:      
            table = 'masterstats_lbp_upt50_en_test_seen'
        else:
            table = 'masterstats_lbp_testset'
    elif data_type=='test_qlength':
        table = 'masterstats_lbp_testset_qlen100'

    stmt = select_clause + table + where_clause +order_clause + limit_clause
    results = conn.execute(stmt)

    labels = pd.DataFrame(results.fetchall()) 
    labels.columns = ['user_id', 'label']
    
    return labels

def append_insep(data, tokenizer):
    data['message'] = data['message'] + tokenizer.sep_token

def concat(data):
    return data.groupby('user_id')['message'].apply(''.join).reset_index()

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
        
        i_values, counter = pad_message(i_values, tokenizer.eos_token_id, counter)
        a_values, _ = pad_message(a_values, 0)

        return [BatchEncoding(dict(input_ids = i_values[x], 
                            attention_mask=a_values[x])) 
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
    data_new = data[['user_id', 'message']].copy()
    append_insep(data_new, tokenizer)
    data_new = concat(data_new)
    process_data(data_new, tokenizer, block_size, max_blocks)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return data_new 
    
def join_data_and_labels(data, labels):
    assert len(data)==len(labels)
    merged_data = pd.merge(data, labels, on='user_id')
    assert len(merged_data)==len(data)
    assert merged_data.shape[-1]==4
    return merged_data
    
def group_data(data, max_blocks, logger):
    batch = data.explode('batch_encodings').reset_index(drop=True)
    batch = batch[['user_id', 'label', 'batch_encodings']]
    logger.info('************** Total Number of instances = {} *************'.format(len(batch)))
    return batch.to_numpy().tolist()

def load_dataset(logger, tokenizer, table, block_size, max_blocks, data_args, data_type):
    add_insep_token(tokenizer)
    label_field = data_args.task_name
    data_type = 'test_qlength' if data_args.task_name == 'ope' else data_type
    data, labels = get_data(logger, table, label_field, data_args, data_type)
    data = transform_data(logger, tokenizer, data, block_size, max_blocks)
    data = join_data_and_labels(data, labels)
    logger.info('************** Block size = {} *************'.format(block_size))
    instances = group_data(data, max_blocks, logger)
    return instances, len(instances)

