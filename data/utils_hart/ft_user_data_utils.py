import time
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from transformers import BatchEncoding

user_id_column = 'user_id'
message_column = 'message'
order_by_fields = [user_id_column, 'updated_time']
label_column = 'label'

def get_fields(data_args):
    if data_args.task_name is not None:
        return {
                'order_by_fields': [user_id_column, 'message_id'],
                'label_field': data_args.task_name
        }
    else:
        return {
            'order_by_fields': order_by_fields,
            'label_field': label_column
        }

def get_conn(data_args):
    myDB = URL(drivername='mysql', host=data_args.hostname,
                database=data_args.db, query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})
    engine = create_engine(myDB, encoding='latin1')
    conn = engine.connect()
    return conn

def get_data_from_db(logger, table, label_field, data_args, data_type):

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
    labels.columns = [user_id_column, label_column]
    
    return labels

def get_data_from_csv(logger, csv_file, fields, data_type):
    logger.info("Getting data from {} data pickle file:{}".format(data_type, csv_file))
    data = pd.read_csv(csv_file)
    data.sort_values(by=fields['order_by_fields'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data_new = data[[user_id_column, message_column]].copy()
    labels = data[[user_id_column, label_column]].copy()
    labels.drop_duplicates(inplace=True)
    return data_new, labels

def get_data_from_pkl(logger, pkl_file, fields, data_type):
    logger.info("Getting data from {} data pickle file:{}".format(data_type, csv_file))
    data = pd.read_pickle(pkl_file)
    data.sort_values(by=fields['order_by_fields'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data_new = data[[user_id_column, message_column]].copy()
    labels = data[[user_id_column, label_column]].copy()
    labels.drop_duplicates(inplace=True)
    return data_new, labels

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
    
    #TODO: check if this is even required 
    def convert_to_int(data):
        data['input_ids'] = list(map(int,data['input_ids']))
        data['attention_mask'] = list(map(int,data['attention_mask']))
    
    def chunks(data):
        i_values = data['input_ids']
        a_values = data['attention_mask']
        return [BatchEncoding(dict(input_ids = i_values[x:x+block_size], 
                            attention_mask=a_values[x:x+block_size])) 
                            for x in range(0, len(i_values), block_size)]

    def process(data):
        tokenized = tokenize(data)
        tokenized['input_ids'] = pad(tokenized['input_ids'], tokenizer.eos_token_id)
        tokenized['attention_mask'] = pad(tokenized['attention_mask'], 0)
        convert_to_int(tokenized)
        return chunks(tokenized)

    data['batch_encodings'] = data['message'].apply(process)

def transform_data(logger, tokenizer, data, block_size):
    start_time = time.time()
    data_new = data[[user_id_column, message_column]].copy()
    append_insep(data_new, tokenizer)
    data_new = concat(data_new)
    process_data(data_new, tokenizer, block_size)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return data_new 
    
def join_data_and_labels(data, labels):
    assert len(data)==len(labels)
    merged_data = pd.merge(data, labels, on=user_id_column)
    assert len(merged_data)==len(data)
    assert merged_data.shape[-1]==4
    return merged_data

def group_data(data, max_blocks, logger):
    batch = pd.DataFrame(data.batch_encodings.tolist())
    actual_blocks = len(batch.columns)
    logger.info('************** Total Number of blocks = {} *************'.format(len(batch.columns)))
    if max_blocks is not None and len(batch.columns) > max_blocks:
        batch = batch[range(max_blocks)]
        logger.info('************ Trimmed Number of blocks = {} *************'.format(len(batch.columns)))
    assert len(data)==len(batch)
    data = pd.concat((data[[user_id_column, label_column]], batch), axis=1)
    assert data.shape[-1]==batch.shape[-1] + 2
    return data.to_numpy().tolist(), actual_blocks

def load_dataset(logger, tokenizer, table, block_size, max_blocks, data_args, data_type, disable_hulm_batching):
    label_field = data_args.task_name
    data_type = 'test_qlength' if data_args.task_name == 'ope' else data_type
    fields = get_fields(data_args)
    if 'pkl' in table:
        data, labels = get_data_from_pkl(logger, table, fields, data_type)
    elif 'csv' in table:
        data, labels = get_data_from_csv(logger, table, fields, data_type)
    else:
        data, labels = get_data_from_db(logger, table, label_field, data_args, data_type)
    data = transform_data(logger, tokenizer, data, block_size)
    data = join_data_and_labels(data, labels)
    logger.info('************** Block size = {} *************'.format(block_size))
    if not disable_hulm_batching:
        return group_data(data, max_blocks, logger) 
    else:
        instances, uncut_num_blocks = group_data(data, max_blocks, logger)
        flat_list = [item for sublist in instances for item in sublist if item is not None]
        return flat_list, uncut_num_blocks


