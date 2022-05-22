import time
import math
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from transformers import BatchEncoding

user_id_column = 'user_id'
message_column = 'message'
label_column = 'label'

def get_fields(data_args):
    if data_args.task_name is not None:
        if data_args.task_name=='stance':
            return {
                'select_fields': [user_id_column, 'message_id', message_column, 'stance', 'timestamp'],
                'transform_data_fields': [user_id_column, message_column, 'stance'],
                'label_field': 'stance'
            }
        elif data_args.task_name=='sentiment':
            return {
                'select_fields': [user_id_column, 'message_id', message_column, label_column],
                'transform_data_fields': [user_id_column, message_column, label_column],
                'label_field': label_column
            }
    else:
        return {
            'select_fields': [user_id_column, message_column, label_column],
            'transform_data_fields': [user_id_column, message_column, label_column],
            'label_field': label_column
        }

def get_conn(data_args):
    myDB = URL(drivername='mysql', host=data_args.hostname,
                database=data_args.db, query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})
    engine = create_engine(myDB, encoding='latin1')
    conn = engine.connect()
    return conn

def get_data_from_db(logger, table, fields, data_args, data_type):

    logger.info("Getting data from table:{} in {} database".format(table, data_args.db))
    conn = get_conn(data_args)   
    
    select_clause = 'select ' + ', '.join(fields['select_fields']) + ' from ' + table
    where_clause = ' where user_id not in (select user_id from sentiment_dev_users)' \
                        if data_args.task_name=='sentiment' and data_type=='train' \
                    else \
                    ' where user_id in (select user_id from sentiment_dev_users)' \
                        if data_args.task_name=='sentiment' and data_type=='dev' \
                    else ''
    limit_clause =  '' #if not __debug__ else ' limit 10'

    stmt = select_clause + where_clause + limit_clause
    results = conn.execute(stmt)

    data = pd.DataFrame(results.fetchall()) 
    data.columns = results.keys()
    data = data[data.message.notnull()]

    conn.close()
    return data

def get_data_from_csv(logger, csv_file, fields, data_type):
    logger.info("Getting data from {} data pickle file:{}".format(data_type, csv_file))
    data = pd.read_csv(csv_file)
    data.sort_values(by=fields['order_by_fields'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def get_data_from_pkl(logger, pkl_file, fields, data_type):
    logger.info("Getting data from {} data pickle file:{}".format(data_type, pkl_file))
    data = pd.read_pickle(pkl_file)
    data.sort_values(by=fields['order_by_fields'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def map_labels_to_int_value(data, label_field):
    labels_map = {
        'positive': 1,
        'negative': -1,
        'neutral': 0,
        'objective-OR-neutral': 0,
        'objective': 0,
        None: float('NaN'),
    }
    data[label_field] = data[label_field].map(lambda x : labels_map[x])

def tokenize_with_labels(data, label_field, tokenizer, data_args):
    def tokenize(data):
        return tokenizer(data)

    def process(data):
        # get the input_ids (i.e., token_ids) and the attention_mask
        # attention_mask is not altered since it's required to attend to all tokens.
        be = tokenize(data['message'])
        
        if (data_args.task_name == 'stance' or data_args.task_name == 'sentiment') and not math.isnan(data[label_field]):
            be['labels'] = int(data[label_field]) + 1
        elif not math.isnan(data[label_field]):
            be['labels'] = int(data[label_field])
        return be

    data['tokenized'] = data.apply(process, axis=1)

def normalize(data):
    normalized = pd.json_normalize(data['tokenized'])
    data = pd.concat([data, normalized], axis=1)
    return data

def pad_and_chunk(data, tokenizer, block_size):

    def pad_or_truncate(data, pad_value):
        if len(data) > block_size:
            data = data[0:block_size]
        else:
            multiplier = (block_size - len(data))%block_size
            data.extend([pad_value]*multiplier)
        return data
     
    def chunks(data):
        i_values = data['input_ids']
        a_values = data['attention_mask']
        l_values = data['labels']
        return [BatchEncoding(dict(input_ids = i_values, 
                            attention_mask=a_values, 
                            labels = l_values)) 
                            ]

    def process(data):
        data['input_ids'] = pad_or_truncate(data['input_ids'], tokenizer.eos_token_id)
        data['attention_mask'] = pad_or_truncate(data['attention_mask'], 0)
        return chunks(data)

    data['batch_encodings'] = data.apply(process, axis=1)

def transform_data(logger, tokenizer, data, fields, block_size):
    start_time = time.time()
    data_new = data[fields['transform_data_fields']].copy()
    tokenize_with_labels(data_new, fields['label_field'], tokenizer)
    data_new = normalize(data_new)
    pad_and_chunk(data_new, tokenizer, block_size)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return data_new 
    
def group_data(data, max_blocks, logger):
    batch = pd.DataFrame(data.batch_encodings.tolist())
    actual_blocks = len(batch.columns)
    logger.info('************** Total Number of blocks = {} *************'.format(len(batch.columns)))
    logger.info('************** Total Number of instances = {} *************'.format(batch.shape[0]))
    if max_blocks is not None and len(batch.columns) > max_blocks:
        batch = batch[range(max_blocks)]
        logger.info('************ Trimmed Number of blocks = {} *************'.format(len(batch.columns)))
    return batch.to_numpy().tolist(), actual_blocks

def load_dataset(logger, tokenizer, table, block_size, max_blocks, data_args, data_type, disable_hulm_batching):
    fields = get_fields(data_args)
    if 'pkl' in table:
        data = get_data_from_pkl(logger, table, fields, data_type)
    elif 'csv' in table:
        data = get_data_from_csv(logger, table, fields, data_type)
    else:
        data = get_data_from_db(logger, table, fields, data_args, data_type)
    if data_args.task_name=='sentiment':
        map_labels_to_int_value(data, fields['label_field'])
    data = transform_data(logger, tokenizer, data, fields, block_size, data_args)
    logger.info('************** Block size = {} *************'.format(block_size))

    if not disable_hulm_batching:
        return group_data(data, max_blocks, logger) 
    else:
        instances, uncut_num_blocks = group_data(data, max_blocks, logger)
        logger.info('************** Total number of instances = {} *************'.format(len(instances)))
        flat_list = [item for sublist in instances for item in sublist if item is not None]
        return flat_list, uncut_num_blocks

