import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from data.utils_hart.hulm_data_utils import transform_data, group_data

def get_conn(data_args):
    myDB = URL(drivername='mysql', host=data_args.hostname,
                database=data_args.db, query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})
    engine = create_engine(myDB, encoding='latin1')
    conn = engine.connect()
    return conn

''''
To be run only once! This will save the sampled users' IDs in a csv that will be used for all following optuna trials.
'''
def sample_train_users(logger, table, data_args, filename):
    logger.info("Getting {} sampled train users from table:{} in {} database, to run optuna trials.".format(str(data_args.num_users_for_optuna), table, data_args.db))
    conn = get_conn(data_args)
    
    select_clause = 'select distinct user_dataset_id from ' + table
    order_clause = ' order by rand() limit ' + str(data_args.num_users_for_optuna)

    dev_filter_column = 'is_oosusr_dev'
    test_filter_column = 'is_oosusr_test'
    source_filter_column = 'dataset '
    source_not_included = "'fb'"
    where_clause = ' where ' + source_filter_column + 'not in (' + source_not_included + ') and '  + dev_filter_column + '=0' + ' and ' + test_filter_column + '=0'
    stmt = select_clause + where_clause + order_clause
    results = conn.execute(stmt)

    data = pd.DataFrame(results.fetchall()) 
    data.columns = results.keys()
    
    data.to_csv(filename, index=False)

    conn.close()
    return data

def get_data(logger, table, data_args, data_type, sampled_users):

    logger.info("Getting data from table:{} in {} database".format(table, data_args.db))
    conn = get_conn(data_args)   
    
    select_clause = 'select user_dataset_id, message_id, message, updated_time from ' + table
    order_clause = ' order by user_dataset_id, updated_time'
    limit_clause =  '' if not __debug__ else ' limit 100'
    source_filter_column = 'dataset '
    source_not_included = "'fb'"
    
    if data_type=='train': 
        if "en_non_oosmsgs" in table:      
            dev_filter_column = 'is_oosusr_dev'
            test_filter_column = 'is_oosusr_test'
            users_id_column = 'user_dataset_id'
            where_clause = ' where ' + source_filter_column + 'not in (' + source_not_included + ') and ' + dev_filter_column + '=0' + ' and ' + test_filter_column + '=0' + ' and ' + users_id_column + ' in (' + sampled_users + ')'
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
    data = data[data.message.notnull()]

    conn.close()
    return data

def sample_users_if_train(logger, table, data_args):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../datasets/pt_sampled_users.csv')
    try:    
        sampled_users = pd.read_csv(filename)
        if sampled_users.size != data_args.num_users_for_optuna:
            sampled_users = sample_train_users(logger, table, data_args, filename)
    except FileNotFoundError:
        sampled_users = sample_train_users(logger, table, data_args, filename)
    sampled_users_string = ', '.join(["'{}'".format(e) for e in sampled_users['user_dataset_id'].to_list()])
    return sampled_users_string

def load_dataset(logger, tokenizer, table, block_size, max_blocks, data_args, data_type, disable_hulm_batching):
    sampled_users_string = sample_users_if_train(logger, table, data_args) if data_type=='train' else ''
    data = get_data(logger, table, data_args, data_type, sampled_users_string)
    data = transform_data(logger, tokenizer, data, block_size)
    logger.info('************** Block size = {} *************'.format(block_size))
    if not disable_hulm_batching:
        return group_data(data, max_blocks, logger) 
    else:
        instances, uncut_num_blocks = group_data(data, max_blocks, logger)
        flat_list = [item for sublist in instances for item in sublist if item is not None]
        return flat_list, uncut_num_blocks

