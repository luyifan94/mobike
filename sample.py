from config import *

import os
import pickle
import pandas as pd

# 将用户到过的地点加入成样本
def get_user_loc(train, test):
    result_path = cache_path + 'user_loc%d.hdf'%(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & use_cache:
        result = pd.read_hdf(result_path)
    else:
        user_start = train[['userid', 'geohashed_start_loc']]
        user_start.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
        user_end = train[['userid', 'geohashed_end_loc']]
        user_loc = pd.concat([user_start, user_end]).drop_duplicates()
        result = pd.merge(test[['orderid', 'userid']], user_loc, on='userid', how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# bike到过的地点加入样本
def get_bike_loc(train, test):
    result_path = cache_path + 'bike_loc%d.hdf'%(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & use_cache:
        result = pd.read_hdf(result_path)
    else:
        bike_start = train[['bikeid', 'geohashed_start_loc']]
        bike_start.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
        bike_end = train[['bikeid', 'geohashed_end_loc']]
        bike_loc = pd.concat([bike_start, bike_end]).drop_duplicates()
        result = pd.merge(test[['orderid', 'bikeid']], bike_loc, on='bikeid', how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 筛选起始地点去向最多的3个地点
def get_loc_loc_rank3(train, test):
    result_path = cache_path + 'loc_loc_rank3%d.hdf'%(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & use_cache:
        result = pd.read_hdf(result_path)
    else:
        sloc_eloc = train.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['geohashed_end_loc'].agg({'sloc_eloc_count':'count'})
        sloc_eloc.sort_values(by='sloc_eloc_count', inplace=True)
        sloc_eloc_3 = sloc_eloc.groupby('geohashed_start_loc').tail(3)
        #eloc_sloc_3 = sloc_eloc.groupby('geohashed_end_loc').tail(3)
        #loc_loc_3 = pd.concat([sloc_eloc_3,eloc_sloc_3]).drop_duplicates()
        result = pd.merge(test[['orderid', 'geohashed_start_loc']], sloc_eloc_3, on='geohashed_start_loc', how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 构造样本
def get_sample(train, test):
    result_path = cache_path + 'sample%d.hdf'%(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & use_cache:
        result = pd.read_hdf(result_path)
    else:
        user_loc = get_user_loc(train, test)
        #bike_loc = get_bike_loc(train, test)
        loc_loc_rank3 = get_loc_loc_rank3(train, test)
        result = pd.concat([user_loc, loc_loc_rank3]).drop_duplicates()
        result = pd.merge(test, result, on='orderid', how='left')
        result = result[result['geohashed_start_loc'] != result['geohashed_end_loc']]
        result = result[(result['geohashed_start_loc'].notnull()) & (result['geohashed_end_loc'].notnull())]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

def get_label(data):
    result_path = cache_path + 'label_dict'
    if os.path.exists(result_path) & use_cache:
        label_dict = pickle.load(open(result_path, 'rb'))
    else:
        train = pd.read_csv(train_path)
        label_dict = dict(zip(train['orderid'],train['geohashed_end_loc']))
        pickle.dump(label_dict, open(result_path, 'wb'))
    data['label'] = data['orderid'].map(label_dict)
    data['label'] = (data['geohashed_end_loc'] == data['label']).astype(int)
    return data