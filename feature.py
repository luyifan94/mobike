import geohash

import numpy as np
import pandas as pd

def cal_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(float(lon1) - float(lon2))  # 经度差
    dy = np.abs(float(lat1) - float(lat2))  # 维度差
    b = (float(lat1) + float(lat2)) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L
# 计算两点之间的欧氏距离
def get_distance(result):
    locs = list(set(result['geohashed_start_loc']) | set(result['geohashed_end_loc']))
    if np.nan in locs:
        locs.remove(np.nan)
    deloc = []
    for loc in locs:
        deloc.append(geohash.decode(loc))
    loc_dict = dict(zip(locs,deloc))
    geohashed_loc = result[['geohashed_start_loc','geohashed_end_loc']].values
    distance = []
    for i in geohashed_loc:
        lat1, lon1 = loc_dict[i[0]]
        lat2, lon2 = loc_dict[i[1]]
        distance.append(cal_distance(lat1,lon1,lat2,lon2))
    result.loc[:,'distance'] = distance
    return result


# 获取用户历史行为次数
def get_user_count(train,result):
    user_count = train.groupby('userid',as_index=False)['geohashed_end_loc'].agg({'user_count':'count'})
    result = pd.merge(result,user_count,on=['userid'],how='left')
    return result

# 获取用户去过某个地点历史行为次数
def get_user_eloc_count(train, result):
    user_eloc_count = train.groupby(['userid','geohashed_end_loc'],as_index=False)['userid'].agg({'user_eloc_count':'count'})
    result = pd.merge(result,user_eloc_count,on=['userid','geohashed_end_loc'],how='left')
    return result

# 获取用户从某个地点出发的行为次数
def get_user_sloc_count(train,result):
    user_sloc_count = train.groupby(['userid','geohashed_start_loc'],as_index=False)['userid'].agg({'user_sloc_count':'count'})
    user_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
    result = pd.merge(result, user_sloc_count, on=['userid', 'geohashed_end_loc'], how='left')
    return result

# 获取用户从这个路径走过几次
def get_user_sloc_eloc_count(train,result):
    user_count = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'user_sloc_eloc_count':'count'})
    result = pd.merge(result,user_count,on=['userid','geohashed_start_loc','geohashed_end_loc'],how='left')
    return result

# 获取用户从这个路径折返过几次
def get_user_eloc_sloc_count(train,result):
    user_eloc_sloc_count = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'user_eloc_sloc_count':'count'})
    user_eloc_sloc_count.rename(columns = {'geohashed_start_loc':'geohashed_end_loc','geohashed_end_loc':'geohashed_start_loc'},inplace=True)
    result = pd.merge(result,user_eloc_sloc_count,on=['userid','geohashed_start_loc','geohashed_end_loc'],how='left')
    return result

# 获取目标地点的热度(目的地)
def get_eloc_count(train,result):
    eloc_count = train.groupby('geohashed_end_loc', as_index=False)['userid'].agg({'eloc_count': 'count'})
    result = pd.merge(result, eloc_count, on='geohashed_end_loc', how='left')
    return result

# 获取目标地点的热度(出发地地)
def get_eloc_as_sloc_count(train,result):
    eloc_as_sloc_count = train.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'eloc_as_sloc_count': 'count'})
    eloc_as_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_start_loc'})
    result = pd.merge(result, eloc_as_sloc_count, on='geohashed_start_loc', how='left')
    return result

def get_feature(train, result):
    result = get_distance(result)  # 获取起始点和最终地点的欧式距离

    result = get_user_count(train, result)  # 获取用户历史行为次数
    result = get_user_eloc_count(train, result)  # 获取用户去过这个地点几次
    result = get_user_sloc_count(train, result)  # 获取用户从目的地点出发过几次
    result = get_user_sloc_eloc_count(train, result)  # 获取用户从这个路径走过几次
    result = get_user_eloc_sloc_count(train, result)  # 获取用户从这个路径折返过几次

    result = get_eloc_count(train, result)  # 获取目的地点的热度(目的地)
    result = get_eloc_as_sloc_count(train, result)  # 获取目的地点的热度(出发地)

    result.fillna(0, inplace=True)
    return result