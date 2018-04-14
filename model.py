from config import *
from sample import get_sample, get_label
from feature import get_feature

import warnings
warnings.filterwarnings("ignore")

import gc,time
import pandas as pd
import xgboost as xgb

def pred_submit(data):
    data.sort_values(by=['orderid','pred'],inplace=True,ascending=False)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby('orderid',as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on='orderid',how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    data = data[data['rank']<3][['orderid','geohashed_end_loc','rank']]
    data=data.set_index(['orderid','rank']).unstack()
    data.reset_index(inplace=True)
    data.columns = ['orderid', 0, 1, 2]
    return data

def make_train_set(train,test):
    print('开始构造样本...')
    result = get_sample(train,test)

    print('开始构造特征...')
    result = get_feature(train, result)

    return result

if __name__ == '__main__':
    print(time.strftime('%H:%M:%S'))

    df = pd.read_csv(train_path)

    train1 = df[df['starttime'] < '2017-05-21 00:00:00']
    train2 = df[(df['starttime'] >= '2017-05-21 00:00:00') & (df['starttime'] < '2017-05-23 00:00:00')]
    del train2['geohashed_end_loc']

    train = df[df['starttime'] < '2017-05-23 00:00:00']
    test = df[df['starttime'] >= '2017-05-23 00:00:00']
    del test['geohashed_end_loc']

    print('构造训练集')
    train_feat = make_train_set(train1, train2)
    train_feat = get_label(train_feat)
    print('构造测试集')
    test_feat = make_train_set(train, test)

    del train, train1, train2
    gc.collect()

    print(time.strftime('%H:%M:%S'))
    print('开始训练模型...')
    feature = ['biketype', 'distance', 'user_count', 'user_eloc_count', 'user_sloc_count', 'user_sloc_eloc_count', 'user_eloc_sloc_count', 'eloc_count', 'eloc_as_sloc_count']
    xgbtrain = xgb.DMatrix(train_feat[feature], train_feat['label'])
    xgbtest = xgb.DMatrix(test_feat[feature])

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 2017,
        'missing': -1,
        'verbose_eval': True,

        'eta': 0.1,

        'max_depth': 10,
        'min_child_weight': 2,
        'scale_pos_weight': 10,

        'gamma': 30,

        'subsample': 0.886,
        'colsample_bytree': 0.886,

        'lambda': 50,
        'alpha': 10
    }
    num_rounds = 100
    early_stopping_rounds = 20
    watchlist = [(xgbtrain, 'train')]

    model = xgb.train(params, xgbtrain, num_rounds, watchlist)

    del train_feat, xgbtrain
    gc.collect()

    print('开始预测...')
    test_feat.loc[:, 'pred'] = model.predict(xgbtest)

    result = pred_submit(test_feat)
    result = pd.merge(test[['orderid']], result, on='orderid', how='left')
    result.fillna('0', inplace=True)
    result.to_csv('result.csv', index=False, header=False)

    print(time.strftime('%H:%M:%S'))