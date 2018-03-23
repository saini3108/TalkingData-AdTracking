import time

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import train_test_split

path = './input/'


def dataPreProcessTime(df):
    df['click_time'] = pd.to_datetime(df['click_time']).dt.date
    df['click_time'] = df['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    return df

start_time = time.time()

train = pd.read_csv(path+"train.csv", nrows=20000000)
test = pd.read_csv(path+"test.csv")

print('[{}] Finished to load data'.format(time.time() - start_time))

train = dataPreProcessTime(train)
test = dataPreProcessTime(test)

y = train['is_attributed']
train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop('click_id', axis=1, inplace=True)

print('[{}] Start XGBoost Training'.format(time.time() - start_time))

params = {'eta': 0.1, 
          'max_depth': 4, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':100,
          'alpha':4,
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'random_state': 99, 
          'silent': True}
          
x1, x2, y1, y2 = train_test_split(train, y, test_size=0.3, random_state=99)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 250, watchlist, maximize=True, verbose_eval=10)

print('[{}] Finish XGBoost Training'.format(time.time() - start_time))

sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('xgb_sub.csv',index=False)