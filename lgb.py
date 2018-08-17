# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 17:54:20 2018

@author: Think
"""

#%% import packages
import pandas as pd 
import numpy as np 
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import datetime

#%% read data
training = pd.read_csv('application_train.csv', index_col = 0)
test = pd.read_csv('application_test.csv', index_col = 0)

#%%  label encode

# find obj columns
obj_cols = training.dtypes.loc[training.dtypes == 'object'].index

# fill NULL with NUKNOW
training[obj_cols] = training[obj_cols].fillna('UNKNOWN')
test[obj_cols] = test[obj_cols].fillna('UNKNOWN')

# label encode
le = defaultdict(LabelEncoder)
training[obj_cols] = training[obj_cols].apply(lambda x: le[x.name].fit_transform(x))
test[obj_cols] = test[obj_cols].apply(lambda x: le[x.name].transform(x))

# fill numeric NULL with -1
training = training.fillna(-1)
test = test.fillna(-1)

#%% feature engineering
training['CREDIT_INCOME_PERCENT'] = training['AMT_CREDIT'] / training['AMT_INCOME_TOTAL']
test['CREDIT_INCOME_PERCENT'] = test['AMT_CREDIT'] / test['AMT_INCOME_TOTAL']

training['ANNUITY_INCOME_PERCENT'] = training['AMT_ANNUITY'] / training['AMT_INCOME_TOTAL']
test['ANNUITY_INCOME_PERCENT'] = test['AMT_ANNUITY'] / test['AMT_INCOME_TOTAL']

training['CREDIT_TERM'] = training['AMT_ANNUITY'] / training['AMT_CREDIT']
test['CREDIT_TERM'] = test['AMT_ANNUITY'] / test['AMT_CREDIT']

training['DAYS_EMPLOYED_PERCENT'] = training['DAYS_EMPLOYED'] / training['DAYS_BIRTH']
test['DAYS_EMPLOYED_PERCENT'] = test['DAYS_EMPLOYED'] / test['DAYS_BIRTH']

#%% split training into training and validation
idx = np.random.permutation(training.shape[0])
X = training.iloc[idx[:200000]]
valid = training.iloc[idx[200000:]]

#%% fit model and predict
print("start training at")
print(datetime.datetime.now())
model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)

model.fit(X.iloc[:,1:], X.iloc[:,0], eval_metric = 'auc',
                  eval_set = [(valid.iloc[:,1:], valid.iloc[:,0]), (X.iloc[:,1:], X.iloc[:,0])],
                  eval_names = ['valid', 'train'], 
                  early_stopping_rounds = 100, verbose = 200)
        
print("training finished at ")
print(datetime.datetime.now())