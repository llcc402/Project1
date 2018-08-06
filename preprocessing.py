# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:51:19 2018

@author: CLUO17
"""

#%% IMPORT PACKAGES
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import preprocessing

#%% READ MAIN DATA
data1 = pd.read_csv('D:/KaggleCompetition/CreditMaster/Data/application_train.csv',
                    header = 0, 
                    index_col = 0)
test = pd.read_csv('D:/KaggleCompetition/CreditMaster/Data/application_test.csv',
                   header = 0,
                   index_col = 0)

#%% PROCESS MAIN DATA: fill obj NULLs with UNKNOWN
obj_cols = data1.dtypes[data1.dtypes == 'object'].index
obj_count_null_cols = np.sum(data1[obj_cols].isnull())
obj_null_cols = obj_count_null_cols[obj_count_null_cols > 0].index

# fill NULLs with UNKNOWN
data1[obj_null_cols] = data1[obj_null_cols].fillna('UNKNOWN') 

#%% PROCESS MAIN DATA: fill numeric NULLS with numbers
num_cols = data1.dtypes[data1.dtypes == 'float64'].index
num_count_null_cols = np.sum(data1[num_cols].isnull())
num_null_cols = num_count_null_cols[num_count_null_cols > 0].index

# fill NULLs with -1
data1[num_null_cols] = data1[num_null_cols].fillna(-1) 

#%% FILL NULLS IN TEST DATA
obj_cols = test.dtypes[test.dtypes == 'object'].index
test[obj_cols] = test[obj_cols].fillna('UNKOWN')

num_cols = test.dtypes[test.dtypes == 'float64'].index
test[num_cols] = test[num_cols].fillna(-1)

#%% LABEL ENCODER
le = preprocessing.LabelEncoder()
train_test = pd.concat([data1.iloc[:,1:], test], axis = 0)
train_test[obj_cols] = train_test[obj_cols].apply(le.fit_transform)

#%% CONSTRUCT TRAINING AND TEST DATA
training = train_test.iloc[:307511]
test = train_test.iloc[307511:]
y = data1['TARGET']

#%% SPLIT TRAINING DATA TO REGULAARIZE MODEL
idx = np.random.permutation(307511)
xTrain = training.iloc[idx[:250000]]
yTrain = y.iloc[idx[:250000]]
xVal = training.iloc[idx[250000:]]
yVal = y.iloc[idx[250000:]]
weights = yTrain * 9 + 1

#%% TRAIN A RANDOM FOREST
rfc = RFC(n_estimators=500, max_features=15, max_depth=5)
model = rfc.fit(xTrain, yTrain, weights)
print('=============== The training score ===============')
print(model.score(xTrain, yTrain))
print('============== The valication score ==============')
print(model.score(xVal, yVal))