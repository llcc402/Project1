# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 21:20:08 2018

@author: Think
"""

#%% import packages
import numpy as np 
import pandas as pd 

#%% read data
appTrain = pd.read_csv('application_train.csv', index_col = 0)
appTest = pd.read_csv('application_test.csv', index_col = 0)
bureau = pd.read_csv('bureau.csv', index_col = 0)
bureau_balance = pd.read_csv('bureau_balance.csv', index_col = 0)

#%% feature engineering

# bureau table
bureau_num_agg = bureau.drop('SK_ID_BUREAU', axis=1).groupby(bureau.index).\
                 agg(['max', 'min', 'count', 'sum', 'mean'])
bureau_cat_agg = pd.get_dummies(bureau[['CREDIT_ACTIVE', 'CREDIT_TYPE']]).\
                 groupby(bureau.index).agg(['sum', 'mean'])
                 
# bureau_balance table
