# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:14:51 2018

@author: CLUO17
"""

#%% load packages
import pandas as pd 
import numpy as np 
from collections import defaultdict

#%% read app data
app_train = pd.read_csv('application_train.csv', index_col = 0)
app_test = pd.read_csv('application_test.csv', index_col = 0)

#%% preprocessing

# remove three rows with errors
app_train = app_train.loc[app_train['CODE_GENDER'] != 'XNA']

# split labels from training
y_train = app_train.iloc[:,0]
app_train = app_train.iloc[:,1:]

app = app_train.append(app_test)

#%% app data encoding 

# encode binary value columns
app['CODE_GENDER'], _ = pd.factorize(app['CODE_GENDER'])
app['FLAG_OWN_CAR'], _ = pd.factorize(app['FLAG_OWN_CAR'])
app['FLAG_OWN_REALTY'], _ = pd.factorize(app['FLAG_OWN_REALTY'])

# encode catogariacal columns
app = pd.get_dummies(app, dummy_na = True)

#%% feature engineering

# replace a strange value 
app['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

# make features
app['EMPLOYED_PERCENT'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']
app['INCOME_CREDIT_RATIO'] = app['AMT_INCOME_TOTAL'] / app['AMT_CREDIT']
app['INCOME_PER_FAM_MEMB'] = app['AMT_INCOME_TOTAL'] / app['CNT_FAM_MEMBERS']
app['ANNUITY_INCOME_RATIO'] = app['AMT_ANNUITY'] / app['AMT_INCOME_TOTAL'] 
app['PAYMENT_RATE'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']

#%% read bureau and bureau_balance

bureau = pd.read_csv('bureau.csv', index_col = 0)
bureau_balance = pd.read_csv('bureau_balance.csv', index_col = 0)

#%% one-hot encoding

bureau = pd.get_dummies(bureau, dummy_na = True)
bureau_balance = pd.get_dummies(bureau_balance, dummy_na = True)

#%% feature engineering

# construct agg calcs
bb_agg_calcs = defaultdict(list)
bb_agg_calcs['MONTHS_BALANCE'] += ['min', 'max', 'size']
for c in bureau_balance.columns:
    bb_agg_calcs[c] += ['mean']

bb_agg = bureau_balance.groupby(bureau_balance.index).agg(bb_agg_calcs)































