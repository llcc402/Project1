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

#%% feature engineering for bureau_balance

# construct agg calcs
bb_agg_calcs = defaultdict(list)
bb_agg_calcs['MONTHS_BALANCE'] += ['min', 'max', 'size']
for c in bureau_balance.columns:
    bb_agg_calcs[c] += ['mean']

bb_agg = bureau_balance.groupby(bureau_balance.index).agg(bb_agg_calcs)

# rename column name, flattening multi-index column names 
bb_agg.columns = [i[0] + '_' + i[1] for i in bb_agg.columns]

# merge bb_agg with bureau
bureau = bureau.join(bb_agg, how = 'left', on = 'SK_ID_BUREAU')

#%% agg bureau

num_columns = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE',
       'DAYS_CREDIT_ENDDATE', 'AMT_CREDIT_MAX_OVERDUE',
       'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
       'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE',
       'AMT_ANNUITY',  'MONTHS_BALANCE_min', 
       'MONTHS_BALANCE_max', 'MONTHS_BALANCE_size']

cat_columns = ['DAYS_ENDDATE_FACT', 'CREDIT_ACTIVE_Active', 'CREDIT_ACTIVE_Bad debt',
       'CREDIT_ACTIVE_Closed', 'CREDIT_ACTIVE_Sold', 'CREDIT_ACTIVE_nan',
       'CREDIT_CURRENCY_currency 1', 'CREDIT_CURRENCY_currency 2',
       'CREDIT_CURRENCY_currency 3', 'CREDIT_CURRENCY_currency 4',
       'CREDIT_CURRENCY_nan', 'CREDIT_TYPE_Another type of loan',
       'CREDIT_TYPE_Car loan', 'CREDIT_TYPE_Cash loan (non-earmarked)',
       'CREDIT_TYPE_Consumer credit', 'CREDIT_TYPE_Credit card',
       'CREDIT_TYPE_Interbank credit',
       'CREDIT_TYPE_Loan for business development',
       'CREDIT_TYPE_Loan for purchase of shares (margin lending)',
       'CREDIT_TYPE_Loan for the purchase of equipment',
       'CREDIT_TYPE_Loan for working capital replenishment',
       'CREDIT_TYPE_Microloan', 'CREDIT_TYPE_Mobile operator loan',
       'CREDIT_TYPE_Mortgage', 'CREDIT_TYPE_Real estate loan',
       'CREDIT_TYPE_Unknown type of loan', 'CREDIT_TYPE_nan',
       'MONTHS_BALANCE_mean', 'STATUS_0_mean', 'STATUS_1_mean',
       'STATUS_2_mean', 'STATUS_3_mean', 'STATUS_4_mean', 'STATUS_5_mean',
       'STATUS_C_mean', 'STATUS_X_mean', 'STATUS_nan_mean']

num_agg_calcs = {'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
                 'CREDIT_DAY_OVERDUE':['mean', 'max'], 
                 'DAYS_CREDIT_ENDDATE':['max', 'min', 'mean'],
                 'AMT_CREDIT_MAX_OVERDUE':['max', 'mean'],
                 'CNT_CREDIT_PROLONG':['sum'],
                 'AMT_CREDIT_SUM':['max', 'mean', 'sum'],
                 'AMT_CREDIT_SUM_DEBT':['max', 'mean', 'sum'],
                 'AMT_CREDIT_SUM_LIMIT':['mean', 'sum'],
                 'AMT_CREDIT_SUM_OVERDUE':['max', 'mean'],
                 'DAYS_CREDIT_UPDATE':['mean'],
                 'AMT_ANNUITY':['max', 'mean'],
                 'MONTHS_BALANCE_min':['min'],
                 'MONTHS_BALANCE_max':['max'],
                 'MONTHS_BALANCE_size':['sum', 'mean']}

cat_agg_calcs = dict()
for c in cat_columns:
    cat_agg_calcs[c] = ['mean']

bureau_agg_calcs = {**num_agg_calcs, **cat_agg_calcs}

bureau_agg = bureau.groupby('SK_ID_BUREAU').agg(bureau_agg_calcs)

# rename column name
bureau_agg.columns = [c[0] + '_' + c[1] for c in bureau_agg.columns]

#%% 















