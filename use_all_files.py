# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:14:51 2018

@author: CLUO17
"""

#%% load packages
import pandas as pd 
import numpy as np 
from collections import defaultdict
import lightgbm as lgb
import datetime

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

#%% read previous
pre = pd.read_csv('previous_application.csv', index_col=1)

#%% 

original_col_names = pre.columns

# one-hot encoding
pre = pd.get_dummies(pre, dummy_na = True)

# find out categorical columns
cat_columns = [c for c in pre.columns if c not in original_col_names]

# replace strange values with np.nan
pre.DAYS_FIRST_DRAWING.replace(365243, np.nan, inplace = True)
pre.DAYS_FIRST_DUE.replace(365243, np.nan, inplace = True)
pre.DAYS_LAST_DUE.replace(365243, np.nan, inplace = True)
pre.DAYS_LAST_DUE_1ST_VERSION.replace(365243, np.nan, inplace = True)
pre.DAYS_TERMINATION.replace(365243, np.nan, inplace = True)

# add new feature:  value_asked / value_received
pre['PRE_APP_CREDIT_RATIO'] = pre['AMT_APPLICATION'] / pre['AMT_CREDIT']

#%% aggreagation 

num_agg_calcs = {'AMT_ANNUITY':['min', 'max', 'sum', 'mean'],
                 'AMT_APPLICATION':['min','max','sum'],
                 'AMT_CREDIT':['min','max','mean'],
                 'AMT_DOWN_PAYMENT':['min', 'max', 'mean'],
                 'AMT_GOODS_PRICE':['min', 'max', 'mean'],
                 'HOUR_APPR_PROCESS_START':['min', 'max', 'mean'],
                 'RATE_DOWN_PAYMENT':['min', 'max', 'sum','mean'],
                 'PRE_APP_CREDIT_RATIO':['min', 'max', 'mean', 'sum', 'var'],
                 'DAYS_DECISION': ['min', 'max', 'mean'],
                 'CNT_PAYMENT': ['mean', 'sum']}

cat_agg_calcs = {}
for c in cat_columns:
    cat_agg_calcs[c] = ['mean']

pre_agg = pre.groupby(pre.index).agg({**num_agg_calcs, **cat_agg_calcs})

# rename column names
pre_agg.columns = [c[0] + '_' + c[1] for c in pre_agg.columns]

#%% read pos_cach_balance
pos_cach_balance = pd.read_csv('POS_CASH_balance.csv', index_col = 1)

#%%
# remove errors
pos_cach_balance = pos_cach_balance.loc[pos_cach_balance.NAME_CONTRACT_STATUS != 'XNA']

# one-hot encoding
pos_cach_balance = pd.get_dummies(pos_cach_balance, dummy_na = True)

# cat columns
cat_columns = ['NAME_CONTRACT_STATUS_Active', 'NAME_CONTRACT_STATUS_Amortized debt',
       'NAME_CONTRACT_STATUS_Approved', 'NAME_CONTRACT_STATUS_Canceled',
       'NAME_CONTRACT_STATUS_Completed', 'NAME_CONTRACT_STATUS_Demand',
       'NAME_CONTRACT_STATUS_Returned to the store',
       'NAME_CONTRACT_STATUS_Signed', 'NAME_CONTRACT_STATUS_nan']

num_agg_calcs = {'MONTHS_BALANCE':['size', 'max', 'mean'],
                 'CNT_INSTALMENT':['max', 'sum'], 
                 'SK_DPD':['max', 'mean'],
                 'SK_DPD_DEF':['max', 'mean']}

cat_agg_calcs = {}
for c in cat_columns:
    cat_agg_calcs[c] = 'mean'
    
pos_cach_balance_agg = pos_cach_balance.groupby(pos_cach_balance.index).\
                                        agg({**num_agg_calcs, **cat_agg_calcs})
                                        
pos_cach_balance_agg.columns = [c[0] + '_' + c[1] for c in pos_cach_balance_agg.columns]

#%% read installments_payments

ins_pay = pd.read_csv('installments_payments.csv', index_col = 1)

#%% 

# create new features: before or late in repay
ins_pay['DPD'] = ins_pay['DAYS_ENTRY_PAYMENT'] - ins_pay['DAYS_INSTALMENT']
ins_pay['DBD'] = ins_pay['DAYS_INSTALMENT'] - ins_pay['DAYS_ENTRY_PAYMENT']
ins_pay['DPD'] = ins_pay['DPD'].apply(lambda x: x > 0)
ins_pay['DBD'] = ins_pay['DBD'].apply(lambda x: x > 0)

# create features: diff and percent should pay and actual payment
ins_pay['INSTALLMENT_ACTUAL_DIFF'] = ins_pay['AMT_INSTALMENT'] - ins_pay['AMT_PAYMENT']
ins_pay['INSTALLMENT_ACTUAL_RATIO'] = ins_pay['AMT_INSTALMENT'] / ins_pay['AMT_PAYMENT']

# there is no "object" type in ins_pay.columns 
num_agg_calcs = {'NUM_INSTALMENT_VERSION':['nunique'],
                 'DPD':['sum', 'mean'],
                 'DBD':['sum', 'mean'],
                 'INSTALLMENT_ACTUAL_DIFF':['sum', 'mean', 'max', 'var'],
                 'INSTALLMENT_ACTUAL_RATIO':['sum', 'mean', 'max', 'var'],
                 'AMT_INSTALMENT':['sum', 'mean'],
                 'AMT_PAYMENT':['sum', 'mean'],
                 'DAYS_ENTRY_PAYMENT':['max', 'min', 'sum']}

ins_pay_agg = ins_pay.groupby(ins_pay.index).agg(num_agg_calcs)

# rename column names
ins_pay_agg.columns = [c[0] + '_' + c[1] for c in ins_pay_agg.columns]

#%% read credit_card_balance
credict_card_balance = pd.read_csv('credit_card_balance.csv', index_col = 1)

#%% 

credict_card_balance = credict_card_balance.drop('SK_ID_PREV', axis = 1)

credict_card_balance_agg = credict_card_balance.groupby(credict_card_balance.index).\
                           agg(['mean', 'max', 'min', 'sum', 'var'])

credict_card_balance_agg.columns = [c[0] + '_' + c[1] for c in credict_card_balance_agg.columns]

#%% combine all the data together
df = app.join(bureau_agg, how = 'left')
df = df.join(pre_agg, how = 'left', rsuffix = 'pre')
df = df.join(pos_cach_balance_agg, how = 'left', rsuffix = 'pos')
df = df.join(ins_pay_agg, how = 'left', rsuffix = 'ins')
df = df.join(credict_card_balance_agg, how = 'left', rsuffix = 'card')

#%% train, valid, test split 

x_train = df.loc[y_train.index]
test_idx = [i for i in df.index if i not in y_train.index]
x_test = df.loc[test_idx]

idx = np.random.permutation(x_train.shape[0])

x_valid = x_train.iloc[idx[250000:]]
x_train = x_train.iloc[idx[:250000]]

y_valid = y_train.iloc[idx[250000:]]
y_train = y_train.iloc[idx[:250000]]

#%% train the model

model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, nthread = 4)

print("=============== start training ================")
print(datetime.datetime.now())
model.fit(x_train, y_train, eval_metric = 'auc',
                  eval_set = [(x_valid, y_valid), (x_train, y_train)],
                  eval_names = ['valid', 'train'], 
                  early_stopping_rounds = 100, verbose = 200)
print("============== training finishes ==============")
print(datetime.datetime.now())