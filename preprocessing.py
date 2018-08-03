# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:51:19 2018

@author: CLUO17
"""

#%% IMPORT PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% READ MAIN DATA
data1 = pd.read_csv('~/Documents/Kaggle/application_train.csv',
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

