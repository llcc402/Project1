# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 17:23:41 2018

@author: CLUO17
"""

#%%
import pandas as pd
import numpy as np


#%%
training = pd.read_csv('application_train.csv')
count_null = np.sum(training.isnull(), axis = 0)
count_null.loc[count_null > 0]