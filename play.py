# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 11:56:00 2018

@author: Think
"""

#%% 
import numpy as np 
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from hyperopt import hp, fmin, tpe, space_eval

#%% construct data
data = np.random.randn(200, 2)
data[:100] = data[:100] + 1.5

labels = np.array([0]*100 + [1]*100)

plt.scatter(data[:100,0], data[:100,1], label = '0')
plt.scatter(data[100:,0], data[100:,1], label = '1')
plt.legend()

#%% split training and validation data
idx = np.random.permutation(200)

xTrain = data[idx[:150]]
xVal = data[idx[150:]]
yTrain = labels[idx[:150]]
yVal = labels[idx[150:]]

#%% construct baseline random forest classifier

# use default settings
rfc = lgb.LGBMClassifier(num_leaves = 31,
                         max_depth = -1)
model = rfc.fit(xTrain, yTrain)
yPred = model.predict_proba(xVal)
fpr, tpr, _  = roc_curve(yVal, yPred[:,1])

print(auc(fpr, tpr))

#%% use hyperopt to select hyper-parameters

def objective(hyperparams):
    num_leaves, max_depth, learning_rate, reg_alpha, reg_lambda = hyperparams
    rfc = lgb.LGBMClassifier(num_leaves = num_leaves, 
                             max_depth = max_depth, 
                             learning_rate = learning_rate,
                             reg_alpha = reg_alpha, 
                             reg_lambda = reg_lambda)
    model = rfc.fit(xTrain, yTrain)
    yPred = model.predict_proba(xVal)
    fpr, tpr, _  = roc_curve(yVal, yPred[:,1])
    return 1 - auc(fpr, tpr)

space = [(2 + hp.randint('num_leaves', 50)),
         (2 + hp.randint('max_depth', 10)),
         hp.lognormal('learning_rate', 0.1, 0.2),
         hp.lognormal('reg_alpha', 0.1, 0.2),
         hp.lognormal('reg_lambda', 0.1, 0.2)]

best = fmin(objective, space, max_evals = 500, algo = tpe.suggest)

print(best)
print(space_eval(space, best))

#%% use found best hyper-parameters

rfc = lgb.LGBMClassifier(num_leaves = 22,
                         max_depth = 5,
                         learning_rate=2.3770531717406653,
                         reg_alpha = 4.685257702254753,
                         reg_lambda=1.6049805501100234)
model = rfc.fit(xTrain, yTrain)
yPred = model.predict_proba(xVal)
fpr, tpr, _  = roc_curve(yVal, yPred[:,1])

print(auc(fpr, tpr))















