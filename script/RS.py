# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:21:48 2017

@author: Manoh
"""
import os
os.chdir('C:\\AV_RS')
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

pd.options.display.max_columns = 10 
pd.options.display.width = 134
pd.options.display.max_rows = 20

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_csv('train_MLWARE2.csv')
test = pd.read_csv('test_MLWARE2.csv')
matrix = train.pivot('userId','itemId','rating')
items_means = matrix.mean()
user_means = matrix.mean(axis=1)
mzm = matrix-items_means
mz = mzm.fillna(0)
mask = -mzm.isnull()

# round 1
iteration = 0
mse_last = 999
while iteration<10:
    iteration += 1
    svd = TruncatedSVD(n_components=15,random_state=42)
    svd.fit(mz)
    mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

    mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
    print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
    mzsvd[mask] = mzm[mask]

    mz = mzsvd
    if mse_last-mse<0.00001: break
    mse_last = mse

m = mz+items_means
m = m.clip(lower=1,upper=5)

test.head()
test['rating_1'] = test.apply(lambda x:m[m.index==x.userId][x.itemId].values[0],axis=1)

# if items do not have enough info to make prediction, use average value for user
missing = np.where(test.rating.isnull())[0]
test.ix[missing,'rating'] = user_means[test.loc[missing].userId].values

# round 2
iteration = 0
mse_last = 999
while iteration<10:
    iteration += 1
    svd = TruncatedSVD(n_components=20,random_state=42)
    svd.fit(mz)
    mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

    mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
    print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
    mzsvd[mask] = mzm[mask]

    mz = mzsvd
    if mse_last-mse<0.00001: break
    mse_last = mse

m = mz+items_means
m = m.clip(lower=1,upper=5)

test.head()
test['rating_2'] = test.apply(lambda x:m[m.index==x.userId][x.itemId].values[0],axis=1)

test['rating'] = test[['rating_1', 'rating_2']].mean(axis=1)

del test['rating_1']
del test['rating_2']

test.to_csv('submission.csv',index=False,columns=['ID','rating'])
