# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
def calc_accuracy(label, pred):
    SS_res = 0
    SS_tot = 0
    avg = np.sum(label)/len(label)
    for i in range(len(label)):
        SS_res += (label[i] - pred[i])**2
        SS_tot += (label[i] - avg)**2

    R_2 = 1 - SS_res/SS_tot
    return R_2


train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv') 
train_set2 = pd.read_csv('train.csv', header=None)
test_set2 = pd.read_csv('test.csv', header=None)

para1 = train_set2.iloc[1:,6:8].to_numpy()
para2 = test_set2.iloc[1:,6:8].to_numpy()
x_test = np.hstack((para2,test_set2.iloc[1:,113:213].to_numpy()))
x_test_2 = np.hstack((para2,test_set2.iloc[1:,13:113].to_numpy()))

y = train_set[['train_error']].to_numpy()
x = np.hstack((para1,train_set2.iloc[1:,117:217].to_numpy()))

y_2 = train_set[['val_error']].to_numpy()
x_2 = np.hstack((para1,train_set2.iloc[1:,17:117].to_numpy()))

x11,x12,y11,y12 = train_test_split(x, y, test_size = 0.2)
x21,x22,y21,y22 = train_test_split(x_2, y_2, test_size = 0.2)
'''
x_test = np.hstack((para2, test_set[113:213].to_numpy()))
x_test_2 = np.hstack((para2, test_set[13:113].to_numpy()))
y = train_set[['train_error']].to_numpy()
x = np.hstack((para1, train_set[117:217].to_numpy()))

y_2 = train_set[['val_error']].to_numpy()
x_2 = np.hstack((para1, train_set[17:117].to_numpy()))

'''
# choose training accuracy
'''
x_test = test_set[['train_accs_49']].to_numpy()

# choose validation accuracy
x_test_2 = test_set[['val_accs_49']].to_numpy()

# choose train error
y = train_set[['train_error']].to_numpy()
x = train_set[['train_accs_49']].to_numpy()

y_2 = train_set[['val_error']].to_numpy()
x_2 = train_set[['val_accs_49']].to_numpy()
'''
rf = RandomForestRegressor()
et = ExtraTreesRegressor()
cb = CatBoostRegressor()
gb = GradientBoostingRegressor()
#regressor = LGBMRegressor()
#regressor = ExtraTreesRegressor()
#regressor = MLPRegressor()
#regressor = LinearRegression()
#regressor = GradientBoostingRegressor()
'''
rf.fit(x11, y11.ravel())
et.fit(x11, y11.ravel())
gb.fit(x11, y11.ravel())
ab.fit(x11, y11.ravel())
lr.fit(x11, y11.ravel())

rf.fit(x, y.ravel())
et.fit(x, y.ravel())
gb.fit(x, y.ravel())
ab.fit(x, y.ravel())
lr.fit(x, y.ravel())
pred_y = 0.25*rf.predict(x12) + 0.25*et.predict(x12) + 0.25*gb.predict(x12) + 0.25*ab.predict(x12)
pred_y_test = 0.25*rf.predict(x_test) + 0.25*et.predict(x_test) + 0.25*gb.predict(x_test) + 0.25*ab.predict(x_test)
accu_1 = calc_accuracy(y12,pred_y)
print("accu_1")
print(accu_1)

# linear regression for validation set
rf.fit(x21, y21.ravel())
et.fit(x21, y21.ravel())
gb.fit(x21, y21.ravel())
ab.fit(x21, y21.ravel())
lr.fit(x21, y21.ravel())

rf.fit(x_2, y_2.ravel())
et.fit(x_2, y_2.ravel())
gb.fit(x_2, y_2.ravel())
ab.fit(x_2, y_2.ravel())
lr.fit(x_2, y_2.ravel())
pred_y_2 = 0.25*rf.predict(x22) + 0.25*et.predict(x22) + 0.25*gb.predict(x22) + 0.25*ab.predict(x22)
pred_y_test_2 = 0.25*rf.predict(x_test_2) + 0.25*et.predict(x_test_2) + 0.25*gb.predict(x_test_2) + 0.25*ab.predict(x_test_2)
accu_2 = calc_accuracy(y22,pred_y_2)
print("accu_2")
print(accu_2)
'''
params = {
    'booster': 'gbtree',
    'objective': 'reg:gamma',
    'gamma': 0.1,
    'max_depth': 5
}

dtrain1 = xgb.DMatrix(x11, y11)
regressor=xgb.train(params, dtrain1)
dtest11 = xgb.DMatrix(x12)
dtest12 = xgb.DMatrix(x_test)
pred_y1 = regressor.predict(dtest11)
dtrain1 = xgb.DMatrix(x, y)
regressor=xgb.train(params, dtrain1)
pred_y_test1 = regressor.predict(dtest12)
#accu_1 = calc_accuracy(y12,pred_y)
#print("accu_1")
#print(accu_1)

# linear regression for validation set
dtest = xgb.DMatrix(x21, y21)
regressor=xgb.train(params, dtest)
dtest21 = xgb.DMatrix(x22)
dtest22 = xgb.DMatrix(x_test_2)
pred_y_21 = regressor.predict(dtest21)
dtrain1 = xgb.DMatrix(x_2, y_2)
regressor=xgb.train(params, dtrain1)
pred_y_test_21 = regressor.predict(dtest22)
#accu_2 = calc_accuracy(y22,pred_y_2)
#print("accu_2")
#print(accu_2)
# linear regression for train set

params2 = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'}, 
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

dtrain1 = lgb.Dataset(x11, y11.ravel())
regressor=lgb.train(params2, dtrain1)
dtest11 = lgb.Dataset(x12)
dtest12 = lgb.Dataset(x_test)
pred_y2 = regressor.predict(x12)
dtrain1 = lgb.Dataset(x, y.ravel())
regressor=lgb.train(params2, dtrain1)
pred_y_test2 = regressor.predict(x_test)
#accu_1 = calc_accuracy(y12,pred_y)
#print("accu_1")
#print(accu_1)

# linear regression for validation set
dtest = lgb.Dataset(x21, y21.ravel())
regressor=lgb.train(params2, dtest)
dtest21 = lgb.Dataset(x22)
dtest22 = lgb.Dataset(x_test_2)
pred_y_22 = regressor.predict(x22)
dtest = lgb.Dataset(x_2, y_2.ravel())
regressor=lgb.train(params2, dtest)
pred_y_test_22 = regressor.predict(x_test_2)
#accu_2 = calc_accuracy(y22,pred_y_2)
#print("accu_2")
#print(accu_2)

#regressor = CatBoostRegressor()

# linear regression for train set
#regressor.fit(x11, y11.ravel())
rf.fit(x11, y11.ravel())
et.fit(x11, y11.ravel())
gb.fit(x11, y11.ravel())
cb.fit(x11, y11.ravel())
pred_y =  (gb.predict(x12) + cb.predict(x12) + pred_y1 + pred_y2)/4
#rf.predict(x12) + et.predict(x12) + 
rf.fit(x, y.ravel())
et.fit(x, y.ravel())
gb.fit(x, y.ravel())
cb.fit(x, y.ravel())
pred_y_test = (gb.predict(x_test) + cb.predict(x_test) + pred_y_test1 + pred_y_test2)/4
#rf.predict(x_test) + et.predict(x_test) + 
#accu_1 = calc_accuracy(y12,pred_y)
#print(accu_1)

# linear regression for validation set
#regressor.fit(x21,y21)
rf.fit(x21, y21.ravel())
et.fit(x21, y21.ravel())
gb.fit(x21, y21.ravel())
cb.fit(x21, y21.ravel())
pred_y_2 = (gb.predict(x22) + cb.predict(x22) + pred_y_21 + pred_y_22)/4
#rf.predict(x22) + et.predict(x22) + 
rf.fit(x_2, y_2.ravel())
et.fit(x_2, y_2.ravel())
gb.fit(x_2, y_2.ravel())
cb.fit(x_2, y_2.ravel())
pred_y_test_2 = (gb.predict(x_test_2) + cb.predict(x_test_2) + pred_y_test_21 + pred_y_test_22)/4
#rf.predict(x_test_2) + et.predict(x_test_2) + 
print(np.vstack((y12,y22)).shape)
print(np.vstack((pred_y.reshape(-1,1),pred_y_2.reshape(-1,1))).shape)
accu_2 = calc_accuracy(np.vstack((y12,y22)),np.vstack((pred_y.reshape(-1,1),pred_y_2.reshape(-1,1))))
print(accu_2)

result = []

for i in range(len(pred_y_test_2)):
    result.append(pred_y_test_2[i])
    result.append(pred_y_test[i])

result = np.asarray(result)

for i in range(len(result)):
    if result[i]<0:
        result[i] = 0


df = pd.DataFrame(data=result)
df.to_csv('test_results_linreg.csv', encoding='utf-8', index=False)
