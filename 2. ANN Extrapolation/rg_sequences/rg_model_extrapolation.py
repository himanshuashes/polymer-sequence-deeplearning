#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 23:21:18 2022

@author: himanshu
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

#%%

dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'predictors.npy')
predictors_all = np.load(filename)

filename = os.path.join(dir, 'targets.npy')
targets_all = np.load(filename)

#%%

mms = MinMaxScaler()
predictors_all = mms.fit_transform(predictors_all)

#%%

temp_index = np.empty([targets_all.size, 1], dtype=bool)
for index in range(targets_all.size):
    if targets_all[index] >= 3.31615 and targets_all[index] <= 4.84935:
        temp_index[index] = True
    else:
        temp_index[index] = False

targets_inrange = targets_all[temp_index.reshape(-1)]
predictors_inrange = predictors_all[temp_index.reshape(-1)]

targets_outrange = targets_all[temp_index.reshape(-1) == False]
predictors_outrange = predictors_all[temp_index.reshape(-1) == False]

#%%

targets_train, targets_test_inrange, predictors_train, predictors_test_inrange = train_test_split(targets_inrange, predictors_inrange, test_size = 0.20)

#%%

targets_test = np.vstack([targets_outrange.reshape([targets_outrange.shape[0],1]), targets_test_inrange.reshape([targets_test_inrange.shape[0],1])])
predictors_test = np.vstack([predictors_outrange, predictors_test_inrange])

#%%

fig, ax = plt.subplots()
ax.hist(targets_train, rwidth=0.85, bins=4, color= 'skyblue', label='Training Data')
ax.hist(targets_test, rwidth=0.85, label='Testing Data', color= 'red')

plt.legend(frameon=False) 

ratio = 0.9
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

plt.xlabel('Radius of Gyration', fontsize=15)
plt.ylabel('Count', fontsize=15)

plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2

plt.locator_params(nbins=5)

plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

#%%

flipped_predictors_train = np.flip(predictors_train, 1)
predictors_train = np.vstack([predictors_train, flipped_predictors_train])

targets_train = targets_train.reshape([targets_train.size, 1])
targets_train = np.vstack([targets_train, targets_train])

#%%

n_cols = predictors_all.shape[1]

def regression_model():
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(n_cols,)))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
   
    model.compile(optimizer='adam', loss='mse')
    return model

model = regression_model()

trial_fit = model.fit(predictors_train, targets_train, epochs=500, verbose=1, validation_split=0.1)

#%%

fig, ax = plt.subplots()
ax.plot(trial_fit.history['loss'], label='Training Set', linewidth=3)
ax.plot(trial_fit.history['val_loss'], label='Validation Set', linewidth=3, linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.legend(frameon=False)

ratio = 0.042
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

plt.xlim([0, 30])
plt.ylim([0, 0.6])
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Losses", fontsize=15)

ax.set_facecolor("white")

plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2

plt.locator_params(nbins=5)

plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

#%%

test_run = model.predict(predictors_train)
print(r2_score(targets_train, test_run))

#%%

predictions_inrange = model.predict(predictors_test_inrange)
print(r2_score(targets_test_inrange, predictions_inrange))

#%%

predictions_outrange = model.predict(predictors_outrange)
print(r2_score(targets_outrange, predictions_outrange))

#%%

fig, ax = plt.subplots()
ax.scatter(targets_test_inrange, predictions_inrange, edgecolor='green', facecolor='none', label='Test Data within Range')
ax.scatter(targets_outrange, predictions_outrange, edgecolor='blue', facecolor='None', marker="^", label='Test Data outside Range')
ax.plot(targets_all, targets_all, 'k-', linewidth=3)

plt.legend(frameon=False)

plt.xlabel("True Radius of Gyration", fontsize=15)
plt.ylabel("Predicted Radius of Gyration", fontsize=15)

plt.margins(0)
plt.axis('square')

plt.locator_params(nbins=5)

plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2


plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

#%%

absolute_error_test_inrange = np.absolute(targets_test_inrange.reshape(-1) - predictions_inrange.reshape(-1))
absolute_error_test_outrange = np.absolute(targets_outrange.reshape(-1) - predictions_outrange.reshape(-1))

fig, ax = plt.subplots()
ax.scatter(targets_test_inrange, absolute_error_test_inrange, edgecolor='green', facecolor='none', label='Test Data within Range')
ax.scatter(targets_outrange, absolute_error_test_outrange, edgecolor='blue', facecolor='None', marker="^", label='Test Data outside Range')

plt.margins(0)


plt.xlabel('Radius of Gyration', fontsize=15)
plt.ylabel('Absolute Error', fontsize=15)
plt.legend(frameon=False)
ax.set_facecolor("white")

plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.locator_params(nbins=6)

ratio = 1
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)