#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 19:21:57 2022

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
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

from sklearn.utils import shuffle

#%%

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = False)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    
    
    x_temp = np.arange(np.min(x)-0.8, np.max(x)+0.2, 0.01)
    y_temp = x_temp
    
    ax.scatter(x, y, c=z, **kwargs )
    ax.plot(x_temp, y_temp,'r-', linewidth=3)
    plt.xlabel('True Radius of Gyration', fontsize=15)
    plt.ylabel('Predicted Radius of Gyration', fontsize=15)
  
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2
    
    plt.locator_params(nbins=4)
    
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)
    
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.margins(0)
      
    plt.axis('square')
    
    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Number of Points', fontsize=15)
    
    
    from matplotlib import ticker
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    
    
    return ax

#%%

dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'predictors.npy')
predictors_all = np.load(filename)

filename = os.path.join(dir, 'targets.npy')
targets_all = np.load(filename)

#%%

# predictors_all, targets_all = shuffle(predictors_all, targets_all, random_state=0)

targets_all_sorted_idx = np.argsort(targets_all, axis = 0)
targets_all_sorted = np.sort(targets_all, axis = 0)
predictors_all_sorted = predictors_all[targets_all_sorted_idx].reshape([29612, 100])

#%%

# idx = random.sample(range(predictors_all_sorted.shape[0]), 500)
# predictors_train = predictors_all_sorted[idx]
# targets_train = targets_all_sorted[idx]

#%%

idx = np.linspace(0, 29611, num=500, dtype=int)
predictors_train = predictors_all_sorted[idx]
targets_train = targets_all_sorted[idx]

#%%

# flipped_predictors_train = np.flip(predictors_train, 1)
# predictors_train = np.vstack([predictors_train, flipped_predictors_train])

# targets_train = targets_train.reshape([targets_train.size, 1])
# targets_train = np.vstack([targets_train, targets_train])

predictors_test = np.delete(predictors_all, idx, axis=0)
targets_test = np.delete(targets_all, idx)

#%%

sc = MinMaxScaler()
predictors_train_norm = sc.fit_transform(predictors_train)
predictors_test_norm = sc.fit_transform(predictors_test)

n_cols = predictors_train_norm.shape[1]

plt.figure()
counts, edges, plot = plt.hist(targets_all, label='All Data')
plt.hist(targets_test, label='Testing Data')
plt.hist(targets_train, label='Training Data')

plt.legend()

#%%

fig, ax = plt.subplots()
a1, b1, c1 = ax.hist(targets_all, rwidth=0.85, label='All Data')
a2, b2, c2 = ax.hist(targets_test, rwidth=0.85, label='Test Data')
a3, b3, c3 = ax.hist(targets_train, rwidth=0.85, label='Training Data')
plt.legend(frameon=False)
plt.grid(False)
ax.set_facecolor("white")

ratio = 0.9
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

plt.xlabel('Radius of Gyration', fontsize=15)
plt.ylabel('Count', fontsize=15)

plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)

plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2

plt.locator_params(nbins=5)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

#%%
def regression_model():
    model = Sequential()
    model.add(Dense(300, activation='relu', input_shape=(n_cols,)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
   
    model.compile(optimizer='adam', loss='mse')
    return model

model = regression_model()

trial_fit = model.fit(predictors_train_norm, targets_train, epochs=100, verbose=1, validation_split=0.1)

#%%

fig, ax = plt.subplots()
ax.plot(trial_fit.history['loss'], label='Training Set', linewidth=3)
# ax.plot(trial_fit.history['val_loss'], label='Validation Set', linewidth=3, linestyle='--')
plt.legend()
ratio = 0.18
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

plt.xlim([0, 20])
plt.ylim([0, 5])
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Losses", fontsize=15)
plt.legend(frameon=False)

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
test_run = model.predict(predictors_train_norm)

print(r2_score(targets_train, test_run))

# density_scatter(targets_train.reshape(-1), test_run.reshape(-1), bins = [50,50])
#%%
fig, ax = plt.subplots()
# ratio = 0.9
# x_left, x_right = ax.get_xlim()
# y_low, y_high = ax.get_ylim()
# ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
ax.scatter(targets_train, test_run)
ax.plot(targets_train, targets_train, 'k-')
plt.title('Training | ' + str(r2_score(targets_train, test_run)))

#%%

density_scatter(targets_train.reshape(-1), test_run.reshape(-1), bins = [10,20])

#%%

predictions = model.predict(predictors_test_norm)

print(r2_score(targets_test, predictions))

# density_scatter(targets_test.reshape(-1), predictions.reshape(-1), bins = [20,20])
#%%
fig, ax = plt.subplots()
ratio = 0.77
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
ax.scatter(targets_test, predictions)
ax.plot(targets_test, targets_test, 'k-')
plt.title("Testing | " +str(r2_score(targets_test, predictions)))

#%%

density_scatter(targets_test.reshape(-1), predictions.reshape(-1), bins = [10,20])