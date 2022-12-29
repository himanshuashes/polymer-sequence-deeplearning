#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 22:16:47 2022

@author: himanshu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

#%%

from timeit import default_timer as timer

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

cb = TimingCallback()

#%%

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = False )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    
    
    x_temp = np.arange(np.min(x)-1, np.max(x)+2)
    y_temp = x_temp
    
    ax.scatter( x, y, c=z, **kwargs )
    ax.plot(x_temp, y_temp ,'r-', linewidth=3)
    plt.xlabel('True Energies', fontsize=15)
    plt.ylabel('Predicted Energies', fontsize=15)
  
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
filename = os.path.join(dir, 'chain_double_clean.npy')
predictors = np.load(filename)

filename = os.path.join(dir, 'F_double_clean.npy')
targets = np.load(filename)

flipped_predictors = np.flip(predictors, 1)
predictors = np.vstack([predictors, flipped_predictors])

targets = targets.reshape([targets.size, 1])
targets = np.vstack([targets, targets])

#%%

predictors_train, predictors_test, targets_train, targets_test = train_test_split(predictors, targets, test_size=0.20, random_state=8, shuffle=True)

#%%

n_cols = predictors.shape[1]

#%%

def regression_model():
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(n_cols,)))
    model.add(Dropout(0.2))
    model.add(Dense(19, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(18, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(17, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(14, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
   
    
    model.compile(optimizer='adam', loss='mse')
    return model

model = regression_model()

trial_fit = model.fit(predictors_train, targets_train, epochs=200, verbose=1, validation_split=0.1, callbacks=[cb])

#%%

fig, ax = plt.subplots()
ax.plot(trial_fit.history['loss'], label='Training Set', linewidth=3)
ax.plot(trial_fit.history['val_loss'], label='Validation Set', linewidth=3, linestyle='--')

ratio = 0.132
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

plt.xlim([0, 30])
plt.ylim([0, 130])
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

test_run = model.predict(predictors_train)

print(r2_score(targets_train, test_run))

density_scatter(targets_train.reshape(-1), test_run.reshape(-1), bins = [50,50])

#%%

predictions = model.predict(predictors_test)

print(r2_score(targets_test, predictions))

density_scatter(targets_test.reshape(-1), predictions.reshape(-1), bins = [60,60])

#%%

time = np.asarray(cb.logs)

cum_time = np.cumsum(time)

plt.figure()
plt.plot(time)
plt.title("Training Time per epoch")
plt.xlabel("Epoch")
plt.ylabel('Time (seconds)')

#%%
print(sum(cb.logs))