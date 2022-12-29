#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 19:20:16 2022

@author: himanshu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%

data = pd.read_excel('/home/himanshu/Projects/ANN Topology/st_sequences/algo_data.xlsx')

#%%

nodes_1 = data.iloc[0:14, 0:1]
losses_1 = data.iloc[0:14, 2:4]

fig, ax = plt.subplots()
ax.plot(nodes_1, np.log(losses_1.iloc[:,0]), label='One Hidden Layer', marker='o', linewidth=3, markersize=8)
# plt.plot(nodes_1, losses_1.iloc[:,1], label='Validation Loss',  marker='x')



#%%

losses_2 = data.iloc[14:28, 2:4]
nodes_2 = data.iloc[14:28, 0:1] + 350

ax.plot(nodes_2, np.log(losses_2.iloc[:,0]), label='Two Hidden Layers', marker='*', linewidth=3, markersize=8)
# plt.plot(nodes_2, losses_2.iloc[:,1],   marker='x')

#%%

losses_3 = data.iloc[28:36, 2:4]
nodes_3 = data.iloc[28:36, 0:1] + 350 + 350

ax.plot(nodes_3, np.log(losses_3.iloc[:,0]), label='Three Hidden Layers', marker='x', linewidth=3, markersize=8)
# plt.plot(nodes_3, losses_3.iloc[:,1],   marker='x')

#%%

losses_4 = data.iloc[36:40, 2:4]
nodes_4 = data.iloc[36:40, 0:1] + 1100 - 10

ax.plot(nodes_4, np.log(losses_4.iloc[:,0]), label='Four Hidden Layers', marker='v', linewidth=3, markersize=8)
# plt.plot(nodes_4, losses_4.iloc[:,1],   marker='x')

#%%

losses_5 = data.iloc[40:44, 2:4]
nodes_5 = data.iloc[40:44, 0:1] + 1190 - 10

ax.plot(nodes_5, np.log(losses_5.iloc[:,0]), label='Five Hidden Layers', marker='^', linewidth=3, markersize=8)

#%%

losses_6 = data.iloc[44:, 2:4]
nodes_6 = data.iloc[44:, 0:1] + 1280 - 10

ax.plot(nodes_6, np.log(losses_6.iloc[:,0]), label='Six Hidden Layers', marker='p', linewidth=3, markersize=8)

#%%

plt.ylabel('log(Loss)', fontsize=15)
plt.xlabel('Node', fontsize=15)

# plt.axvline(x = 250)
# plt.axvline(x = 550)
# plt.axvline(x = 600)

plt.legend(frameon=False)

#%%

plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2

plt.locator_params(nbins=6)

plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

ratio = 1
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)