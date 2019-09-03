# -*- coding: utf-8 -*-
"""
plot trees data

@author: David Yanni
"""

#%% Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#%% Read the data
path1 = 'C://Users//root//Documents//snowflakeTreesDat.csv'
path2 = 'C://Users//root//Downloads//treeData.csv'
path3 = 'C://Users/root//Documents//un_dat.csv'
treesData = pd.read_csv(path2, skiprows=0)
snData = pd.read_csv(path1,names=['nNodes','alpha'])
unData = pd.read_csv(path3,names=['nNodes','alpha'])

n_ternary = treesData.nNodesTernary.dropna().values
alpha_ternary = treesData.AlphaTernary.dropna().values
n_binary = treesData.nNodesBinary.dropna().values
alpha_binary = treesData.AlphaBinary.dropna().values
n_sn = np.array([1]+list(snData.nNodes.values))
alpha_sn = np.array([1]+list(snData.alpha.values))

n_neighb = np.array([2*i for i in range(2,68)])
alpha_neighb = .75*np.ones(n_neighb.size)

#%% Plotting

fig,ax = plt.subplots(figsize=(8,6))
ax.plot(n_ternary,alpha_ternary,color='purple',marker='o',label = "Ternary Tree",mec="black",mew=2,ms=10,lw=2,alpha=.7)
ax.plot(n_binary,alpha_binary,color='red',marker='o', label = "Binary Tree",mec="black",mew=2,ms=10,lw=2,alpha=.7)
ax.plot(n_sn,alpha_sn,color='cadetblue',marker='o', label = "Snowflake Tree",mec="black",mew=2,ms=10,lw=2,alpha=.7)
ax.plot(unData.nNodes,unData.alpha,color='green',marker='o', ms=7,alpha=.4,mew=1,mec='k', label = "Filament",lw=2)
ax.plot(n_neighb,alpha_neighb,color='orange',marker='o', ms=.2, label = "Neighbor Graph",lw=3)

ax.grid(lw=1,color='gray')
ax.set_ylabel(r"Crossover Specialization Power,  $\mathbf{\alpha^*}$",fontsize=18)
ax.set_xlabel(r"Number of individuals in group, $n$",fontsize=18)
ax.set_xticklabels(labels = [20*i for i in range(8)],fontsize=14)
ax.set_ylim(.63,.90)
ax.set_yticklabels(labels = [round(.60+.05*i,2) for i in range(7)],fontsize=14)
ax.legend()
