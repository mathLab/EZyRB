#!/usr/bin/env python
# coding: utf-8

# # EZyRB Tutorial 2
# ## Test several frameworks at once
#
# In this tutorial, we will explain step by step how to use the **EZyRB** library to test different techniques for building the reduced order model. We will compare different methods of dimensionality reduction, interpolation and accuracy assessment.
#
# We consider here a computational fluid dynamics problem described by the (incompressible) Navier Stokes equations.
# We will be using the **Navier Stokes Dataset** that contains the output data from a full order flow simulation and can be found on **GitHub** under [Smithers library](https://github.com/mathLab/Smithers).
# **Smithers** is developed by **SISSA mathlab** and it contains some useful datasets and a multi-purpose toolbox that inherits functionality from other packages to make the process of dealing with these datasets much easier with more compact coding.
#
# The package can be installed using `python -m pip install smithers -U`, but for a detailed description about installation and usage we refer to original [Github page](https://github.com/mathLab/Smithers/blob/master/README.md).
#
# First of all, we just import the package and instantiate the dataset object.

# In[1]:

import os
from smithers.dataset import NavierStokesDataset
from sklearn.ensemble import RandomForestRegressor
data = NavierStokesDataset()

# The `NavierStokesDataset()` class contains the attribute:
# - `snapshots`: the matrices of snapshots stored by row (one matrix for any output field)
# - `params`: the matrix of corresponding parameters
# - `pts_coordinates`: the coordinates of all nodes of the discretize space
# - `faces`: the actual topology of the discretize space
# - `triang`: the triangulation, useful especially for rendering purposes.
#
# In the details, `data.snapshots` is a dictionary with the following output of interest:
# - **vx:** velocity in the X-direction.
# - **vy:** velocity in the Y-direction.
# - **mag(v):** velocity magnitude.
# - **p:** pressure value.
#
# In total, the dataset contains 500 parametric configurations in a space of 1639 degrees of freedom. In this case, we have just one parameter, which is the velocity (along $x$) we impose at the inlet.

# In[2]:


for name in ['vx', 'vy', 'p', 'mag(v)']:
    print('Shape of {:7s} snapshots matrix: {}'.format(name, data.snapshots[name].shape))

print('Shape of parameters matrix: {}'.format(data.params.shape))


# ### Initial setting
#
# First of all, we import the required packages.
#
# From `EZyRB` we need:
# 1. The `ROM` class, which performs the model order reduction process.
# 2. A module such as `Database`, where the matrices of snapshots and parameters are stored.
# 3. A dimensionality reduction method such as Proper Orthogonal Decomposition `POD` or Auto-Encoder network `AE`.
# 4. An interpolation method to obtain an approximation for the parametric solution for a new set of parameters such as the Radial Basis Function `RBF`, Gaussian Process Regression `GPR`,  K-Neighbors Regressor `KNeighborsRegressor`,  Radius Neighbors Regressor `RadiusNeighborsRegressor` or Multidimensional Linear Interpolator `Linear`.
#
# We also need to import:
# * `numpy:` to handle arrays and matrices we will be working with.
# * `torch:` to enable the usage of Neural Networks
# * `matplotlib.pyplot:` to handle the plotting environment.
# * `matplotlib.tri:` for plotting of the triangular grid.

# In[3]:


# Database module
from ezyrb import Database

# Dimensionality reduction methods
from ezyrb import POD, AE

# Approximation/interpolation methods
from ezyrb import RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN

# Model order reduction calss
from ezyrb import ReducedOrderModel as ROM
from ezyrb import MultiReducedOrderModel as MultiROM
import copy

import numpy as np
import torch
import torch.nn as nn

import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", message="Ill-conditioned matrix ")


# Before starting with the reduced order model, we reduce the dimension
# of the dataset
field = 'vx'
rank = 2
data.params = data.params[:300, :]
data.snapshots[field] = data.snapshots[field][:300, :]

# Divide the dataset in train, validation and test set (0.6, 0.3, 0.1).
mu_train, mu_test, snap_train, snap_test = train_test_split(data.params,
        data.snapshots[field], test_size=0.1, random_state=50)
mu_train, mu_val, snap_train, snap_val = train_test_split(
        mu_train, snap_train, test_size=0.333, random_state=50)

db_train = Database(mu_train, snap_train)
db_val = Database(mu_val, snap_val)
db_test = Database(mu_test, snap_test)

new_params = db_test.parameters_matrix[:3]

# Define some reduction and approximation methods to test
reductions = {
    'POD': POD('svd', rank=rank),
    'AE': AE([100, 10, rank], [rank, 10, 100], nn.Softplus(), nn.Softplus(), 3000),
}

approximations = {
#    'Linear': Linear(),
    'RBF': RBF(),
#    'GPR': GPR(),
#    'KNeighbors': KNeighborsRegressor(),
#    'RadiusNeighbors':  RadiusNeighborsRegressor(),
#    'ANN': ANN([20, 20], nn.Tanh(), 10),
}

def fit_multirom(multirom, db_val, model, fname):
    sigma = multirom.optimize_sigma(db_val, model, sigma_range=[1e-5, 1])
    multirom.fit_weights(db_val, model, sigma=sigma)
    multirom.save(f'multirom_{fname}.pkl')
    return multirom

# Save the dictionary of roms (fitted), if no file is found
if not os.path.exists('multirom.pkl'):
    # Define a dictionary to store the ROMs
    roms_dict = {}
    for redname, redclass in reductions.items():
        for approxname, approxclass in approximations.items():
            rom = ROM(db_train, copy.deepcopy(redclass), copy.deepcopy(approxclass))
            rom.fit()
            roms_dict[redname+'-'+approxname] = rom
    multirom = MultiROM(roms_dict)
    multirom.save('multirom.pkl')

    # Visualize the results on new parameters

    fig, ax = plt.subplots(nrows=3, ncols=int(len(roms_dict.keys()))+1,
            figsize=(16, 8), sharex=True, sharey=True)
    for i, param in enumerate(new_params):
        for j, rom in enumerate(roms_dict.values()):
            ax[i, j].tricontourf(data.triang, *rom.predict([param]))
            ax[i, j].set_title(f'{list(roms_dict.keys())[j]}')
        ax[i, -1].tricontourf(data.triang, db_test.snapshots_matrix[i])
        ax[i, -1].set_title(f'Original at mu={param}')
    plt.show()
    fig.savefig(f'roms_prediction_{rank}modes.png')



# load the multirom
multirom_pre = MultiROM.load('multirom.pkl')
roms_dict = multirom_pre.roms

# build different multiroms with different models
ann = ANN([20, 20], nn.Softplus(), 1000)
models = {
        'GPR': GPR(),
        'KNeighbors': KNeighborsRegressor(),
        'RandomForest': RandomForestRegressor(),
        #'ANN': ann,
        }

multirom = MultiROM(roms_dict)
multiroms = {}
for model_name in models:
    if not os.path.exists(f'multirom_{model_name}.pkl'):
        multirom_ = copy.deepcopy(multirom)
        multiroms[model_name] = fit_multirom(multirom_, db_val,
                models[model_name], model_name)

    multiroms[model_name] = MultiROM.load(f'multirom_{model_name}.pkl')

# print the errors of the individual ROMs and of the multirom
header = '{:10s}'.format('')
for name in approximations:
    header += ' {:>16s}'.format(name)

print(header)
for redname, redclass in reductions.items():
    row = '{:10s}'.format(redname)
    for approxname, approxclass in approximations.items():
        rom = roms_dict[redname+'-'+approxname]
        row += ' {:16e}'.format(rom.test_error(db_test))
    print(row)

for model_name in models:
    row = '{:10s}'.format(model_name)
    multirom_ = multiroms[model_name]
    row += 'MultiROM - {:16e}'.format(multirom_.test_error(db_test))
    print(row)

# POD-RBF                   8.323814e-03
# AE-RBF                    2.725386e-03
# Mixed-ROM (GPR)           4.751706e-03
# Mixed-ROM (KNeighbors)    2.177482e-03
# Mixed-ROM (RandomForest)  2.090957e-03

fig2, ax2 = plt.subplots(nrows=3, ncols=int(len(multiroms.keys())+1),
        figsize=(16, 8), sharex=True, sharey=True)
for i, param in enumerate(new_params):
    for j, model_name in enumerate(multiroms.keys()):
        ax2[i, j].tricontourf(data.triang, *multiroms[model_name].predict([param]))
        ax2[i, j].set_title(f'{model_name}')
    ax2[i, -1].tricontourf(data.triang, db_test.snapshots_matrix[i])
    ax2[i, -1].set_title(f'Original at mu={param}')
plt.show()
fig2.savefig(f'multiroms_prediction_{rank}modes.png')
