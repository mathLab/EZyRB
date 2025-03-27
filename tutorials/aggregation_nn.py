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
from ezyrb import Database
from ezyrb import POD, AE
from ezyrb import RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN
from ezyrb import ReducedOrderModel as ROM
from ezyrb import MultiReducedOrderModel as MultiROM
from ezyrb.plugin import Aggregation, DatabaseSplitter
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ann_demo import ANN_Demo
import warnings
warnings.filterwarnings("ignore", message="Ill-conditioned matrix ")

# Before starting with the reduced order model, we reduce the dimension
# of the dataset
field = 'vx'
rank = 2
coords = data.pts_coordinates.T
data.params = data.params[:300, :]
data.snapshots[field] = data.snapshots[field][:300, :]

#### Commented part because we use the DatabaseSplitter plugin
## Divide the dataset in train, validation and test set (0.6, 0.3, 0.1).
#mu_train, mu_test, snap_train, snap_test = train_test_split(data.params,
#        data.snapshots[field], test_size=0.1, random_state=50)
#mu_train, mu_val, snap_train, snap_val = train_test_split(
#        mu_train, snap_train, test_size=0.333, random_state=50)
#
#db_train = Database(mu_train, snap_train, coords)
#db_val = Database(mu_val, snap_val, coords)
#db_test = Database(mu_test, snap_test, coords)
db_all = Database(data.params, data.snapshots[field], coords)


## Define some new parameters to test
new_params = db_test.parameters_matrix[:3]

# Define some reduction and approximation methods to test
reductions = {
    'POD': POD('svd', rank=rank),
#    'AE': AE([100, 10, rank], [rank, 10, 100], nn.Softplus(), nn.Softplus(), 3000),
}

approximations = {
    'Linear': Linear(),
    'RBF': RBF(),
    'GPR': GPR(),
#    'KNeighbors': KNeighborsRegressor(),
#    'RadiusNeighbors':  RadiusNeighborsRegressor(),
#    'ANN': ANN([20, 20], nn.Tanh(), 10),
}



# Save the dictionary of roms (fitted), if no file is found
if not os.path.exists(f'multirom_{rank}modes/multirom_noaggregation.pkl'):
    os.makedirs(f'multirom_{rank}modes', exist_ok=True)
    # Define a dictionary to store the ROMs
    roms_dict = {}
    db_splitter_plugin = DatabaseSplitter(train=180, validation=90, test=0,
                                            predict=30, seed=50)
    for redname, redclass in reductions.items():
        for approxname, approxclass in approximations.items():
            rom = ROM(db_all, copy.deepcopy(redclass), copy.deepcopy(approxclass),
                      plugins=[db_splitter_plugin])
            roms_dict[redname+'-'+approxname] = rom
    # Build simple multirom without aggregation and save it
    multirom = MultiROM(roms_dict)
    # fit the multirom (only ROMs are fitted here, no aggregation is performed)
    multirom.fit()
    multirom.save(f'multirom_{rank}modes/multirom_noaggregation.pkl')


# load the multirom
multirom_pre = MultiROM.load(f'multirom_{rank}modes/multirom_noaggregation.pkl')
roms_dict = multirom_pre.roms


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
fig.savefig(f'multirom_{rank}modes/roms_prediction_{rank}modes.png', dpi=300)

# Build different multiroms with different models
ann_pred = ANN_Demo(multirom_pre, [20, 20, 20], nn.Softplus(), stop_training=10000, lr=3e-3)
ann_ = ANN_Demo(multirom_pre, [10, 10], nn.Softplus(), stop_training=10000, lr=3e-3)
ann_2 = ANN_Demo(multirom_pre, [3, 3, 3], nn.Softplus(), stop_training=10000, lr=3e-3)

# models used only to predict the weights (using standard aggregation formula)
models_predict = {
        #'RBF': RBF(),
        #'GPR': GPR(),
        'KNeighbors': KNeighborsRegressor(),
        'RandomForest': RandomForestRegressor(),
        'ANN': ann_pred,
        }

# models used to fit and predict the weights (have space also as inputs)
models = {'ANN_fit_10_10': ann_,
          'ANN_fit_3_3_3': ann_2}

# Fit the multiroms with Predict models and save the result
multiroms = {}
for model_name in models_predict:
    if not os.path.exists(f'multirom_{rank}modes/multirom_{model_name}_sigma05.pkl'):
        multirom_ = MultiROM(roms_dict, plugins=[Aggregation(
            #fit_function=models[model_name])])
            predict_function=models_predict[model_name])])
        multirom_.fit()
        multirom_.save(f'multirom_{rank}modes/multirom_{model_name}_sigma05.pkl')
    multiroms[model_name] = MultiROM.load(f'multirom_{rank}modes/multirom_{model_name}_sigma05.pkl')

# Fit the multiroms with Fit+predict models and save the result
for model_name in models:
    if not os.path.exists(f'multirom_{rank}modes/multirom_{model_name}.pkl'):
        multirom_ = MultiROM(roms_dict, plugins=[Aggregation(
            fit_function=models[model_name])])
            #predict_function=models[model_name])])
        multirom_.fit()
        multirom_.save(f'multirom_{rank}modes/multirom_{model_name}.pkl')
    multiroms[model_name] = MultiROM.load(f'multirom_{rank}modes/multirom_{model_name}.pkl')

# Print the errors of the individual ROMs and of the multirom
header = '{:10s}'.format('')
for name in approximations:
    header += ' {:>16s}'.format(name)

print(header)
for redname, redclass in reductions.items():
    row = '{:10s}'.format(redname)
    for approxname, approxclass in approximations.items():
        rom = roms_dict[redname+'-'+approxname]
        row += ' {:16e}'.format(rom.test_error(db_val))
    print(row)

for model_name in multiroms:
    row = '{:10s}'.format(model_name)
    multirom_ = multiroms[model_name]
    row += 'MultiROM - {:16e}'.format(multirom_.test_error(db_val))
    print(row)


# Plot the results of the multiroms
fig2, ax2 = plt.subplots(nrows=3, ncols=int(len(multiroms.keys())+1),
        figsize=(16, 8), sharex=True, sharey=True)
for i, param in enumerate(new_params):
    for j, model_name in enumerate(multiroms.keys()):
        ax2[i, j].tricontourf(data.triang, *multiroms[model_name].predict([param]))
        ax2[i, j].set_title(f'{model_name}')
    ax2[i, -1].tricontourf(data.triang, db_test.snapshots_matrix[i])
    ax2[i, -1].set_title(f'Original at mu={param}')
plt.show()
fig2.savefig(f'multirom_{rank}modes/multiroms_prediction_{rank}modes_ann.png')

fig3, ax3 = plt.subplots(nrows=2, ncols=int(len(multiroms.keys())),
        figsize=(16, 8), sharex=True, sharey=True)


# Plot the weights (test) associated to the ROMs with different regression models
for i, rom in enumerate(roms_dict.keys()):
    for j, model_name in enumerate(multiroms.keys()):
        weight = multiroms[model_name].weights_predict[rom][0, :]
        if len(multiroms.keys()) == 1:
            ax = ax3[i]
        else:
            ax = ax3[i, j]
        c = ax.tricontourf(data.triang, weight.flatten())
        ax.set_title(f'Weights {model_name}')
        fig3.colorbar(c, ax=ax)
plt.show()
fig3.savefig(f'multirom_{rank}modes/weights_{rank}modes_ann.png')

