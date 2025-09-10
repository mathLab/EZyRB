
# Import the dataset from smithers
import os
from smithers.dataset import NavierStokesDataset
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
# In total, the dataset contains 500 parametric configurations in a space of 1639 degrees of freedom. In this case, we have just one parameter, which is the velocity (along $x$) we impose at the inlet.

# Import the necessary libraries and modules

from ezyrb import Database
from ezyrb import POD, AE
from ezyrb import RBF, GPR, ANN, KNeighborsRegressor
from ezyrb import ReducedOrderModel as ROM
from ezyrb import MultiReducedOrderModel as MultiROM
from ezyrb.plugin import Aggregation, DatabaseSplitter
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from ann_demo import ANN_Demo
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor
import warnings
warnings.filterwarnings("ignore", message="Ill-conditioned matrix ")

# Enable LaTeX for all text in plots
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

# Before starting with the reduced order model, we reduce the dimension
# of the dataset
field = 'vx'
rank = 2
coords = data.pts_coordinates.T
data.params = data.params[:300, :]
data.snapshots[field] = data.snapshots[field][:300, :]

# Create the database
db_all = Database(data.params, data.snapshots[field], coords)

# Define some reduction and approximation methods to test
reduction_methods = {
    'POD': POD('svd', rank=rank),
    'AE': AE([100, 10, rank], [rank, 10, 100], nn.Softplus(), nn.Softplus(), 3000, frequency_print=500)
}
approximation_methods = {
    'RBF': RBF(),
    'GPR': GPR(),
    'ANN': ANN([20, 20], nn.Tanh(), 50000, frequency_print=5000),
    'KNN': KNeighborsRegressor()
}

# Define a dictionary to store the ROMs
roms_dict = {}
db_splitter_plugin = DatabaseSplitter(train=180, validation=90, test=0,
                                            predict=30, seed=50)
# Train a ROM for each combination of reduction and approximation
for redname, redclass in reduction_methods.items():
    for approxname, approxclass in approximation_methods.items():
        rom = ROM(copy.deepcopy(db_all),
                  copy.deepcopy(redclass),
                  copy.deepcopy(approxclass),
                  plugins=[db_splitter_plugin])
        roms_dict[f'{redname}_{approxname}'] = rom
# Build a simple multiROM without aggregation and save it
multirom_noagg = MultiROM(roms_dict)
# Fit the multiROM
multirom_noagg.fit()
# Save the multiROM
multirom_noagg.save(f'multirom_{rank}modes/multirom_noaggregation.pkl')

# Get the dictionary of ROMs
roms_dict = multirom_noagg.roms

# Visualize the results of each ROM in the multiROM without aggregation on
# some new parameters
fig, ax = plt.subplots(nrows=3, ncols=int(len(roms_dict.keys()))+1,
        figsize=(16, 8), sharex=True, sharey=True)
# Define some new parameters to test
rom_one = list(multirom_noagg.roms.values())[0]
new_params = rom_one.predict_full_database.parameters_matrix[:3]
for i, param in enumerate(new_params):
    for j, rom in enumerate(roms_dict.values()):
        ax[i, j].tricontourf(data.triang, *rom.predict([param]))
        ax[i, j].set_title(f'{list(roms_dict.keys())[j]}')
    ax[i, -1].tricontourf(data.triang, rom.predict_full_database.snapshots_matrix[i])
    ax[i, -1].set_title(f'Original at mu={param}')
plt.show()
fig.savefig(f'multirom_{rank}modes/roms_prediction_{rank}modes.png', dpi=300)

# Define different multiROMs with different models of aggregation
aggr_models = {
    "KNN": KNNRegressor(n_neighbors=8),
    "ANN": ANN_Demo(copy.deepcopy(multirom_noagg), [10, 10, 10],[nn.Softplus(), nn.Softplus(), nn.Softplus(), nn.Softmax(dim=-1)],
                    stop_training=10000, lr=1e-3, frequency_print=5000, l2_regularization=1e-4)
}

multirom_KNN = MultiROM(roms_dict)
multirom_ANN = MultiROM(copy.deepcopy(roms_dict), plugins=[Aggregation(fit_function=aggr_models["ANN"])])
multiroms = {}

# Fit and save the different multiROMs with aggregation
# KNN
print("Fitting multiROM with KNN aggregation...")
sigma_KNN = multirom_KNN.optimize_sigma(rom_one.validation_full_database,
                                        aggr_models["KNN"], sigma_range=[1e-3, 1])
multirom_KNN.fit_weights(rom_one.validation_full_database, aggr_models["KNN"], sigma=sigma_KNN)
multirom_KNN.save(f'multirom_{rank}modes/multirom_KNN_aggregation.pkl')
multiroms["KNN"] = MultiROM.load(f'multirom_{rank}modes/multirom_KNN_aggregation.pkl')
# ANN
print("Fitting multiROM with ANN aggregation...")
multirom_ANN.fit()
multirom_ANN.save(f'multirom_{rank}modes/multirom_ANN_aggregation.pkl')
multiroms["ANN"] = MultiROM.load(f'multirom_{rank}modes/multirom_ANN_aggregation.pkl')

# Print the errors of the individual ROMs and of the multirom
db_validation = rom_one.validation_full_database
db_test = rom_one.predict_full_database
header = '{:10s}'.format('')
for name in approximation_methods:
    header += ' {:>16s}'.format(name)
print(header)
for redname, redclass in reduction_methods.items():
    row = '{:10s}'.format(redname)
    for approxname, approxclass in approximation_methods.items():
        rom = roms_dict[redname+'_'+approxname]
        row += ' {:16e}'.format(rom.test_error(db_test))
    print(row)
    print('-'*len(row))
for model_name in multiroms:
    row = '{:10s}'.format(model_name)
    multirom_ = multiroms[model_name]
    if model_name == "ANN":
        row += '- MultiROM {:16e}'.format(multirom_.test_error(db_test, gaussians=False))
    else:
        row += '- MultiROM {:16e}'.format(multirom_.test_error(db_test))
    print(row)
    
# Plot the results of the multiroms
fig2, ax2 = plt.subplots(nrows=3, ncols=int(len(multiroms.keys())+1),
        figsize=(16, 8), sharex=True, sharey=True)
for i, param in enumerate(new_params):
    for j, model_name in enumerate(multiroms.keys()):
        if model_name == "ANN":
            db_test_predicted = multiroms[model_name].predict(db_test, gaussians=False)
            ax2[i, j].tricontourf(data.triang, db_test_predicted.snapshots_matrix[i])
        else:
            db_test_predicted = multiroms[model_name].predict(db_test)
            ax2[i, j].tricontourf(data.triang, db_test_predicted.snapshots_matrix[i])
        ax2[i, j].set_title(f'{model_name}')
    ax2[i, -1].tricontourf(data.triang, multiroms[model_name].predict_full_database.snapshots_matrix[i])
    ax2[i, -1].set_title(f'Original at mu={param}')
plt.show()
fig2.savefig(f'multirom_{rank}modes/multiroms_prediction_{rank}modes_knnann.png')

# Plot the weights (test) associated to the ROMs with different regression models
fig3, ax3 = plt.subplots(nrows=8, ncols=int(len(multiroms.keys())),
        figsize=(16, 8), sharex=True, sharey=True)
for i, rom in enumerate(roms_dict.keys()):
    for j, model_name in enumerate(multiroms.keys()):
        if model_name == "ANN":
            weight = multiroms[model_name].weights_predict[rom][0, :]
        else:
            weight = multiroms[model_name].weights_test_database[rom][0, :]
        if len(multiroms.keys()) == 1:
            ax = ax3[i]
        else:
            ax = ax3[i, j]
        c = ax.tricontourf(data.triang, weight.flatten())
        ax.set_title(f'Weights {model_name} - {rom}')
        fig3.colorbar(c, ax=ax)
plt.show()
fig3.savefig(f'multirom_{rank}modes/weights_{rank}modes_knnann.png')
