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

import numpy as np
import torch
import torch.nn as nn

import matplotlib.tri as mtri
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="Ill-conditioned matrix ")
get_ipython().run_line_magic('matplotlib', 'inline')


# Before starting with the reduced order model, we visualize some of the snapshots in our dataset.

# In[4]:


fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 8), sharey=True, sharex=True)
ax = ax.flatten()
for i in range(9):
    ax[i].tricontourf(data.triang, data.snapshots['vx'][i], levels=16)
    ax[i].set_title('Original snapshot at time = {}'.format(*data.params[i].round(2)))


# In this step, we perform the model order reduction to obtain a reduced space from the full order space. We refer to [Tutorial 1](https://github.com/mathLab/EZyRB/blob/master/tutorials/tutorial-1.ipynb) for the description of the basic workflow, here we just quickly describe the steps implemented in the next cell.
# 
# We start by passing the matrices of the parameters and snapshots to the `Database()` class. It must be said that at this time we just create the ROM for the `vx` field. We create also the `POD` and `RBF` object to have a benchmark ROM.

# In[5]:


db = Database(data.params, data.snapshots['vx'])
rom = ROM(db, POD(), RBF())
rom.fit();


# Three lines for a data-driven reduced order model, not bad!
# 
# Just to have a visual check that everything is going well, we plot the approximation for new parameters in the range $[1, 80]$.

# In[6]:


new_params = np.random.uniform(size=(2))*79.+1.

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 3))
for i, param in enumerate(new_params):
    ax[i].tricontourf(data.triang, rom.predict([param]))
    ax[i].set_title('Predicted snapshots at time = {}'.format(param))


# We are now calculating the approximation error to see how close is our reduced solution to the full-order solution/simulation using the **k-fold Cross-Validation** strategy by passing the number of splits to the `ReducedOrderModel.kfold_cv_error(n_splits)` method, which operates as follows: 
# 
# 1. Split the dataset (parameters/snapshots) into $k$-number of groups/folds.
# 2. Use $k-1$ groups to calculate the reduced space and leave one group for testing.
# 3. Use the approximation/interpolation method to predict each snapshot in the testing group.
# 4. Calculate the error for each snapshot in the testing group by taking the difference between the predicted and the original snapshot. 
# 5. Average the errors for predicting snapshots of the testing group/fold.
# 6. Repeat this procedure using different groups for testing and the remaining $k-1$ groups to calculate the reduced space.
# 7. In the end, we will have $k$-number errors for predicting each group/fold that we can average them to have one value for the error.

# In[7]:


errors = rom.kfold_cv_error(n_splits = 5)
print('Average error for each fold:')
for e in errors:
    print('  ',e)
print('\nAverage error = {}'.format(errors.mean()))


# Another strategy for calculating the approximation error is called **leave-one-out** by using the  `ReducedOrderModel.loo_error()` method, which is similar to setting the number of folds equal to the number of snapshots (eg. in this case setting `n_splits` = 500) and it operates as follows:
# 1. Combine all the snapshots except one.
# 2. Calculate the reduced space.
# 3. Use the approximation/interpolation method to predict the removed snapshot.
# 4. Calculate the error by taking the difference between the predicted snapshot and the original removed one.
# 5. The error vector is obtained by repeating this procedure for each snapshot in the database.
# 
# It is worth mentioning that it consumes more time because we have 500 snapshots and the algorithm will perform space order reduction and calculate the approximation error 500 times. For this reason, we commented the next line of code, in order to limit the computational effort needed to run this tutorial. Uncomment it only if you are a really brave person!

# In[8]:


# errors = rom.loo_error()


# ### Comparison between different methods
# 
# One of the advantages of the data-driven reduced order modeling is the modular nature of the method. Practically speaking, we need
# - a method for reducing the dimensionality of input snapshots;
# - a method for approximate the solution manifold;
# 
# allowing in principle to a large variety of combinations.
# 
# The list of implemented **reduction methods** in EZyRB contains: 
# - `POD`: *proper orthogonal decomposition*
# - `AE`: *autoencoder*
# 
# while the list of implemented **approximation methods** contains:
# - `RBF`: *radial basis function interpolation*
# - `GPR`: *gaussian process regression*
# - `KNeighborsRegressor`: *k-neighbors regression*
# - `RadiusNeighborsRegressor`: *radius neighbors regression*
# - `Linear`: *multidimensional linear interpolation*
# 
# Moreover, new state-of-the-art methods will arrive, so we invite to read the [documentation](https://mathlab.github.io/EZyRB/) for the complete list of all the possibilities!
# 
# In the next cell, we create two dictionaries with the objects, such that we can easily test everything with simple `for` cycles. **WARNING** since several methods require the solution of an optimization problem (eg. GPR, ANN, AE), the cell may require some minutes to been run.

# In[9]:


reductions = {
    'POD': POD('svd',rank=10),
    'AE': AE([200, 100, 10], [10, 100, 200], nn.Tanh(), nn.Tanh(), 1000),
}

approximations = {
#    'Linear': Linear(),
    'RBF': RBF(),
    'GPR': GPR(),
    'KNeighbors': KNeighborsRegressor(),
    'RadiusNeighbors':  RadiusNeighborsRegressor(),
    'ANN': ANN([20, 20], nn.Tanh(), 1000),
}

header = '{:10s}'.format('')
for name in approximations:
    header += ' {:>15s}'.format(name)

print(header)
for redname, redclass in reductions.items():
    row = '{:10s}'.format(redname)    
    for approxname, approxclass in approximations.items():
        rom = ROM(db, redclass, approxclass)
        rom.fit()
        row += ' {:15e}'.format(rom.kfold_cv_error(n_splits=5).mean())
    
    print(row)


# In a very compact way, we tested several frameworks - like POD-RBF, POD-GPR, POD-NN -, showing the accuracy reached by any of them.
# 
# We can also note that the frameworks that involves neural networks (`AE` and `ANN`) show a very poor precision. This is due to the fact of the limited number of epoch we impose in the learning procedure. You can try to increase the number of epochs as we shown in next cell in order to obtain better results, a the cost of a longer taining phase.

# In[10]:


reductions['AE'] = AE([100, 10], [10, 100], nn.ReLU(), nn.ReLU(), 30000)
approximations['ANN'] = ANN([50, 10], nn.ReLU(), 30000)

