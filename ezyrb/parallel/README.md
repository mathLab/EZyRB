
# EZyRB Parallel 
Adding parallelization capabilities to the classical [EZyRB](https://github.com/mathLab/EZyRB).

## Table of contents
* [Description](#description)
* [Features](#features)
* [Prerequisites](#prerequisites)
* [How to use?](#how-to-use)
* [Authors and contributors](#authors-and-contributors)

## Description

This work is developed within the [eFlows4HPC](https://eflows4hpc.eu/) project to investigate the abilities of the completely data-driven non-intrusive Model Order Reduction, in particular employing nonlinear reduction techniques such as autoencoders instead of the linear (SVD-based) approach to generate a low-dimensional representation of the solution manifold. 

Since the size of the input data obtained from the full order simulations can be very large, resulting in a huge set of weights and biases for the neural network, we are aiming to explore the capabilities of the non-linear reduction for various types of problems within the HPC framework taking the advantage of data parallelism techniques.

## Features
### Parallel execution
In this submodule of EZyRB, we utilised the   [PyCOMPSs](https://compss-doc.readthedocs.io/en/stable/index.html)  package to convert some methods of EZyRB to tasks that can be executed simultaneously in different worker nodes.  This can enhance the process of multiple predictions or error calculations via [`ezyrb.ReducedOrderModel.kfold_cv_error()`](https://mathlab.github.io/EZyRB/reducedordermodel.html#) and [`ezyrb.ReducedOrderModel.loo_error()`](https://mathlab.github.io/EZyRB/reducedordermodel.html#) methods, where the model has to be executed multiple times for different parameters. 

The picture below shows the simultaneous execution of 12 different setups, each performing 3 predictions.

![Alt text](https://github.com/karimyehia92/EZyRB/blob/parallel_ezyrb/ezyrb/parallel/examples/All1.png?raw=true "Simultaneous execution of 12 different setups, each performing 3 predictions")

### Non-linear reduction with data parallelism
In the sequential version of EZyRB, the non-linear reduction is performed by the `ae.py` module  which relies on [PyTorch](https://pytorch.org/). In this version we added a new module `ae_EDDL.py` which is built using the European Distributed Deep Learning library [PyEDDL](https://github.com/deephealthproject/pyeddl).  The `ae_EDDL.py` module supports data parallelism and can run on CPUs, GPUs and FPGAs by easialy changing the type and number of the computing services you want to use during the execution.

## How to use?
For examples of how to use the parallel submodule see the following notebooks:
1. [Parallel execution](https://github.com/karimyehia92/EZyRB/blob/parallel_ezyrb/ezyrb/parallel/examples/parallel_execution.ipynb)
2. [Non-linear reduction with data parallelism](https://github.com/karimyehia92/EZyRB/blob/parallel_ezyrb/ezyrb/parallel/examples/autoencoder_with_data_parallelism.ipynb)

## Prerequisites
Besides the regular installation of EZyRB you need to have PyCOMPSs and PyEDDL installed.
1. [Dpendencies and installation of EZyRB](https://github.com/mathLab/EZyRB#dependencies-and-installation)
2. [PyCOMPSs installation](https://compss-doc.readthedocs.io/en/stable/Sections/01_Installation/03_Pip.html)
3. [PyEDDL installation](https://deephealthproject.github.io/pyeddl/installation.html)


## Authors and contributors

This work is developed in [SISSA mathLab](https://mathlab.sissa.it/) within the framework of the [eFlows4HPC](https://eflows4hpc.eu/) project, in cooperation with 16 [partners](https://eflows4hpc.eu/partners/) from seven different countries.
![Alt text](https://github.com/karimyehia92/EZyRB/blob/parallel_ezyrb/ezyrb/parallel/examples/pictures/logos.png?raw=true "Logos")
