Quick Start
===========

This guide will help you get started with EZyRB in just a few minutes.


Basic Workflow
--------------

The typical workflow for using EZyRB consists of three main steps:

1. **Create a Database**: Collect your parameter-snapshot pairs
2. **Build a ROM**: Choose reduction and approximation methods
3. **Make Predictions**: Evaluate the ROM at new parameter values


Minimal Example
---------------

Here's a complete example to get you started:

.. code-block:: python

    import numpy as np
    from ezyrb import POD, RBF, Database, ReducedOrderModel
    
    # Step 1: Create a database
    params = np.array([[1.0], [2.0], [3.0], [4.0]])
    snapshots = np.random.rand(4, 100)  # 4 snapshots of size 100
    db = Database(params, snapshots)
    
    # Step 2: Build a ROM
    pod = POD(rank=5)  # Use 5 POD modes
    rbf = RBF()        # Radial Basis Function interpolation
    rom = ReducedOrderModel(db, pod, rbf)
    rom.fit()
    
    # Step 3: Predict for new parameters
    new_param = np.array([[2.5]])
    prediction = rom.predict(new_param)
    print(prediction.snapshots_matrix.shape)  # (1, 100)


Understanding the Components
-----------------------------

Database
^^^^^^^^

The ``Database`` class stores parameter-snapshot pairs:

.. code-block:: python

    from ezyrb import Database, Parameter, Snapshot
    
    # Simple creation
    db = Database(parameters, snapshots)
    
    # Or add pairs individually
    db = Database()
    db.add(Parameter([1.0, 2.0]), Snapshot(values))


Reduction Methods
^^^^^^^^^^^^^^^^^

Reduce the dimensionality of your snapshots:

.. code-block:: python

    from ezyrb import POD, AE
    
    # Proper Orthogonal Decomposition
    pod = POD(rank=10)
    
    # Autoencoder
    import torch
    ae = AE([100, 50, 10], [10, 50, 100], 
            torch.nn.Tanh(), torch.nn.Tanh(), 1000)


Approximation Methods
^^^^^^^^^^^^^^^^^^^^^

Interpolate in the reduced space:

.. code-block:: python

    from ezyrb import RBF, GPR, ANN, Linear
    
    # Radial Basis Functions
    rbf = RBF()
    
    # Gaussian Process Regression
    gpr = GPR()
    
    # Artificial Neural Network
    import torch
    ann = ANN([10, 20, 10], torch.nn.Tanh(), 1000)
    
    # Linear interpolation
    linear = Linear()


Next Steps
----------

- Check out the :doc:`tutorials` for detailed examples
- Explore the :doc:`code` for complete API reference
- Learn about :doc:`plugin` system for advanced customization
