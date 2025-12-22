EZyRB - Easy Reduced Basis
===================================================

.. image:: _static/logo_EZyRB.png
    :height: 150px
    :width: 150 px
    :align: right

**Easy Reduced Basis method for Model Order Reduction**

EZyRB is a Python library for **Model Order Reduction** based on various reduction and approximation techniques. It provides a flexible framework for creating fast surrogate models from high-fidelity simulations.


Key Features
^^^^^^^^^^^^

- **Multiple Reduction Methods**: POD, Autoencoders (AE), POD-AE
- **Flexible Approximation**: RBF, Linear, GPR, ANN, K-Neighbors, and more
- **Non-Intrusive Approach**: Works with any simulation output format
- **Plugin System**: Extensible architecture for preprocessing and postprocessing
- **Easy Integration**: Simple API for building reduced order models
- **Database Management**: Built-in tools for handling parameter-snapshot pairs


User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   Installation <installation>
   Quick Start <quickstart>
   API Documentation <code>
   Tutorials <tutorials>


Developer Info
--------------

.. toctree::
   :maxdepth: 1
   :caption: Developer Info

   contributing
   contact
   LICENSE


Indices and tables
^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
