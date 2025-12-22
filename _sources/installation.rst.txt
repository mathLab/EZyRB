Installation
============

Requirements
------------

EZyRB requires:

- Python >= 3.8
- numpy
- scipy
- matplotlib
- scikit-learn
- torch (for neural network-based methods)

Install via pip
---------------

The easiest way to install EZyRB is using pip:

.. code-block:: bash

    pip install ezyrb

This will automatically install all required dependencies.


Install from source
-------------------

To get the latest development version, clone the repository from GitHub:

.. code-block:: bash

    git clone https://github.com/mathLab/EZyRB
    cd EZyRB
    pip install .

For development purposes, you can install in editable mode:

.. code-block:: bash

    pip install -e .


Optional Dependencies
---------------------

For additional features, you may want to install:

- **vtk**: For reading/writing VTK files
- **GPy**: For advanced Gaussian Process Regression

Install them with:

.. code-block:: bash

    pip install vtk GPy


Verify Installation
-------------------

To verify that EZyRB is correctly installed, run:

.. code-block:: python

    import ezyrb
    print(ezyrb.__version__)
