Welcome to EZyRB's documentation!
===================================================

.. image:: _static/logo_EZyRB.png
    :height: 150px
    :width: 150 px
    :align: right

Easy Reduced Basis method.


Description
^^^^^^^^^^^^

EZyRB is a python library for the Model Order Reduction based on baricentric triangulation for the selection of the parameter points and on Proper Orthogonal Decomposition for the selection of the modes. It is ideally suited for actual industrial problems, since its structure can interact with several simulation software simply providing the output file of the simulations. Up to now, it handles files in the vtk and mat formats. It has been used for the model order reduction of problems solved with matlab and openFOAM.


Guide
^^^^^

.. toctree::
    :maxdepth: 3

    code
    contact
    contributing
    LICENSE


Tutorials
^^^^^^^^^^

We made some tutorial examples:

- `Tutorial 1 <tutorial1.html>`_ shows how to deal with a pressure field in vtk files (offline phase)
- `Tutorial 2 <tutorial2.html>`_ shows how to deal with a pressure field in vtk files (online phase)
- `Tutorial 3 <tutorial3.html>`_ shows how to deal with the GUI


Indices and tables
^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
