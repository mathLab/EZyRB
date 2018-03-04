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


Installation
--------------------
EZyRB requires numpy, scipy, matplotlib, and sphinx (for the documentation). They can be easily installed via pip. Moreover EZyRB depends on vtk. The code is compatible with Python 2.7. It can be installed directly from the source code.


The official distribution is on GitHub, and you can clone the repository using
::

    git clone https://github.com/mathLab/EZyRB

To install the package just type:
::

    python setup.py install

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

::

    python setup.py install --record installed_files.txt
    cat installed_files.txt | xargs rm -rf




Developer's Guide
--------------------

.. toctree::
   :maxdepth: 1

   code
   contact
   contributing
   LICENSE



Tutorials
^^^^^^^^^^

We made some tutorial examples:

- `Tutorial 1 <tutorial1.html>`_ shows how to deal with a pressure field in vtk files (offline phase)
- `Tutorial 2 <tutorial2.html>`_ shows how to deal with a pressure field in vtk files (online phase)
- `Tutorial 3 <tutorial3.html>`_ shows how to interpolate the solution on a new mesh.


Indices and tables
^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
