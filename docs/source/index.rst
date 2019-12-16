Welcome to EZyRB's documentation!
===================================================

.. image:: _static/logo_EZyRB.png
    :height: 150px
    :width: 150 px
    :align: right

Easy Reduced Basis method.


Description
^^^^^^^^^^^^

EZyRB is a python library for the Model Order Reduction based on baricentric triangulation for the selection of the parameter points and on Proper Orthogonal Decomposition for the selection of the modes. It is ideally suited for actual industrial problems, since its structure can interact with several simulation software simply providing the output file of the simulations. The software uses a POD interpolation approach in which the solutions are projected on the low dimensional space spanned by the POD modes (see "Bui-Thanh et al. - Proper orthogonal decomposition extensions for parametric applications in compressible aerodynamics" and "Chinesta et al. - Model Order Reduction: a survey"). The new solution is then obtained by interpolating the low rank solutions into the parametric space. This approach makes the package non intrusive with respect to the high fidelity solver actually used. This allows an easy integration into existing simulation pipelines, and it can deal with both vtk files and matlab files.

In the EZyRB package we implemented in Python the algorithms described above. We also provide tutorials that show all the characteristics of the software, from the offline part in which it is possible to construct the database of snapshots, to the online part for fast evaluations of the fields for new parameters. There are also modules to allow the consistency of all the solutions (often with different degrees of freedom) in order to process them.


Installation
--------------------
EZyRB requires numpy, scipy, matplotlib, and sphinx (for the documentation). They can be easily installed via pip. Moreover EZyRB depends on vtk. The code is compatible with Python 2.7. It can be installed directly from the source code.


The `official distribution <https://github.com/mathLab/EZyRB>`_ is on GitHub, and you can clone the repository using
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

   index
   code
   contact
   contributing
   LICENSE



Tutorials
^^^^^^^^^^

We made some tutorial examples:

- `Tutorial 1 <tutorial-1.html>`_ shows how to construct a simple reduced order model for a heat conduction problem.


Indices and tables
^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
