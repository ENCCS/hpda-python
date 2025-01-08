Installation and HPC access
===========================

This page contains instructions for installing the required dependencies on a local computer 
as well as instructions for logging in and using two EuroHPC systems.

Local installation
------------------

If you already have a preferred way to manage Python versions and 
libraries, you can stick to that. If not, we recommend that you 
install Python3 and all libraries using 
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`__, 
a free minimal installer for the package, dependency and environment manager 
`conda <https://docs.conda.io/en/latest/index.html>`__.

Please follow the installation instructions on 
https://docs.conda.io/en/latest/miniconda.html to install Miniconda3.

Make sure that both Python and conda are correctly installed:

.. code-block:: console

   $ python --version
   $ # should give something like Python 3.9.7
   $ conda --version
   $ # should give something like conda 4.10.2

With conda installed, install the required dependencies by running:

.. code-block:: console

   $ conda env create -f https://raw.githubusercontent.com/ENCCS/hpda-python/main/content/env/environment.yml

This will create a new environment ``pyhpda`` which you need to activate by:

.. code-block:: console

   $ conda activate pyhpda

To use MPI4Py on your computer you need to install MPI libraries. With conda, these libraries are 
installed automatically when installing the mpi4py package:

.. code-block:: console

   $ conda install -c conda-forge mpi4py

Finally, open Jupyter-Lab in your browser:

.. code-block:: console

   $ jupyter-lab
   

