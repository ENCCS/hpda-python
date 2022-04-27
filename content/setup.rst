Setup
=====

This lesson requires Python3 and a number of Python libraries:

.. code-block:: bash

   numpy
   scipy
   pandas
   matplotlib
   dask
   ipython
   ipyparallel
   mpi4py
   xarray


MeluXina
--------

- instructions for getting an account
- login instructions

Software on MeluXina is available through a module system. The modules are however 
only available on compute nodes, so to load modules you first have to allocate resources  
through the SLURM job scheduler. To allocate resources you will need the project ID, which 
you can see by typing:

.. code-block:: bash

   sacctmgr show user $USER withassoc format=user,account,defaultaccount

Resources can be allocated both through batch jobs (submitting a script to the scheduler)
and interactively. To allocate one interactive node for 1 hour on 1 node in the CPU partition:

.. code-block:: bash

   salloc -A pNNNNNN -t 01:00:00 -q dev --res cpudev -p cpu -N 1

Once your interactive allocation starts, the prompt will change and you will be on a compute 
node. To see available modules you can now type:

.. code-block:: bash

   module avail




Local installation
------------------

>>>>>>> Stashed changes
If you already have a preferred way to manage Python versions and 
libraries, you can stick to that. If not, we recommend that you 
install Python3 and all libraries using 
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`__, 
a free minimal installer for the package, dependency and environment manager 
`conda <https://docs.conda.io/en/latest/index.html>`__.

Please follow the installation instructions on 
https://docs.conda.io/en/latest/miniconda.html to install Miniconda3.

Make sure that both Python and conda are correctly installed:

.. code-block:: bash

   python --version
   # should give something like Python 3.9.7
   conda --version
   # should give something like conda 4.10.2

With conda installed, create a new conda environment named `hpda` by 

.. code-block:: bash

   conda create -n hpda python   

Then copy the following into a new file `requirements.txt`:

.. code-block:: bash

   WRITEME

and install the required dependencies by running:

.. code-block:: bash

   pip install -r requirements.txt --user
