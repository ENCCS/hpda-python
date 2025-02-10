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

Karolina
--------

Thanks to `IT4I <https://www.it4i.cz/en>`__, we will have an allocation on the Karolina supercomputer for the whole 
duration of the workshop. Here are instructions for accessing Karolina, setting up the Python environment and 
running jobs.

- instructions for getting an account
- login instructions

Software on Karolina is available through a module system. 
First load the Anaconda module to get access to the ``conda`` package manager:

.. code-block:: bash

   ml add Anaconda3/2021.11

To be able to create conda environments in your home directory you need to initialize it. 
The following command adds the necessary configuration to your ``.bashrc`` file:

.. code-block:: bash

   conda init bash

You now need to either log in to Karolina again or start a new shell session by typing ``bash``.

Now download the environment.yml file:

.. code-block:: bash

   wget FIXME (add env file to lesson repo)

and create a new environment with all required dependencies by:

.. code-block:: bash

   conda env create -f environment.yml

The installation will take a few minutes.   

Now activate the environment by:

.. code-block:: bash

   conda activate hpda


Running jobs
^^^^^^^^^^^^

Resources can be allocated both through batch jobs (submitting a script to the scheduler)
and interactively. To allocate one interactive node for 1 hour on 1 node in the CPU partition 
and express queue:

.. code-block:: bash

   qsub -A DD-22-28 -q qexp -l walltime=01:00:00 -I


Running Jupyter
^^^^^^^^^^^^^^^

The following procedure starts a Jupyter-Lab server on a compute node, creates an SSH tunnel from 
your local machine to the compute node, and then connects to the remote Jupyter-Lab server from your 
browser.

First make sure to:

- Allocate an interactive compute node for a sufficiently long time
- Switch to the hpda conda environment.

After allocating an interactive node your terminal session will be connected to that node.
Find out the name of your compute node. Your terminal prompt should show it but you can also run the 
``hostname`` command. Look only at the node name (e.g. ``cn012``) and disregard the ``.karolina.it4i.cz`` part.

Now start Jupyter-Lab on the compute node and specify both a port number (between 8000 and 9000) and the IP, which 
should be the name of the compute node. For example (replace port number and IP):

.. code-block:: bash

   jupyter-lab --no-browser --port=8123 --ip=cn012

Now create an SSH tunnel **from a new terminal on your local machine** to the correct port and IP:

.. code-block:: bash

   ssh -TN -f YourUsername@login2.karolina.it4i.cz -L localhost:8123:cn012:8123

Go back to the terminal running Jupyter-Lab on the compute node, and copy-paste the URL starting with 
``127.0.0.1`` which contains a long token into your local browser. If that does not work, try replacing 
``127.0.0.1`` with ``localhost``.

If everything is working as it should, you should now be able to create a new Jupyter notebook in your browser 
which is connected to a Karolina compute node and the ``hpda`` conda environment.
=======
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

With conda installed, download the environment.yml file above and install the required dependencies by running:

.. code-block:: bash

   conda env create -f environment.yml

This will create a new environment ``hpda`` which you need to activate by:

.. code-block:: bash

   conda activate hpda

Now open a Jupyter-Lab instance in your browser:

.. code-block:: bash

   jupyter-lab
   