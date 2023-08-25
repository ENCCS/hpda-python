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
   

EuroHPC systems
---------------

Here are instructions for accessing the EuroHPC system, setting up the Python environment 
and running jobs. Please follow the instructions for the HPC system that will be used 
during the workshop that you are attending.

.. tabs::

   .. group-tab:: Vega

      Thanks to `IZUM <https://www.izum.si/en/hpc-en/>`__ in Slovenia we will have an allocation on the 
      petascale `Vega <https://en-vegadocs.vega.izum.si/>`__ EuroHPC system for the duration of the workshop.
      The sustained and peak performance of Vega is 6.9 petaflops and 10.1 petaflops, respectively. 

      **Architecture**:

      Vega has both `GPU and CPU partititions <https://en-vegadocs.vega.izum.si/architecture/>`__:

      - CPU partition: Each node has two AMD Epyc 7H12 CPUs, each with 64 cores. 768 nodes with 256 GB, 
        192 nodes with 1 TB of RAM DDR4-3200, local 1.92 TB M.2 SSD.
      - GPU partition: Each node has 4 GPUs NVidia A100 with 40 GB HBMI2 and two AMD Epyc 7H12 CPUs.
        In total 60 nodes with 512 GB of RAM DDR4-3200, local 1.92 TB M.2 SSD

   .. group-tab:: Karolina

      Thanks to `IT4I <https://www.it4i.cz/en>`__ in the Czech Republic we will have an allocation 
      on the petascale `Karolina supercomputer <https://www.it4i.cz/en/infrastructure/karolina>`__ 
      for the duration of the workshop. The peak performance of Karolina is 15.7 petaflops.

      **Architecture**:

      - 720x 2x AMD 7H12, 64 cores, 2,6 GHz, 92,160 cores in total
      - 72x 2x AMD 7763, 64 cores, 2,45 GHz, 9,216 cores in total
      - 72x 8x NVIDIA A100 GPU, 576 GPU in total
      - 32x Intel Xeon-SC 8628, 24 cores, 2,9 GHz, 768 cores in total
      - 36x 2x AMD 7H12, 64 cores, 2,6 GHz, 4,608 cores in total
      - 2x 2x AMD 7452, 32 cores, 2,35 GHz, 128 cores in total

Software on the cluster is available through a module system. 
First load the Anaconda module to get access to the ``conda`` package manager:

.. tabs:: 

   .. group-tab:: Vega

      .. code-block:: console
      
         $ #check available Anaconda modules:
         $ ml av Anaconda3
         $ ml add Anaconda3/2020.11

   .. group-tab:: Karolina

      .. code-block:: console
      
         $ #check available Anaconda modules:
         $ ml av Anaconda3
         $ ml add Anaconda3/2021.11


To be able to create conda environments in your home directory you need to initialize it. 
The following command adds the necessary configuration to your ``.bashrc`` file:

.. code-block:: console

   $ conda init bash

You now need to either log in to the cluster again or start a new shell session by typing ``bash``:

.. code-block:: console

   $ bash

Now, either create a new environment with all required dependencies or activate 
a pre-existing environment created in a directory you have access to:

.. tabs:: 

   .. tab:: Create new environment in $HOME

      .. code-block:: console
      
         $ conda env create -f https://raw.githubusercontent.com/ENCCS/HPDA-Python/main/content/env/environment.yml

      The installation can take several minutes. 
      Now activate the environment by:

      .. code-block:: console
      
         $ conda activate pyhpda

   .. tab:: Activate existing environment

      .. code-block:: console

         $ conda activate /path/to/envdir/


mpi4py
^^^^^^

Additional steps are required to use mpi4py since the Python package needs to be 
linked with the system's MPI libraries.

.. tabs:: 

   .. group-tab:: Vega

      To use mpi4py you need to load a module which contains MPI libraries and then install ``mpi4py``
      using ``pip``:

      .. code-block:: console
      
         $ ml add foss/2020b
         $ CC=gcc MPICC=mpicc python3 -m pip install mpi4py --no-binary=mpi4py

   .. group-tab:: Karolina

      To use mpi4py you only need to load a module:

      .. code-block:: console
      
         $ ml add mpi4py/3.1.1-gompi-2020b      

Running jobs
^^^^^^^^^^^^

Resources can be allocated both through batch jobs (submitting a script to the scheduler)
and interactively. You will need to provide a project ID when asking for an allocation.
To find out what projects you belong to on the cluster, type:

.. code-block:: console

   $ sacctmgr -p show associations user=$USER

The second column of the output contains the project ID.

.. tabs::

   .. group-tab:: Vega

      Vega uses the SLURM scheduler. 
      Use the following command to allocate one interactive node with 8 cores for 1 hour 
      in the CPU partition. If there is a reservation on the cluster for the workshop, 
      add ``--reservation=RESERVATIONNAME`` to the command.

      .. code-block:: console
      
         $ salloc -N 1 --ntasks-per-node=8 --ntasks-per-core=1 -A <PROJECT-ID> --partition=cpu  -t 01:00:00

      To instead book a GPU node, type (again adding reservation flag if relevant):

      .. code-block:: console
      
         $ salloc -N 1 --ntasks-per-node=1 --ntasks-per-core=1 -A <PROJECT-ID> --partition=gpu --gres=gpu:1 --cpus-per-task 1 -t 01:00:00

   .. group-tab:: Karolina 

      Karolina uses the PBS scheduler.
      To allocate one interactive node for 
      1 hour on 1 node in the CPU partition and express queue:

      .. code-block:: console
      
         $ qsub -A DD-22-28 -q qexp -l walltime=01:00:00 -I

Running Jupyter
^^^^^^^^^^^^^^^

The following procedure starts a Jupyter-Lab server on a compute node, creates an SSH tunnel from 
your local machine to the compute node, and then connects to the remote Jupyter-Lab server from your 
browser.

First make sure to follow the above instructions to:

- Allocate an interactive compute node for a sufficiently long time
- Switch to the pyhpda conda environment.

After allocating an interactive node you will see the name of the node in the output. 

.. tabs:: 

   .. group-tab:: Vega

      After allocating an interactive node you will see the name of the node in the 
      output, e.g. ``salloc: Nodes cn0709 are ready for job``.

      You now need to ssh to that node, switch to the pyhpda 
      conda environment, and start the Jupyter-Lab server on a particular port 
      (choose one between 8000 and 9000) 
      and IP address (the name of the compute node). Also load a module containing 
      OpenMPI to have access to MPI inside Jupyter:

      .. code-block:: console
      
         $ ssh cn0709
         $ conda activate pyhpda
         $ ml add foss/2021b
         $ jupyter-lab --no-browser --port=8123 --ip=cn0709

   .. group-tab:: Karolina

      After allocating an interactive node your terminal session will be connected to that node. 
      Find out the name of your compute node. Your terminal prompt should show it but you can 
      also run the hostname command. Look only at the node name (e.g. cn012) and disregard 
      the ``.karolina.it4i.cz`` part.    

      Now start the Jupyter-Lab server on a particular port 
      (choose one between 8000 and 9000) 
      and IP address (the name of the compute node):

      .. code-block:: console

         $ jupyter-lab --no-browser --port=8123 --ip=cn012
  

Now create an SSH tunnel **from a new terminal on your local machine** to the correct 
port and IP:

.. tabs:: 

   .. group-tab:: Vega

      .. code-block:: console
      
         $ ssh -TN -f YourUsername@login.vega.izum.si -L localhost:8123:cn0709:8123 -L localhost:8787:cn0709:8787

   .. group-tab:: Karolina

      .. code-block:: console

         $ ssh -TN -f YourUsername@login2.karolina.it4i.cz -L localhost:8123:cn012:8123

Go back to the terminal running Jupyter-Lab on the compute node, and copy-paste the URL 
starting with ``127.0.0.1`` which contains a long token into your local browser. 
If that does not work, try replacing ``127.0.0.1`` with ``localhost``.

If everything is working as it should, you should now be able to create a new Jupyter notebook in your browser 
which is connected to the compute node and the ``pyhpda`` conda environment.

