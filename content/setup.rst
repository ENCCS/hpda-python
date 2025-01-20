Installation and HPC access
===========================


This page contains instructions for installing the required dependencies on a local computer 
as well as instructions for logging into a EuroHPC system.


Local installation
------------------


If you already have a preferred way to manage Python versions and libraries, you can stick to that. If not, we recommend that you install Python3 and all libraries using `Miniforge <https://conda-forge.org/download/>`__, a free minimal installer for the package, dependency and environment manager `conda <https://docs.conda.io/en/latest/index.html>`__.

Please follow the installation instructions on https://conda-forge.org/download/ to install Miniforge.

Make sure that conda is correctly installed:

.. code-block:: console

   $ conda --version
   conda 24.11.2

With conda installed, install the required dependencies by running:

.. code-block:: console

   $ conda env create --yes -f https://raw.githubusercontent.com/ENCCS/hpda-python/main/content/env/environment.yml

This will create a new environment ``pyhpda`` which you need to activate by:

.. code-block:: console

   $ conda activate pyhpda

.. To use MPI4Py on your computer you need to install MPI libraries. With conda, these libraries are 
.. installed automatically when installing the mpi4py package:
..
.. .. code-block:: console
..
..    $ conda install -c conda-forge mpi4py

Ensure that the Python version is fairly recent:

.. code-block:: console

   $ python --version
   Python 3.12.8

Finally, open Jupyter-Lab in your browser:

.. code-block:: console

   $ jupyter-lab
   

LUMI
------------


Login to LUMI cluster
^^^^^^^^^^^^^^^^^^^^^

Follow practical instructions `HERE <https://enccs.se/tutorials/2024/02/log-in-to-lumi-cluster/>`_ to get your access to LUMI cluster.

- On `Step 5 <https://enccs.se/tutorials/2024/02/log-in-to-lumi-cluster/>`_, you can login to LUMI cluster through terminal.
- On `Step 6 <https://enccs.se/tutorials/2024/02/log-in-to-lumi-cluster/>`_, you can login to LUMI cluster from the web-interface.


Running jobs on LUMI cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to run an interactive job asking for 1 node, 1 GPU, and 1 hour:  

.. code-block:: console

   $ salloc -A project_465001310 -N 1 -t 1:00:00 -p standard-g --gpus-per-node=1
   $ srun <some-command>

Exit interactive allocation with ``exit``.

Interacive terminal session on compute node:

.. code-block:: console

   $ srun --account=project_465001310 --partition=standard-g --nodes=1 --cpus-per-task=1 --ntasks-per-node=1 --gpus-per-node=1 --time=1:00:00 --pty bash
   $ <some-command>


You can also submit your job with a batch script ``submit.sh``:

.. code-block:: bash

   #!/bin/bash -l
   #SBATCH --account=project_465001310
   #SBATCH --job-name=example-job
   #SBATCH --output=examplejob.o%j
   #SBATCH --error=examplejob.e%j
   #SBATCH --partition=standard-g
   #SBATCH --nodes=1
   #SBATCH --gpus-per-node=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --time=1:00:00

   srun <some_command> 


Some useful commands are listed below:

- Submit the job: ``sbatch submit.sh``
- Monitor your job: ``squeue --me``
- Kill job: ``scancel <JOB_ID>``
