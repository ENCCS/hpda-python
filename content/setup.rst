Setup
=====

Karolina
--------

Thanks to `IT4I <https://www.it4i.cz/en>`__, we will have an allocation on the Karolina supercomputer for the whole 
duration of the workshop. Here are instructions for accessing Karolina, setting up the Python environment and 
running jobs.

- instructions for getting an account
- login instructions

Software on Karolina is available through a module system. 
First load the Anaconda module to get access to the ``conda`` package manager:

.. code-block:: console

   $ ml add Anaconda3/2021.11

To be able to create conda environments in your home directory you need to initialize it. 
The following command adds the necessary configuration to your ``.bashrc`` file:

.. code-block:: console

   $ conda init bash

You now need to either log in to Karolina again or start a new shell session by typing ``bash``.

Now create a new environment with all required dependencies by:

.. code-block:: console

   $ conda env create -f https://raw.githubusercontent.com/ENCCS/HPDA-Python/main/content/env/environment.yml

The installation will take a few minutes.   

Now activate the environment by:

.. code-block:: console

   $ conda activate pyhpda

mpi4py
^^^^^^

To use MPI4Py you also need to load a module:

.. code-block:: console

   $ ml add mpi4py/3.1.1-gompi-2020b

Running jobs
^^^^^^^^^^^^

Resources can be allocated both through batch jobs (submitting a script to the scheduler)
and interactively. To allocate one interactive node for 1 hour on 1 node in the CPU partition 
and express queue:

.. code-block:: console

   $ qsub -A DD-22-28 -q qexp -l walltime=01:00:00 -I


Running Jupyter
^^^^^^^^^^^^^^^

The following procedure starts a Jupyter-Lab server on a compute node, creates an SSH tunnel from 
your local machine to the compute node, and then connects to the remote Jupyter-Lab server from your 
browser.

First make sure to:

- Allocate an interactive compute node for a sufficiently long time
- Switch to the pyhpda conda environment.

After allocating an interactive node your terminal session will be connected to that node.
Find out the name of your compute node. Your terminal prompt should show it but you can also run the 
``hostname`` command. Look only at the node name (e.g. ``cn012``) and disregard the ``.karolina.it4i.cz`` part.

Now start Jupyter-Lab on the compute node and specify both a port number (between 8000 and 9000) and the IP, which 
should be the name of the compute node. For example (replace port number and IP):

.. code-block:: console

   $ jupyter-lab --no-browser --port=8123 --ip=cn012

Now create an SSH tunnel **from a new terminal on your local machine** to the correct port and IP:

.. code-block:: console

   $ ssh -TN -f YourUsername@login2.karolina.it4i.cz -L localhost:8123:cn012:8123

Go back to the terminal running Jupyter-Lab on the compute node, and copy-paste the URL starting with 
``127.0.0.1`` which contains a long token into your local browser. If that does not work, try replacing 
``127.0.0.1`` with ``localhost``.

If everything is working as it should, you should now be able to create a new Jupyter notebook in your browser 
which is connected to a Karolina compute node and the ``pyhpda`` conda environment.

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
   # should give something like Python 3.9.7
   $ conda --version
   # should give something like conda 4.10.2

With conda installed, install the required dependencies by running:

.. code-block:: console

   $ conda env create -f https://raw.githubusercontent.com/ENCCS/HPDA-Python/main/content/env/environment.yml

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
   