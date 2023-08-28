Parallel workflows with Snakemake
---------------------------------

Many scientific problems involve complicated workflows with multiple interdependent steps.
If the workflow involves performing the same analysis on many different datasets we can 
use the inherent ("embarrassing") parallelism of the problem and perform these simultaneously.

Let us have a look at a toy example which many of us can hopefully relate to. 

.. demo:: Demo: The word-count project

   Head over to https://github.com/enccs/word-count-hpda and clone the repository:

   .. code-block:: console

      $ git clone https://github.com/ENCCS/word-count-hpda.git

   This project is about counting words in a given text and print out the 10 most common 
   words which can be used to test `Zipf's law <https://en.wikipedia.org/wiki/Zipf%27s_law>`__.
   The ``data`` directory contains 64 public domain books from `Project Gutenberg <https://www.gutenberg.org/>`__ 
   and source files under ``source`` can be used to count words:

   .. code-block:: console

      $ # count words in two books
      $ python source/wordcount.py data/pg10.txt processed_data/pg10.dat
      $ python source/wordcount.py data/pg65.txt processed_data/pg65.dat
      
      $ # print frequency of 10 most frequent words in both books to file
      $ python source/zipf_test.py 10 processed_data/pg10.dat processed_data/pg65.dat > results/results.csv
      
   This workflow is encoded in the ``Snakefile`` which can be used to run
   through all data files:

   .. code-block:: console

      $ # run workflow in serial
      $ snakemake -j 1      


   The workflow can be visualised in a directed-acyclic graph:

   .. code-block:: console

      $ # requires dot from Graphviz
      $ snakemake -j 1 --dag | dot -Tpng  > dag.png

   .. figure:: img/dag.png
      :align: center
      :scale: 80 %

   The workflow can be parallelized to utilize multiple cores:

   .. code-block:: console

      $ # first clear all output
      $ snakemake -j 1 --delete-all-output      
      $ # run in parallel on 4 processes
      $ snakemake -j 4

    For embarrassingly parallel work one can achieve significant speedup with parallel Snakemake execution.

The Snakefile describes the workflow in declarative style, i.e. we describe 
the dependencies but let Snakemake figure out the series of steps to produce 
results (targets). This is how the Snakefile looks:

.. code-block:: python

   # a list of all the books we are analyzing
   DATA = glob_wildcards('data/{book}.txt').book
   
   # the default rule
   rule all:
       input:
           'results/results.csv'
   
   # count words in one of our books
   # logfiles from each run are put in .log files"
   rule count_words:
       input:
           wc='source/wordcount.py',
           book='data/{file}.txt'
       output: 'processed_data/{file}.dat'
       log: 'processed_data/{file}.log'
       shell:
           '''
               python {input.wc} {input.book} {output} >> {log} 2>&1
           '''
   
   # generate results table
   rule zipf_test:
       input:
           zipf='source/zipf_test.py',
           books=expand('processed_data/{book}.dat', book=DATA)
       params:
           nwords = 10
       output: 'results/results.csv'
       shell:  'python {input.zipf} {params.nwords} {input.books} > {output}'


ipyparallel
-----------

`ipyparallel <https://ipyparallel.readthedocs.io/en/latest/>`__, also known as IPython Parallel, 
is yet another tool for parallel computing in Python. However, it's more than just parallel Python, 
it's parallel *IPython*, and this adds interactivity to parallel computing.

The architecture of ipyparallel for parallel and distributed computing abstracts out parallelism in a 
general way and this enables many different styles of parallelism, including:

- Single program, multiple data (SPMD) parallelism
- Multiple program, multiple data (MPMD) parallelism
- Message passing using MPI
- Task farming
- Data parallel
- Combinations of these approaches
- Custom user-defined approaches

This is similar to Dask which will be covered in a later episode. 

Let's explore how ipyparallel can be used together with MPI.  
The following code will initialize an IPython Cluster with 8 MPI engines in one of two ways:

- Inside a context manager to automatically manage starting and stopping engines.
- In a terminal and connect to it from a Jupyter notebook. 

After initializing the cluster, we create a "broadcast view" to the engines, and finally 
use the :meth:`apply_sync` function to run the :meth:`mpi_example` function on the engines:

.. tabs:: 

   .. tab:: Context manager

      Define function: 

      .. code-block:: python
      
         def mpi_example():
             from mpi4py import MPI
             comm = MPI.COMM_WORLD
             return f"Hello World from rank {comm.Get_rank()}. Total ranks={comm.Get_size()}"

      Start cluster in context manager:

      .. code-block:: python
      
         import ipyparallel as ipp
         # request an MPI cluster with 4 engines
         with ipp.Cluster(engines='mpi', n=4) as cluster:
            # get a broadcast_view on the cluster which is best suited for MPI style computation
            view = cluster.broadcast_view()
            # run the mpi_example function on all engines in parallel
            r = view.apply_sync(mpi_example)

         # Retrieve and print the result from the engines
         print("\n".join(r))   

   .. tab:: In terminal with ``ipcluster``

      Define function: 

      .. code-block:: python
      
         def mpi_example():
             from mpi4py import MPI
             comm = MPI.COMM_WORLD
             return f"Hello World from rank {comm.Get_rank()}. Total ranks={comm.Get_size()}"

      Start engines in terminal:

      .. code-block:: console
      
         $ # load module with MPI
         $ ml add foss/2021b
         $ ipcluster start -n 8 --engines=MPI

      Connect from a code cell in Jupyter:
      
      .. code-block:: python
      
         import ipyparallel as ipp
         cluster = ipp.Client()
         # print engine indices
         print(cluster.ids)
         view = cluster.broadcast_view()
         r = view.apply_sync(mpi_example)
         print("\n".join(r))


In an exercise below you can practice using ipyparallel for running an interactive MPI job in Jupyter 
for the word-count project.
