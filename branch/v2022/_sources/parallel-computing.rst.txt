.. _parallel-computing:

Parallel computing
==================

.. questions::

   - What is the Global Interpreter Lock in Python?
   - How can Python code be parallelised?

.. objectives::

   - Become familiar with different types of parallelism 
   - Learn the basics of parallel workflows, multiprocessing and distributed memory parallelism

.. instructor-note::

   - 40 min teaching/type-along
   - 40 min exercises


The performance of a single CPU core has stagnated over the last ten years
and most of the speed-up in modern CPUs is coming from using multiple
CPU cores, i.e. parallel processing. Parallel processing is normally based
either on multiple threads or multiple processes. 

There are three main models of parallel computing:

- **"Embarrassingly" parallel:** the code does not need to synchronize/communicate
  with other instances, and you can run
  multiple instances of the code separately, and combine the results
  later.  If you can do this, great!  

- **Shared memory parallelism (multithreading):** 
 
  - Parallel threads do separate work and communicate via the same memory and write to shared variables.
  - Multiple threads in a single Python program cannot execute at the same time (see GIL below)
  - Running multiple threads in Python is *only effective for certain I/O-bound tasks*
  - External libraries in other languages (e.g. C) which are called from Python can still use multithreading

- **Distributed memory parallelism (multiprocessing):** Different processes manage their own memory segments and 
  share data by communicating (passing messages) as needed.

  - A process can contain one or more threads
  - Two processes can run on different CPU cores and different computers
  - Processes have more overhead than threads (creating and destroying processes takes more time)
  - Running multiple processes is *only effective for CPU-bound tasks*

In the next episode we will look at `Dask <https://dask.org/>`__, an array model extension and task scheduler, 
which combines multiprocessing with (embarrassingly) parallel workflows and "lazy" execution.

In the Python world, it is common to see the word `concurrency` denoting any type of simultaneous 
processing, including *threads*, *tasks* and *processes*.

.. warning::

   Parallel programming requires that we adopt a different mental model compared to serial programming. 
   Many things can go wrong and one can get unexpected results or difficult-to-debug 
   problems. It is important to understand the possible pitfalls before embarking 
   on code parallelisation. For an entertaining take on this, see 
   `Raymond Hettinger's PyCon2016 presentation <https://www.youtube.com/watch?v=Bv25Dwe84g0>`__.

The Global Interpreter Lock
---------------------------

The designers of the Python language made the choice
that **only one thread in a process can run actual Python code**
by using the so-called **global interpreter lock (GIL)**.
This means that approaches that may work in other languages (C, C++, Fortran),
may not work in Python without being a bit careful.
At first glance, this is bad for parallelism.  *But it's not all bad!:*

- External libraries (NumPy, SciPy, Pandas, etc), written in C or other
  languages, can release the lock and run multi-threaded.  
- Most input/output releases the GIL, and input/output is slow.
- There are several Python libraries that side-step the GIL, e.g. by using 
  *subprocesses* instead of threads.


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


Multithreading
--------------

Due to the GIL only one thread can execute Python code at once, and this makes 
threading rather useless for *compute-bound* problems in pure Puthon. 
However, multithreading is still relevant in two situations:

- External libraries written in non-Python languages can take advantage of multithreading 
- Multithreading can be useful for running *multiple I/O-bound tasks simultaneously*.

Multithreaded libraries
^^^^^^^^^^^^^^^^^^^^^^^

NumPy and SciPy are built on external libraries such as LAPACK, FFTW append BLAS, 
which provide optimized routines for linear algebra, Fourier transforms etc.
These libraries are written in C, C++ or Fortran and are thus not limited 
by the GIL, so they typically support actual multihreading during the execution.
It might be a good idea to use multiple threads during calculations 
like matrix operations or frequency analysis.

Depending on configuration, NumPy will often use multiple threads by default, 
but we can use the environment variable ``OMP_NUM_THREADS`` to set the number 
of threads manually:

.. code-block:: console

   $ export OMP_NUM_THREADS=<N>

After setting this environment variable we continue as usual 
and multithreading will be turned on.

.. demo:: Demo: Multithreading NumPy 

   Here is an example which does a symmetrical matrix inversion of size 4000 by 4000.
   To run it, we can save it in a file named `omp_test.py` or download from :download:`here <example/omp_test.py>`.

   .. literalinclude:: example/omp_test.py
      :language: python

   Let us test it with 1 and 4 threads:

   .. code-block:: console

      $ export OMP_NUM_THREADS=1
      $ python omp_test.py

      $ export OMP_NUM_THREADS=4
      $ python omp_test.py

Multithreaded I/O
^^^^^^^^^^^^^^^^^

This is how an I/O-bound application might look:

.. figure:: img/IOBound.png
   :align: center
   :scale: 40 %

   From https://realpython.com/, distributed via a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported licence

The `threading library <https://docs.python.org/dev/library/threading.html#>`__ 
provides an API for creating and working with threads. The simplest approach to 
create and manage threads is to use the ``ThreadPoolExecutor`` class.
An example use case could be to download data from multiple websites using 
multiple threads:

.. code-block:: python

   import concurrent.futures

   def download_all_sites(sites):
       with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
           executor.map(my_download_function, sites)
  
The speedup gained from multithreading I/O bound problems can be understood from the following image.

.. figure:: img/Threading.png
  :align: center
  :scale: 50 %

  From https://realpython.com/, distributed via a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported licence

Further details on threading in Python can be found in the **See also** section below.


Multiprocessing
---------------

The ``multiprocessing`` module in Python supports spawning processes using an API 
similar to the ``threading`` module. It effectively side-steps the GIL by using 
*subprocesses* instead of threads, where each subprocess is an independent Python 
process.

One of the simplest ways to use ``multiprocessing`` is via ``Pool`` objects and 
the parallel :meth:`Pool.map` function, similarly to what we saw for multithreading above. 
In the following code, we define a :meth:`square` 
function, call the :meth:`cpu_count` method to get the number of CPUs on the machine,
and then initialize a Pool object in a context manager and inside of it call the 
:meth:`Pool.map` method to parallelize the computation.
We can save the code in a file named `mp_map.py` or download from :download:`here <example/mp_map.py>`.

.. literalinclude:: example/mp_map.py
   :language: python
   :emphasize-lines: 1, 11-12

For functions that take multiple arguments one can instead use the :meth:`Pool.starmap`
function (save as `mp_starmap.py` or download :download:`here <example/mp_starmap.py>`)

.. literalinclude:: example/mp_starmap.py
   :language: python
   :emphasize-lines: 1, 10-11

.. callout:: Interactive environments

   Functionality within multiprocessing requires that the ``__main__`` module be 
   importable by children processes. This means that for example ``multiprocessing.Pool`` 
   will not work in the interactive interpreter. A fork of multiprocessing, called 
   ``multiprocess``, can be used in interactive environments like Jupyter.

``multiprocessing`` has a number of other methods which can be useful for certain 
use cases, including ``Process`` and ``Queue`` which make it possible to have direct 
control over individual processes. Refer to the `See also`_ section below for a list 
of external resources that cover these methods.

At the end of this episode you can turn your attention back to the word-count problem 
and practice using ``multiprocessing`` pools of processes.


MPI
---

The message passing interface (MPI) is a standard workhorse of parallel computing. Nearly 
all major scientific HPC applications use MPI. Like ``multiprocessing``, MPI belongs to the 
distributed-memory paradigm.

The idea behind MPI is that:

- Tasks have a rank and are numbered 0, 1, 2, 3, ...
- Each task manages its own memory
- Each task can run multiple threads
- Tasks communicate and share data by sending messages.
- Many higher-level functions exist to distribute information to other tasks
  and gather information from other tasks.
- All tasks typically *run the entire code* and we have to be careful to avoid
  that all tasks do the same thing.

``mpi4py`` provides Python bindings for the Message Passing Interface (MPI) standard.
This is how a hello world MPI program looks like in Python:

.. code-block:: python
 
   from mpi4py import MPI

   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()
   
   print('Hello from process {} out of {}'.format(rank, size))

- ``MPI.COMM_WORLD`` is the `communicator` - a group of processes that can talk to each other
- ``Get_rank`` returns the individual rank (0, 1, 2, ...) for each task that calls it
- ``Get_size`` returns the total number of ranks.

To run this code with a specific number of processes we use the ``mpirun`` command which 
comes with the MPI library:

.. code-block:: console

   # on some HPC systems you might need 'srun -n 4' instead of 'mpirun -np 4'
   # on Vega, add this module for MPI libraries: ml add foss/2020b  
   $ mpirun -np 4 python hello.py

   # Hello from process 1 out of 4
   # Hello from process 0 out of 4
   # Hello from process 2 out of 4
   # Hello from process 3 out of 4

.. callout:: MPI libraries

   A number of available MPI libraries have been developed (`OpenMPI <https://www.open-mpi.org/>`__, 
   `MPICH <https://www.mpich.org/>`__, `IntelMPI <https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html#gs.up6uyn>`__, 
   `MVAPICH <http://mvapich.cse.ohio-state.edu/>`__) and HPC centers normally offer one or more of these for users 
   to compile/run MPI code.

   For example, on Vega one can load the GNU compiler suite along with OpenMPI using:

   .. code-block:: console

      $ ml add foss/2021b

Point-to-point and collective communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MPI standard contains a `lot of functionality <https://mpi4py.readthedocs.io/en/stable/index.html>`__, 
but in principle one can get away with only point-to-point communication (``MPI.COMM_WORLD.send`` and 
``MPI.COMM_WORLD.recv``). However, collective communication can sometimes require less effort as you 
will learn in an exercise below.
In any case, it is good to have a mental model of different communication patterns in MPI.

.. figure:: img/send-recv.png
   :align: center
   :scale: 100 %

   ``send`` and ``recv``: blocking point-to-point communication between two ranks.    

.. figure:: img/gather.png
   :align: right
   :scale: 80 %

   ``gather``: all ranks send data to rank ``root``.

.. figure:: img/scatter.png
   :align: center
   :scale: 80 %

   ``scatter``: data on rank 0 is split into chunks and sent to other ranks


.. figure:: img/broadcast.png
   :align: left
   :scale: 80 %

   ``bcast``: broadcast message to all ranks


.. figure:: img/reduction.png
   :align: center
   :scale: 100 %

   ``reduce``: ranks send data which are reduced on rank ``root``


Examples
~~~~~~~~

.. tabs::
 
   .. tab:: send/recv

      .. code-block:: python
         :emphasize-lines: 10, 14

         from mpi4py import MPI
   
         comm = MPI.COMM_WORLD
         rank = comm.Get_rank()
         n_ranks = comm.Get_size()
   
         if rank != 0:
             # All ranks other than 0 should send a message
             message = "Hello World, I'm rank {:d}".format(rank)
             comm.send(message, dest=0, tag=0)
         else:
             # Rank 0 will receive each message and print them
             for sender in range(1, n_ranks):
                 message = comm.recv(source=sender, tag=0)
                 print(message)      

   .. tab:: isend/irecv

      .. code-block:: python
         :emphasize-lines: 10,15

         from mpi4py import MPI

         comm = MPI.COMM_WORLD
         rank = comm.Get_rank()
         n_ranks = comm.Get_size()

         if rank != 0:
             # All ranks other than 0 should send a message
             message = "Hello World, I'm rank {:d}".format(rank)
             req = comm.isend(message, dest=0, tag=0)
             req.wait()
         else:
             # Rank 0 will receive each message and print them
             for sender in range(1, n_ranks):
                 req = comm.irecv(source=sender, tag=0)
                 message = req.wait()
                 print(message)          
   .. tab:: broadcast

      .. code-block:: python
         :emphasize-lines: 13
            
         from mpi4py import MPI
   
         comm = MPI.COMM_WORLD
         rank = comm.Get_rank()
         n_ranks = comm.Get_size()
   
         # Rank 0 will broadcast message to all other ranks
         if rank == 0:
             send_message = "Hello World from rank 0"
         else:
             send_message = None
   
         receive_message = comm.bcast(send_message, root=0)
   
         if rank != 0:
             print(f"rank {rank} received message: {receive_message}")       

   .. tab:: gather
      
      .. code-block:: python
         :emphasize-lines: 9
         
         from mpi4py import MPI
   
         comm = MPI.COMM_WORLD
         rank = comm.Get_rank()
         n_ranks = comm.Get_size()
   
         # Use gather to send all messages to rank 0
         send_message = "Hello World, I'm rank {:d}".format(rank)
         receive_message = comm.gather(send_message, root=0)
   
         if rank == 0:
             for i in range(n_ranks):
                 print(receive_message[i])     
   
   .. tab:: scatter

      .. code-block:: python
         :emphasize-lines: 14

         from mpi4py import MPI
         
         comm = MPI.COMM_WORLD
         size = comm.Get_size()
         rank = comm.Get_rank()
         
         if rank == 0:
             sendbuf = []
             for i in range(size):
                 sendbuf.append(f"Hello World from rank 0 to rank {i}")
         else:
             sendbuf = None
         
         recvbuf = comm.scatter(sendbuf, root=0)
         print(f"rank {rank} received message: {recvbuf}")

   MPI excels for problems which can be divided up into some sort of subdomains and 
   communication is required between the subdomains between e.g. timesteps or iterations.
   The word-count problem is simpler than that and MPI is somewhat overkill, but in an exercise 
   below you will learn to use point-to-point communication to parallelize it.

In addition to the lower-case methods :meth:`send`, :meth:`recv`, :meth:`broadcast` etc., there 
are also *upper-case* methods :meth:`Send`, :meth:`Recv`, :meth:`Broadcast`. These work with 
*buffer-like* objects (including strings and NumPy arrays) which have known memory location and size. 
Upper-case methods are faster and are strongly recommended for large numeric data.

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

Exercises
---------

.. demo:: Word-autocorrelation example project

   Inspired by a study of `dynamic correlations of words in written text 
   <https://www.scirp.org/journal/paperinformation.aspx?paperid=92643>`__,
   we decide to investigate autocorrelations (ACFs) of words in our database of book texts
   in the `word-count project <https://github.com/enccs/word-count-hpda>`__.
   Many of the exercises below are based on working with the following 
   word-autocorrelation code, so let us get familiar with it.

   .. solution:: Full source code

      .. literalinclude:: exercise/autocorrelation.py

   - The script takes three command-line arguments: the path of a datafile (book text), 
     the path to the processed word-count file, and the output filename for the 
     computed autocorrelation function.
   - The ``__main__`` block calls the :meth:`setup` function to preprocess the text  
     (remove delimiters etc.) and load the pre-computed word-count results.
   - :meth:`word_acf` computes the word ACF in a text for a given word using simple 
     for-loops (you will learn to speed it up later).
   - :meth:`ave_word_acf` loops over a list of words and computes their average ACF.

   To run this code for one book:

   .. code-block:: console

      $ python source/autocorrelation.py data/pg99.txt processed_data/pg99.dat results/acf_pg99.dat

   It will print out the time it took to calculate the ACF.      


.. exercise:: Measure Snakemake parallelisation efficiency

   Explore the parallel efficiency of Snakemake for the word-count project.

   First clone the repo:

   .. code-block:: console

      $ git clone https://github.com/ENCCS/word-count-hpda.git

   Run the workflow on one core and time it:

   .. code-block:: console

      $ time snakemake -j 1

   Now compare the execution time when using more processes. How much improvement can be obtained?

   The more time-consuming each job in the workflow is, the larger will be the parallel efficiency, 
   as you will see if you get to the last exercise below!

.. exercise:: Parallelize word-autocorrelation code with multiprocessing

   A serial version of the code is available in the 
   `source/autocorrelation.py <https://github.com/ENCCS/word-count-hpda/blob/main/source/autocorrelation.py>`__
   script in the word-count repository. The full script can be viewed above, 
   but we focus on the :meth:`word_acf` and :meth:`ave_word_acf` functions:

   .. literalinclude:: exercise/autocorrelation.py
      :pyobject: word_acf
      
   .. literalinclude:: exercise/autocorrelation.py
      :pyobject: ave_word_acf

   - Think about what this code is doing and try to find a good place to parallelize it using 
     a pool of processes. 
   - With or without having a look at the hints below, try to parallelize 
     the code using ``multiprocessing`` and use :meth:`time.time()` to measure the speedup when running 
     it for one book.
   - **Note**: You will not be able to use Jupyter for this task due to the above-mentioned limitation of ``multiprocessing``.

   .. solution:: Hints
 
      The most time-consuming parts of this code is the double-loop inside 
      :meth:`word_acf` (you can confirm this in an exercise in the next episode). 
      This function is called 16 times in the :meth:`ave_word_acf`
      function, once for each word in the top-16 list. This looks like a perfect place to use a multiprocessing 
      pool of processes!

      We would like to do something like:

      .. code-block:: python

         with Pool(4) as p:
             results = p.map(word_autocorr, words)

      However, there's an issue with this because :meth:`word_acf` takes 3 arguments ``(word, text, timesteps)``.
      We could solve this using the :meth:`Pool.starmap` function:

      .. code-block:: python

         with Pool(4) as p:
             results = p.starmap(word_acf, [(i,j,k) for i,j,k in zip(words, 10*[text], 10*[timestep])]

      But this might be somewhat inefficient because ``10*[text]`` might take up quite a lot of memory.
      One workaround is to use the ``partial`` method from ``functools`` which returns a new function with 
      partial application of the given arguments:

      .. code-block:: python

         from functools import partial
         word_acf_partial = partial(word_autocorr, text=text, timesteps=timesteps)
         with Pool(4) as p:
             results = p.map(word_acf_partial, words)

   .. solution::

      .. literalinclude:: exercise/autocorrelation_multiproc.py
         :language: python
   

.. exercise:: Write an MPI version of word-autocorrelation

   Just like with ``multiprocessing``, the most natural MPI solution parallelizes over 
   the words used to compute the word-autocorrelation.  
   For educational purposes, both point-to-point and collective communication 
   implementations will be demonstrated here.

   Start by importing mpi4py (``from mpi4py import MPI``) at the top of the script.

   Here is a new function which takes care of managing MPI tasks.
   The problem needs to be split up between ``N`` ranks, and the method needs to be general 
   enough to handle cases where the number of words is not a multiple of the number of ranks.
   Below we see a standard algorithm to accomplish this. The function also calls 
   two functions which implement point-to-point and collective communication, respectively, to collect 
   individual results to one rank which computes the average

   .. literalinclude:: exercise/autocorrelation_mpi.py
      :pyobject: mpi_acf
      :emphasize-lines: 11-12, 14-16, 18-20


   .. discussion:: What type of communication can we use?

      The end result should be an average of all the word-autocorrelation functions. 
      What type of communication can be used to collect the results on one rank which 
      computes the average and prints it to file?

   Study the two "faded" MPI function implementations below, one using point-to-point communication and the other using 
   collective communication. Try to figure out what you should replace the ``____`` with.

   .. tabs:: 

      .. tab:: Point-to-point

         .. code-block:: python

            def ave_word_acf_p2p(comm, my_words, text, timesteps=100):
                rank = comm.Get_rank()
                n_ranks = comm.Get_size()
                # each rank computes its own set of acfs
                my_acfs = np.zeros((len(____), timesteps))
                for i, word in enumerate(my_words):
                    my_acfs[i,:] = word_acf(word, text, timesteps)

                if ____ == ____:
                    results = []
                    # append own results
                    results.append(my_acfs)
                    # receive data from other ranks and append to results
                    for sender in range(1, ____):
                        results.append(comm.____(source=sender, tag=12))
                    # compute total 
                    acf_tot = np.zeros((timesteps,))
                    for i in range(____):
                        for j in range(len(results[i])):
                            acf_tot += results[i][j]
                    return acf_tot
                else:
                    # send data
                    comm.____(my_acfs, dest=____, tag=12)

      .. tab:: Collective

         .. code-block:: python

               def ave_word_acf_gather(comm, my_words, text, timesteps=100):
                   rank = comm.Get_rank()
                   n_ranks = comm.Get_size() 
                   # each rank computes its own set of acfs
                   my_acfs = np.zeros((len(____), timesteps))
                   for i, word in enumerate(my_words):
                       my_acfs[i,:] = word_acf(word, text, timesteps)

                   # gather results on rank 0
                   results = comm.____(____, root=0)
                   # loop over ranks and results. result is a list of lists of ACFs
                   if ____ == ____:
                       acf_tot = np.zeros((timesteps,))
                       for i in range(n_ranks):
                           for j in range(len(results[i])):
                               acf_tot += results[i][j]
                       return acf_tot



   After implementing one or both of these functions, run your code and time the result for different number of tasks!

   .. code-block:: console

      $ time mpirun -np <N> python source/autocorrelation.py data/pg58.txt processed_data/pg58.dat results/pg58_acf.csv


   .. solution:: 

      .. literalinclude:: exercise/autocorrelation_mpi.py

.. exercise:: Use the MPI version of word-autocorrelation with ipyparallel

   Now try to use the MPI version of the autocorrelation.py script inside Jupyter 
   using ipyparallel! Of course, you can also use the provided MPI solution above.

   Start by creating a new Jupyter notebook :file:`autocorrelation.ipynb` 
   in the :file:`word-count-hpda/source/` directory.

   Then start the IPython cluster with e.g. 8 cores in a Jupyter **terminal**:

   .. code-block:: console

      $ ipcluster start -n 8 --engines=MPI

   Now create a cluster in Jupyter:

   .. code-block:: python

      import ipyparallel as ipp
      cluster = ipp.Client()

   Instead of copying functions from :file:`autocorrelation.py` to your notebook, you can 
   import them *on each engine*. But you may first need to change the current working 
   directory (CWD) if your Jupyter session was started in the :file:`word-count-hpda/` directory:

   .. code-block:: python

      import os
      # create a direct view to be able to change CWD on engines
      dview = rc.direct_view()
      # print CWD on each engine
      print(dview.apply_sync(os.getcwd))
      # set correct CWD, adapt if needed (run %pwd to find full path)
      dview.map(os.chdir, ['/full/path/to/word-count-hpda/source']*len(cluster))

   Now you need to import all needed functions explicitly on the engines: 

   .. code-block:: python

      with view.sync_imports():
          from autocorrelation import preprocess_text, setup, word_acf
          from autocorrelation import ave_word_acf_gather, ave_word_acf_p2p, mpi_acf

   Finally you're ready to run MPI code on the engines! The following code uses 
   :meth:`apply_sync` to run the :meth:`mpi_acf` function on all engines with given 
   input arguments:

   .. code-block:: python

      # run the mpi_example function on all engines in parallel
      book = "../data/pg99.txt"
      wc_book = "../processed_data/pg99.dat"
      r = view.apply_sync(mpi_acf, book, wc_book)

      # Print the result from the engines
      print(r[0])

   Tasks:

   - Time the execution of the last code cell by adding ``%%time`` at the top of the cell.
   - Stop the cluster in terminal (CTRL-c), and start a new cluster with a different number 
     of MPI engines. Time the cell again to explore the parallel efficiency.
   - Instead of running through only one data file (book), create a loop to run through 
     them all.

.. exercise:: Extend the Snakefile

   Extend the Snakefile in the word-count repository to compute the autocorrelation function for all 
   books! If you are running on a cluster you can add e.g. ``threads: 4`` to the rule and run a parallel 
   version of the ``autocorrelation.py`` script that you wrote in an earlier exercise.

   .. solution:: Hints

      Apart from adding a new rule for computing the autocorrelation functions, you will need to add dependencies 
      to the top-level ``all`` rule in order to instruct Snakemake to run your new rule. For instance, you 
      can replace it with:

      .. code-block:: python

         rule all:
             input:
                 'results/results.txt', expand('results/acf_{book}.dat', book=DATA)
 
      Make sure to name the ``output`` files accordingly in your new rule.

   .. solution::

      .. literalinclude:: exercise/Snakefile
         :language: python



.. _See also:

See also
--------

- `More on the global interpreter lock
  <https://wiki.python.org/moin/GlobalInterpreterLock>`__
- `RealPython concurrency overview <https://realpython.com/python-concurrency/>`__
- `RealPython threading tutorial <https://realpython.com/intro-to-python-threading/>`__
- Parallel programming in Python with multiprocessing, 
  `part 1 <https://www.kth.se/blogs/pdc/2019/02/parallel-programming-in-python-multiprocessing-part-1/>`__
  and `part 2 <https://www.kth.se/blogs/pdc/2019/03/parallel-programming-in-python-multiprocessing-part-2/>`__
- Parallel programming in Python with mpi4py, `part 1 <https://www.kth.se/blogs/pdc/2019/08/parallel-programming-in-python-mpi4py-part-1/>`__
  and `part 2 <https://www.kth.se/blogs/pdc/2019/11/parallel-programming-in-python-mpi4py-part-2/>`__
- `ipyparallel documentation <https://ipyparallel.readthedocs.io/en/latest/>`__
- `IPython Parallel in 2021 <https://blog.jupyter.org/ipython-parallel-in-2021-2945985c032a>`__
- `ipyparallel tutorial <https://github.com/DaanVanHauwermeiren/ipyparallel-tutorial>`__




.. keypoints::

   - 1
   - 2
   - 3
