.. _parallel-computing:

Parallel computing
==================

.. objectives::

   - Understand the Global Interpreter Lock in Python
   - Understand concurrency
   - Understand the difference between multithreading and multiprocessing
   - Learn the basics of *multiprocessing*, *threading*, *ipyparallel* and *MPI4Py*

.. instructor-note::

   - 40 min teaching/type-along
   - 40 min exercises


The performance of a single CPU core has stagnated over the last ten years,
and as such most of the speed-up in modern CPUs is coming from using multiple
CPU cores, i.e. parallel processing. Parallel processing is normally based
either on multiple threads or multiple processes. Unfortunately, the memory
management of the standard CPython interpreter is not thread-safe, and it uses
something called Global Interpreter Lock (GIL) to safeguard memory integrity.
In practice, this limits the benefits of multiple threads only to some
special situations (e.g. I/O). Fortunately, parallel processing with multiple
processes is relatively straightforward also with Python.

There are three main models of parallel computing:

- **Embarrassingly parallel:** the code does not need to synchronize/communicate
  with other instances, and you can run
  multiple instances of the code separately, and combine the results
  later.  If you can do this, great!  

- **Shared memory parallelism (multithreading):** 
 
  - Parallel threads do separate work and communicate via the same memory and write to shared variables.
  - Multiple threads in a single Python program cannot execute at the same time (see GIL below)
  - Running multiple threads in Python is *only effective for I/O-bound tasks*
  - External libraries in other languages (e.g. C) which are called from Python can still use multithreading

- **Distributed memory parallelism (multiprocessing):** Different processes manage their own memory segments and 
  share data by communicating (passing messages) as needed.

  - A process can contain one or more threads
  - Two processes can run on different CPU cores and different computers
  - Processes have more overhead than threads (creating and destroying processes takes more time)
  - Running multiple processes is *only effective for CPU-bound tasks*

In the next episode we will look at `Dask <https://dask.org/>`__, an array model extension and task scheduler, 
which goes beyond these three approaches.

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
- If speed is important enough you need things parallel, you usually
  wouldn't use pure Python.
- There are several Python libraries that side-step the GIL, e.g. by using 
  *subprocesses* instead of threads.


Parallel workflows with Snakemake
---------------------------------

Many scientific problems involve complicated workflows with multiple interdependent steps.
If the workflow involves performing the same analysis on many different datasets we can 
use the inherent ("embarrassing") parallelism of the problem and perform these simultaneously.

Let us have a look at a toy example which many of us can hopefully relate to. 

.. demo:: The word-count project

   Head over to https://github.com/enccs/word-count-hpda and clone the repository:

   .. code-block:: bash

      git clone git@github.com:ENCCS/word-count-hpda.git

   This toy project is about analyzing the frequency of words in texts. The ``data``
   directory contains 64 public domain books from Project Gutenberg and source files 
   under ``source`` can be used to count words:

   .. code-block:: bash

      # count words in two books
      python source/wordcount.py data/pg10.txt > processed_data/pg10.dat
      python source/wordcount.py data/pg65.txt > processed_data/pg65.dat
      
      # print frequency of 10 most frequent words in both books to file
      python source/zipf_test.py 10 pg10.dat pg65.dat > results.txt
      
   This workflow is encoded in the ``Snakefile`` which can be used to run
   through all data files:

   .. code-block:: bash

      # run workflow in serial
      snakemake -j 1      


   The workflow can be visualised in a directed-acyclic graph:

   .. code-block:: bash

      # requires dot from Graphviz
      snakemake -j 1 --dag | dot -Tpng  > dag.png

   .. figure:: img/dag.png
      :align: center
      :scale: 80 %

   The workflow can be parallelized to utilize multiple cores:

   .. code-block:: bash

      # first clear all output
      snakemake -j 1 --delete-all-output      
      # run in parallel on 4 processes
      snakemake -j 4

   **Task:**

   - Compare the execution time when using 1, 2 and 4 processes

The Snakefile describes the workflow in declarative style, i.e. we describe 
the dependencies but let Snakemake figure out the series of steps to produce 
results (targets). This is how the Snakefile looks:

.. code-block:: python

   # a list of all the books we are analyzing
   DATA = glob_wildcards('data/{book}.txt').book
   
   # the default rule
   rule all:
       input:
           'results/results.txt'
   
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
       output: 'results/results.txt'
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

.. code-block:: bash

   export OMP_NUM_THREADS=<N>

After setting this environment variable we continue as usual 
and multithreading will be turned on.

.. demo:: Multithreading NumPy 

   Here is an example which does a symmetrical matrix inversion of size 4000 by 4000.
   To run it, we can save it in a file named `omp_test.py`.

   .. code-block:: python

      import numpy as np
      import time
      
      A = np.random.random((4000,4000))
      A = A * A.T
      time_start = time.time()
      np.linalg.inv(A)
      time_end = time.time()
      print("time spent for inverting A is", round(time_end - time_start,2), 's')

   Let us test it with 1 and 4 threads:

   .. code-block:: bash

      export OMP_NUM_THREADS=1
      python omp_test.py

      export OMP_NUM_THREADS=4
      python omp_test.py

Multithreaded I/O
^^^^^^^^^^^^^^^^^

This is how an I/O-bound application might look:

.. figure:: img/IOBound.png
   :align: center
   :scale: 40 %

   From https://realpython.com/, distributed via a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported licence

The `threading library <https://docs.python.org/dev/library/threading.html#>`__ 
provides an API for creating and working with threads. We restrict our discussion 
here to using the ``ThreadPoolExecutor`` class to multithread reading and writing 
to files. For further details on ``threading`` refer to the **See also** section below.


.. demo:: Multithreading file I/O

   We continue with the word-count project and explore how we can use multithreading 
   for I/O. After running ``snakemake -j 1`` we should have 64 ``.dat`` files in the 
   ``processed_data`` directory. Let's say we want to convert them all to csv format.

   The easiest way to use multithreading is to use the ``ThreadPoolExecutor``
   from ``concurrent.futures``. Here is a comparison of serial and multithreaded 
   code to accomplish this:

   .. tabs:: 

      .. tab:: Serial

         .. code-block:: python
      
            import glob
            import time
            
            def csvify_file(file):
                with open(file, 'r') as f:
                    lines = f.readlines()
                with open(file.replace('.dat', '.csv'), 'w') as f:
                    for line in lines:
                        f.write(line.replace(' ', ','))
            
            def csvify_all_files(files):
                for file in files:
                    csvify_file(file)
                    #break
                    
            if __name__ == '__main__':
                files = glob.glob("processed_data/*.dat")
                start_time = time.time()
                csvify_all_files(files)
                duration = time.time() - start_time
                print(f"Read {len(files)} in {duration} seconds")   

      .. tab:: Multithreaded

         .. code-block:: python
            :emphasize-lines: 2, 13-14

            import glob
            import concurrent.futures
            import time

            def csvify_file(file):
                with open(file, 'r') as f:
                    lines = f.readlines()
                with open(file.replace('.dat', '.csv'), 'w') as f:
                    for line in lines:
                        f.write(line.replace(' ', ','))        

            def csvify_all_files(files):
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    executor.map(read_file, files)

            if __name__ == '__main__':
                files = glob.glob("processed_data/*.dat")
                start_time = time.time()
                csvify_all_files(files)
                duration = time.time() - start_time
                print(f"Read {len(files)} in {duration} seconds")      

   Tasks:

   1. Run these codes and observe the timing information.
   2. You will likely not see a speedup. Try increasing the I/O by multiplying the data before writing 
      it to file, i.e. insert ``line *= 100`` just before ``f.write(...)``. Does multithreading now pay off?
  
The speedup gained from multithreading our problem can be understood from the following image.

.. figure:: img/Threading.png
  :align: center
  :scale: 50 %

  From https://realpython.com/, distributed via a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported licence




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
:meth:`Pool.map` method to parallelize the computation:

.. code-block:: python

   import multiprocessing as mp
   
   def square(x):
       return x * x
   
   if __name__ == '__main__':
       nprocs = mp.cpu_count()
       print(f"Number of CPU cores: {nprocs}")
   
       # use context manager to allocate and release the resources automatically
       with mp.Pool(processes=nprocs) as pool:
           result = pool.map(square, range(20))    
       print(result)
 
For functions that take multiple arguments one can instead use the :meth:`Pool.starmap`
function:

.. code-block:: python

   import multiprocessing as mp

   def power_n(x, n):
       return x ** n

   if __name__ == '__main__':
       nprocs = mp.cpu_count()
       print(f"Number of CPU cores: {nprocs}")
  
       with mp.Pool(processes=nprocs) as pool:
           result = pool.starmap(power_n, [(x, 2) for x in range(20)])
       print(result)

.. callout:: Interactive environments

   Functionality within multiprocessing requires that the ``__main__`` module be 
   importable by children processes. This means that for example ``multiprocessing.Pool`` 
   will not work in the interactive interpreter. A fork of multiprocessing, called 
   ``multiprocess``, can be used in interactive environments like IPython sessions.

``multiprocessing`` has a number of other methods which can be useful for certain 
use cases, including ``Process`` and ``Queue`` which make it possible to have direct 
control over individual processes. Refer to the `See also`_ section below for a list 
of external resources that cover these methods.

At the end of this episode you can turn your attention back to the word-count problem 
and practice using ``multiprocessing`` pools of processes.


ipyparallel
-----------

- https://blog.jupyter.org/ipython-parallel-in-2021-2945985c032a
- https://coderefinery.github.io/jupyter/examples/#parallel-python-with-ipyparallel
- https://github.com/DaanVanHauwermeiren/ipyparallel-tutorial


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

.. code-block:: bash

   # on some HPC systems you might need 'srun -n 4' instead of 'mpirun -np 4'  
   mpirun -np 4 hello.py

A number of available MPI libraries have been developed (`OpenMPI <https://www.open-mpi.org/>`__, 
`MPICH <https://www.mpich.org/>`__, `IntelMPI <https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html#gs.up6uyn>`__, 
`MVAPICH <http://mvapich.cse.ohio-state.edu/>`__) and HPC centers normally offer one or more of these for users 
to compile/run their own code.


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

         from mpi4py import MPI
   
         comm = MPI.COMM_WORLD
         # Get my rank and the number of ranks
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

   .. tab:: broadcast

      .. code-block:: python
            
         from mpi4py import MPI
   
         comm = MPI.COMM_WORLD
         # Get my rank and the number of ranks
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
         
         from mpi4py import MPI
   
         comm = MPI.COMM_WORLD
         # Get my rank and the number of ranks
         rank = comm.Get_rank()
         n_ranks = comm.Get_size()
   
         # Use gather to send all messages to rank 0
         send_message = "Hello World, I'm rank {:d}".format(rank)
         receive_message = comm.gather(send_message, root=0)
   
         if rank == 0:
             for i in range(n_ranks):
                 print(receive_message[i])     
   
   MPI excels for problems which can be divided up into some sort of subdomains and 
   communication is required between the subdomains between e.g. timesteps or iterations.
   The word-count problem is simpler than that and MPI is somewhat overkill, but in an exercise 
   below you will learn to use point-to-point communication to parallelize it.


Exercises
---------

.. exercise:: Word-autocorrelation: parallelizing word-count with multiprocessing

   Inspired by a study of 
   `dynamic correlations of words in written text <https://www.scirp.org/journal/paperinformation.aspx?paperid=92643>`__,
   we decide to investigate autocorrelations of words in our database of book texts.

   A serial version of the code is available in the 
   `source/autocorrelation.py <https://github.com/ENCCS/word-count-hpda/blob/main/source/autocorrelation.py>`__
   script in the word-count repository. The full script can be viewed below, 
   but we focus on the :meth:`word_autocorr` and :meth:`word_autocorr_average` functions:

   .. code-block:: python
         
      def word_autocorr(word, text, timesteps):
          """
          Calculate word-autocorrelation function for given word 
          in a text. Each word in the text corresponds to one "timestep".
          """
          acf = np.zeros((timesteps,))
          mask = [w==word for w in text]
          nwords_chosen = np.sum(mask)
          nwords_total = len(text)
          for t in range(timesteps):
              for i in range(1,nwords_total-t):
                  acf[t] += mask[i]*mask[i+t]
              acf[t] /= nwords_chosen      
          return acf
          
      def word_autocorr_average(words, text, timesteps=100):
          """
          Calculate an average word-autocorrelation function 
          for a list of words in a text.
          """
          acf = np.zeros((len(words), timesteps))
          for n, word in enumerate(words):
              acf[n, :] = word_autocorr(word, text, timesteps)
          return np.average(acf, axis=0)


   .. solution:: Full script

      .. code-block:: python
   
         import sys
         import numpy as np
         from wordcount import load_word_counts, load_text, DELIMITERS
         import time
         
         def preprocess_text(text):
             """
             Remove delimiters, split lines into words and remove whitespaces, 
             and make lowercase. Return list of all words in the text.
             """
             clean_text = []
             for line in text:
                 for purge in DELIMITERS:
                     line = line.replace(purge, " ")    
                 words = line.split()
                 for word in words:
                     word = word.lower().strip()
                     clean_text.append(word)
             return clean_text
         
         def word_autocorr(word, text, timesteps):
             """
             Calculate word-autocorrelation function for given word 
             in a text. Each word in the text corresponds to one "timestep".
             """
             acf = np.zeros((timesteps,))
             mask = [w==word for w in text]
             nwords_chosen = np.sum(mask)
             nwords_total = len(text)
             for t in range(timesteps):
                 for i in range(1,nwords_total-t):
                     acf[t] += mask[i]*mask[i+t]
                 acf[t] /= nwords_chosen      
             return acf
             
         def word_autocorr_average(words, text, timesteps=100):
             """
             Calculate an average word-autocorrelation function 
             for a list of words in a text.
             """
             acf = np.zeros((len(words), timesteps))
             for n, word in enumerate(words):
                 acf[n, :] = word_autocorr(word, text, timesteps)
             return np.average(acf, axis=0)
         
         if __name__ == '__main__':          
             # load book text and preprocess it
             book = sys.argv[1]
             text = load_text(book)
             clean_text = preprocess_text(text)
             # load precomputed word counts and select top 10 words
             wc_book = sys.argv[2]
             nwords = 10
             word_count = load_word_counts(wc_book)
             top_words = [w[0] for w in word_count[:nwords]]
             # number of "timesteps" to use in autocorrelation function
             timesteps = 100
             # compute average autocorrelation and time the execution
             t0 = time.time()
             acf_ave = word_autocorr_average(top_words, clean_text, timesteps=100)
             t1 = time.time()        
             print(f"serial time: {t1-t0}")
             # save results to csv file
             np.savetxt(sys.argv[3], np.vstack((np.arange(1,timesteps+1), acf_ave)).T, delimiter=',')

      

   - :meth:`word_autocorr` computes the autocorrelation in a text for a given word
   - :meth:`word_autocorr_average` loops over a list of words and computes their average autocorrelation
   - To run this code: 

     .. code-block:: bash

        python source/autocorrelation.py data/pg99.txt processed_data/pg99.dat results/pg99_acf.csv

   .. discussion:: Where to parallelise?

      Think about what this code is doing and try to find a good place to parallelize it using 
      a pool of processes. With or without having a look at the hints below, try to parallelize 
      the code using ``multiprocessing`` and use :meth:`time.time()` to measure the speedup when running 
      it for one book.

   .. solution:: Hints
 
      The most time-consuming parts of this code is the double-loop inside 
      :meth:`word_autocorr` (you can confirm this in an exercise below). 
      This function is called 10 times in the :meth:`word_autocorr_average`
      function, once for each word in the top-10 list. This looks like a perfect place to use a multiprocessing 
      pool of processes!

      We would like to do something like:

      .. code-block:: python

         with Pool(4) as p:
             results = p.map(word_autocorr, words)

      However, there's an issue with this because :meth:`word_autocorr` takes 3 arguments ``(word, text, timesteps)``.
      We could solve this using the :meth:`Pool.starmap` function:

      .. code-block:: python

         with Pool(4) as p:
             results = p.starmap(word_autocorr, [(i,j,k) for i,j,k in zip(words, 10*[text], 10*[timestep])]

      But this might be somewhat inefficient because ``10*[text]`` might take up quite a lot of memory.
      A workaround is to use the ``partial`` method from ``functools`` which returns a new function with 
      partial application of the given arguments:

      .. code-block:: python

         from functools import partial
         word_autocorr_partial = partial(word_autocorr, text=text, timesteps=timesteps)
         with Pool(4) as p:
             results = p.map(word_autocorr_partial, words)

   .. solution::

      .. code-block:: python

         import sys
         import numpy as np
         from wordcount import load_word_counts, load_text, DELIMITERS
         import time
         from multiprocessing import Pool
         from functools import partial
         
         def preprocess_text(text):
             """
             Remove delimiters, split lines into words and remove whitespaces, 
             and make lowercase. Return list of all words in the text.
             """
             clean_text = []
             for line in text:
                 for purge in DELIMITERS:
                     line = line.replace(purge, " ")    
                 words = line.split()
                 for word in words:
                     word = word.lower().strip()
                     clean_text.append(word)
             return clean_text
         
         def word_autocorr(word, text, timesteps):
             """
             Calculate word-autocorrelation function for given word 
             in a text. Each word in the text corresponds to one "timestep".
             """
             acf = np.zeros((timesteps,))
             mask = [w==word for w in text]
             nwords_chosen = np.sum(mask)
             nwords_total = len(text)
             for t in range(timesteps):
                 for i in range(1,nwords_total-t):
                     acf[t] += mask[i]*mask[i+t]
                 acf[t] /= nwords_chosen      
             return acf
             
         def word_autocorr_average(words, text, timesteps=100):
             """
             Calculate an average word-autocorrelation function 
             for a list of words in a text.
             """
             acf = np.zeros((len(words), timesteps))
             for n, word in enumerate(words):
                 acf[n, :] = word_autocorr(word, text, timesteps)
             return np.average(acf, axis=0)
         
         def word_autocorr_average_pool(words, text, timesteps=100):
             """
             Calculate an average word-autocorrelation function 
             for a list of words in a text using multiprocessing.
             """
             word_autocorr_partial = partial(word_autocorr, text=text, timesteps=timesteps)
             with Pool(4) as p:
                 results = p.map(word_autocorr_partial, words)
             acf = np.array(results)
             return np.average(acf, axis=0)
         
         if __name__ == '__main__':          
             # load book text and preprocess it
             book = sys.argv[1]
             text = load_text(book)
             clean_text = preprocess_text(text)
             # load precomputed word counts and select top 10 words
             wc_book = sys.argv[2]
             nwords = 10
             word_count = load_word_counts(wc_book)
             top_words = [w[0] for w in word_count[:nwords]]
             # number of "timesteps" to use in autocorrelation function
             timesteps = 100
             # compute average autocorrelation and time the execution
             t0 = time.time()
             acf_ave = word_autocorr_average(top_words, clean_text, timesteps=100)
             t1 = time.time()        
             acf_pool_ave = word_autocorr_average_pool(top_words, clean_text, timesteps=100)
             t2 = time.time()        
             print(f"serial time: {t1-t0}")
             print(f"parallel map time: {t2-t1}")
             np.testing.assert_array_equal(acf_ave, acf_pool_ave)     
   


.. exercise:: MPI version of word-autocorrelation

   Just like with ``multiprocessing``, the most natural MPI solution parallelizes over 
   the words used to compute the word-autocorrelation.  
   For educational purposes, both point-to-point and collective communication 
   implementations will be demonstrated here.

   Start by standard boilerplate code in the ``__main__`` module:

   .. code-block:: python
      :emphasize-lines: 2, 18-20

      # this should go at the top of the script
      from mpi4py import MPI

      # this is at the bottom
      if __name__ == '__main__':
          # load book text and preprocess it
          book = sys.argv[1]
          text = load_text(book)
          clean_text = preprocess_text(text)
          # load precomputed word counts and select top 10 words
          wc_book = sys.argv[2]
          nwords = 10
          word_count = load_word_counts(wc_book)
          top_words = [w[0] for w in word_count[:nwords]]
          # number of "timesteps" to use in autocorrelation function
          timesteps = 100
      
          # initialize MPI
          comm = MPI.COMM_WORLD
          rank = comm.Get_rank()
          n_ranks = comm.Get_size()    
      
   You now need to split the problem up between ``N`` ranks. The method needs to be general 
   enough to handle cases where the number of words is not a multiple of the number of ranks.
   Here's a standard algorithm to accomplish this. Again edit the ``__main__`` module:

   .. code-block:: python
      :emphasize-lines: 3-4, 6-8, 10-12

      #
          # distribute words among MPI tasks
          count = nwords // n_ranks
          remainder = nwords % n_ranks
          # first 'remainder' ranks get 'count + 1' tasks each
          if rank < remainder:
              first = rank * (count + 1)
              last = first + count + 1
          # remaining 'nwords - remainder' ranks get 'count' task each
          else:
              first = rank * count + remainder
              last = first + count 
          # each rank gets unique words
          my_words = top_words[first:last]
          print(f"My rank number is {rank} and first, last = {first}, {last}")

   With the ``top_words`` list split between the ranks, the ranks can now perform their job independently.

   .. discussion:: What type of communication can we use?

      Each rank has now computed word-autocorrelation functions for several texts.
      The end result should be an average of all the word-autocorrelation functions. 
      What type of communication can be used to collect the results on one rank which 
      computes the average and prints it to file?

   Study the two "faded" MPI function implementations below, one using point-to-point communication and the other using 
   collective communication. Try to figure out what you should replace the ``____`` with.

   .. tabs:: 

      .. tab:: Point-to-point

         .. code-block:: python

            def word_count_average_mpi_p2p(my_words, text, rank, n_ranks, timesteps=100):
                # each rank computes its own set of acfs
                my_acfs = np.zeros((len(____), timesteps))
                for i, word in enumerate(my_words):
                    my_acfs[i,:] = word_autocorr(word, text, timesteps)
            
                if ____ == ____:
                    results = []
                    # append own results
                    results.append(my_acfs)
                    # receive data from other ranks and append to results
                    for sender in range(1, ____):
                        results.append(comm.recv(source=____, tag=12))
                    # compute average and write to file
                    acf_tot = np.zeros((timesteps,))
                    for i in range(____):
                        for j in range(len(results[i])):
                            acf_tot += results[i][j]
                    acf_ave = acf_tot / nwords
                    return acf_ave
                else:
                    # send data
                    comm.send(my_acfs, dest=____, tag=12)

      .. tab:: Collective

         .. code-block:: python

            def word_count_average_mpi_collective(my_words, text, rank, n_ranks, timesteps=100):
                # each rank computes its own set of acfs
                my_acfs = np.zeros((len(____), timesteps))
                for i, word in enumerate(my_words):
                    my_acfs[i,:] = word_autocorr(word, text, timesteps)

                # gather results on rank 0
                results = comm.gather(____, root=0)
                # loop over ranks and results. result is a list of lists of ACFs
                if ____ == ____:
                    acf_tot = np.zeros((timesteps,))
                    for i in range(____):
                        for j in range(len(results[i])):
                            acf_tot += results[i][j]
                    # compute average and write to file
                    acf_ave = acf_tot / nwords
                    return acf_ave

   To call these functions and write results to disk in the ``__main__`` module, you can do:

   .. code-block:: python

      # 
          # use collective version
          #acf_ave = word_count_average_mpi_collective(my_words, clean_text, rank, n_ranks, timesteps=100)
      
          # use p2p version
          acf_ave = word_count_average_mpi_p2p(my_words, clean_text, rank, n_ranks, timesteps=100)
      
          # only rank 0 has the averaged data
          if rank == 0:
              np.savetxt(sys.argv[3], np.vstack((np.arange(1,101), acf_ave)).T, delimiter=',')      

   Try running your code and time the result for different number of tanks!

   .. code-block:: bash

      time mpirun -np <N> python source/autocorrelation.py data/pg58.txt processed_data/pg58.dat results/pg58_acf.csv


   .. solution:: 

      A solution with both point-to-point and collective communication can be 
      found on a `branch in the word-count-hpda repository 
      <https://github.com/ENCCS/word-count-hpda/blob/autocorr-mpi/source/autocorrelation.py>`__.
      You can also switch to the branch in your repository:

      .. code-block:: bash

         # first commit any work you have done:
         git add -u 
         git commit -m "save my work"
         # switch branch
         git checkout autocorr-mpi
                
.. exercise:: Extend the Snakefile

   Extend the Snakefile in the word-count repository to compute the autocorrelation function for all 
   books! If you are running on a cluster you can add e.g. ``threads: 4`` to the rule and run a parallel 
   version of the ``autocorrelation.py`` script.


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






.. keypoints::

   - 1
   - 2
   - 3
