Parallel computing
==================

.. objectives::

   - Understand the Global Interpreter Lock in Python
   - Understand concurrency
   - Understand the difference between multithreading and multiprocessing
   - Learn the basics of *multiprocessing*, *threading*, *ipyparallel* and *MPI4Py*



**Parallel computing** is when many different tasks are carried out
simultaneously.  There are three main models:

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

.. type-along:: The word-count project

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
      
      # (optionally) create plots
      python source/plotcount.py processed_data/pg10.dat results/pg10.png
      python source/plotcount.py processed_data/pg65.dat results/pg65.png
      
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
   
   # create a plot for each book
   rule make_plot:
      input:
          plotcount='source/plotcount.py',
          book='processed_data/{file}.dat'
      output: 'results/{file}.png'
      shell: 'python {input.plotcount} {input.book} {output}'
   
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
threading rather useless for compute-bound problems. However, threading is 
still an appropriate model for running *multiple I/O-bound tasks simultaneously*.

This is how an I/O-bound application might look.

.. figure:: img/IOBound.png
   :align: center
   :scale: 40 %

   From https://realpython.com/, distributed via a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported licence

The `threading library <https://docs.python.org/dev/library/threading.html#>`__ 
provides an API for creating and working with threads. We restrict our discussion 
here to using the ``ThreadPoolExecutor`` class to multithread reading and writing 
to files. For further details on ``threading`` refer to the **See also** section below.


.. type-along:: Multithreading file I/O

   We continue with the word-count project and explore how we can use multithreading 
   for I/O. After running ``snakemake -j 1`` we should have 64 ``.dat`` files in the 
   ``processed_data`` directory. Let's say we want to convert them all to csv format.

   Here is code to accomplish this:

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


   The easiest way to multithread this code is to use the ``ThreadPoolExecutor``
   from ``concurrent.futures``:

   .. code-block:: python

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

.. callout:: Interactive environments

   Functionality within multiprocessing requires that the ``__main__`` module be 
   importable by children processes. This means that for example ``multiprocessing.Pool`` 
   will not work in the interactive interpreter. A fork of multiprocessing, called 
   ``multiprocess``, can be used in interactive environments like IPython sessions.


One of the simplest ways to use ``multiprocessing`` is via ``Pool`` objects and 
the parallel ``Pool.map`` function. In the following code, we define a ``square`` 
function, call the ``cpu_count`` method to get the number of CPUs on the machine,
and then initialize a Pool object in a context manager and inside of it call the 
``Pool.map`` method to parallelize the computation:

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
 
For functions that take multiple arguments one can instead use the ``Pool.starmap`` 
function:

.. code-block:: python

   def power_n(x, n):
       return x ** n

   with mp.Pool(processes=nprocs) as pool:
       result = pool.starmap(power_n, [(x, 2) for x in range(20)])
   print(result)

``multiprocessing`` has a number of other methods which can be useful for certain 
use cases, including ``Process`` and ``Queue`` which make it possible to have direct 
control over individual processes. Refer to the `See also`_ section below for a list 
of external resources that cover these methods.

We now turn our attention back to the word-count problem.

.. type-along:: Word-autocorrelation: parallelizing word-count with multiprocessing

   Inspired by a study of 
   `dynamic correlations of words in written text <https://www.scirp.org/journal/paperinformation.aspx?paperid=92643>`__,
   we decide to investigate autocorrelations of words in our database of book texts.

   We add a file to the `word_count_hpda/source` directory called `autocorrelation.py`:

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


   - ``word_autocorr`` computes the autocorrelation in a text for a given word
   - ``word_autocorr_average``` loops over a list of words and computes their average autocorrelation
   - To run this code: ``python source/autocorrelation.py data/pg99.txt processed_data/pg99.dat results/pg99_acf.csv``

   .. discussion:: Where to parallelise?

      Think about what this code is doing and try to find a good place to parallelize it using 
      a pool of processes.

   .. solution:: Hints
 
      The most time-consuming parts of this code is the double-loop inside ``word_autocorr`` (you can 
      confirm this in an exercise below). This function is called 10 times in the ``word_autocorr_average``
      function, once for each word in the top-10 list. This looks like a perfect place to use a multiprocessing 
      pool of processes!

      We would like to do something like:

      .. code-block:: python

         with Pool(4) as p:
             results = p.map(word_autocorr, words)

      However, there's an issue with this because ``word_autocorr`` takes 3 arguments ``(word, text, timesteps)``.
      We could solve this using the ``Pool.starmap`` function:

      .. code-block:: python

         with Pool(4) as p:
             results = p.starmap(word_autocorr, [(i,j,k) for i,j,k in zip(words, 10*[text], 10*[timestep])]

      But this might be somewhat inefficient because ``10*[text]`` might take up quite a lot of memory.
      A workaround is to use the `partial` method from ``functools`` which returns a new function with 
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
   
      


ipyparallel
-----------

- https://blog.jupyter.org/ipython-parallel-in-2021-2945985c032a
- https://coderefinery.github.io/jupyter/examples/#parallel-python-with-ipyparallel
- https://github.com/DaanVanHauwermeiren/ipyparallel-tutorial


MPI
---

The message passing interface (MPI) approach to parallelization
is that:

- Tasks (cores) have a rank and are numbered 0, 1, 2, 3, ...
- Each task (core) manages its own memory
- Tasks communicate and share data by sending messages.
- Many higher-level functions exist to distribute information to other tasks
  and gather information from other tasks.
- All tasks typically run the entire code and we have to be careful to avoid
  that all tasks do the same thing.

Those who use MPI in C, C++, Fortran, will probably understand the steps in the
following example. For learners new to MPI, we can explore this example
together.

Here we reuse the example of approximating *pi* with a stochastic
algorithm from above, and we have highlighted the lines which are important
to get this MPI example to work:

.. code-block:: python
   :emphasize-lines: 3,23-25,29,39,42

   import random
   import time
   from mpi4py import MPI


   def sample(n):
       """Make n trials of points in the square.  Return (n, number_in_circle)

       This is our basic function.  By design, it returns everything it 
       needs to compute the final answer: both n (even though it is an input
       argument) and n_inside_circle.  To compute our final answer, all we
       have to do is sum up the n:s and the n_inside_circle:s and do our
       computation"""
       n_inside_circle = 0
       for i in range(n):
           x = random.random()
           y = random.random()
           if x ** 2 + y ** 2 < 1.0:
               n_inside_circle += 1
       return n, n_inside_circle


   comm = MPI.COMM_WORLD
   size = comm.Get_size()
   rank = comm.Get_rank()

   n = 10 ** 7

   if size > 1:
       n_task = int(n / size)
   else:
       n_task = n

   t0 = time.perf_counter()
   _, n_inside_circle = sample(n_task)
   t = time.perf_counter() - t0

   print(f"before gather: rank {rank}, n_inside_circle: {n_inside_circle}")
   n_inside_circle = comm.gather(n_inside_circle, root=0)
   print(f"after gather: rank {rank}, n_inside_circle: {n_inside_circle}")

   if rank == 0:
       pi_estimate = 4.0 * sum(n_inside_circle) / n
       print(
           f"\nnumber of darts: {n}, estimate: {pi_estimate}, time spent: {t:.2} seconds"
       )



Exercises
---------

.. exercise:: Using MPI

   We can do this as **exercise or as demo**. Note that this example requires ``mpi4py`` and a
   MPI installation such as for instance `OpenMPI <https://www.open-mpi.org/>`__.

   - Try to run this example on one core: ``$ python example.py``.
   - Then compare the output with a run on multiple cores (in this case 2): ``$ mpiexec -n 2 python example.py``.
   - Can you guess what the ``comm.gather`` function does by looking at the print-outs right before and after.
   - Why do we have the if-statement ``if rank == 0`` at the end?
   - Why did we use ``_, n_inside_circle = sample(n_task)`` and not ``n, n_inside_circle = sample(n_task)``?


.. exercise:: Profile the word-autocorrelation code

   Use what you learned in an earlier episode to perform line profiling on the word-autocorrelation code!

   .. solution:: 

      WRITEME

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
- `Parallel programming in Python: mpi4py <https://www.kth.se/blogs/pdc/2019/08/parallel-programming-in-python-mpi4py-part-1/>`__






.. keypoints::

   - 1
   - 2
   - 3