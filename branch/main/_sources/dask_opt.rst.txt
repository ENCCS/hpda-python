Dask (II)
=========

.. challenge:: Testing different schedulers

   We will test different schedulers and compare the performance on a simple task calculating 
   the mean of a random generated array.
   
   Here is the code using NumPy:

   .. literalinclude:: example/dask_gil.py
      :language: ipython
      :lines: 1-7

   Here we run the same code using different schedulers from Dask:

   .. tabs::

      .. tab::  ``serial``

	 .. literalinclude:: example/dask_gil.py
            :language: ipython
            :lines: 9-12

      .. tab::  ``threads``

	 .. literalinclude:: example/dask_gil_threads.py
            :language: ipython
            :lines: 1-10

	 .. literalinclude:: example/dask_gil_threads.py
            :language: ipython
            :lines: 12-15

	 .. literalinclude:: example/dask_gil_threads.py
            :language: ipython
            :lines: 17-20

	 .. literalinclude:: example/dask_gil_threads.py
            :language: ipython
            :lines: 22-25

      .. tab::  ``processes``

	 .. literalinclude:: example/dask_gil_processes.py
            :language: ipython
            :lines: 1-10

	 .. literalinclude:: example/dask_gil_processes.py
            :language: ipython
            :lines: 12-15

	 .. literalinclude:: example/dask_gil_processes.py
            :language: ipython
            :lines: 17-20

	 .. literalinclude:: example/dask_gil_processes.py
            :language: ipython
            :lines: 22-25

      .. tab::  ``distributed``

	 .. literalinclude:: example/dask_gil_distributed.py
            :language: ipython
            :lines: 1-14

	 .. literalinclude:: example/dask_gil_distributed.py
            :language: ipython
            :lines: 16-17

	 .. literalinclude:: example/dask_gil_distributed.py
            :language: ipython
            :lines: 19-21

	 .. literalinclude:: example/dask_gil_distributed.py
            :language: ipython
            :lines: 23-25

	 .. literalinclude:: example/dask_gil_distributed.py
            :language: ipython
            :lines: 27



   .. solution:: Testing different schedulers

      Comparing profiling from mt_1, mt_2 and mt_4: Using ``threads`` scheduler is limited by the GIL on pure Python code. 
      In our case, although it is not a pure Python function, it is still limited by GIL, therefore no multi-core speedup

      Comparing profiling from mt_1, mp_1 and dis_1: Except for ``threads``, the other two schedulers copy data between processes 
      and this can introduce performance penalties, particularly when the data being transferred between processes is large.

      Comparing profiling from serial, mt_1, mp_1 and dis_1: Creating and destroying threads and processes have overheads,
      ``processes`` have even more overhead than ``threads``

      Comparing profiling from mp_1, mp_2 and mp_4: Running multiple processes is only effective when there is enough computational 
      work to do i.e. CPU-bound tasks. In this very example, most of the time is actually spent on transferring the data 
      rather than computing the mean

      Comparing profiling from ``processes`` and ``distributed``: Using ``distributed`` scheduler has advantages over ``processes``, 
      this is related to better handling of data copying, i.e. ``processes`` scheduler copies data for every task, while 
      ``distributed`` scheduler copies data for each worker.



.. challenge:: SVD with large skinny matrix using ``distributed`` scheduler

   We can use dask to compute SVD of a large matrix which does not fit into the memory of a 
   normal laptop/desktop. While it is computing, you should switch to the Dask dashboard and 
   watch column "Workers" and "Graph", so you must run this using ``distributed`` scheduler

   .. code-block:: python

       import dask
       import dask.array as da
       X = da.random.random((2000000, 100), chunks=(10000, 100))
       X
       u, s, v = da.linalg.svd(X)
       dask.visualize(u, s, v)
       s.compute()


   SVD is only supported for arrays with chunking in one dimension, which requires that the matrix
   is either *tall-and-skinny* or *short-and-fat*.
   If chunking in both dimensions is needed, one should use approximate algorithm.

   .. code-block:: python

       import dask
       import dask.array as da
       X = da.random.random((10000, 10000), chunks=(2000, 2000))
       u, s, v = da.linalg.svd_compressed(X, k=5)
       dask.visualize(u, s, v)
       s.compute()


.. callout:: Memory management

   You may observe that there are different memory categories showing on the dashboard:

   - process: Overall memory used by the worker process, as measured by the OS
   - managed: Size of data that Dask holds in RAM, but most probably inaccurate, excluding spilled data.
   - unmanaged: Memory that Dask is not directly aware of, this can be e.g. Python modules,
     temporary arrays, memory leasks, memory not yet free()'d by the Python memory manager to the OS
   - unmanaged recent: Unmanaged memory that has appeared within the last 30 seconds whch is not included 
     in the "unmanaged" memory measure
   - spilled: Memory spilled to disk

   The sum of managed + unmanaged + unmanaged recent is equal by definition to the process memory.

   When the managed memory exceeds 60% of the memory limit (target threshold), 
   the worker will begin to dump the least recently used data to disk. 
   Above 70% of the target memory usage based on process memory measurment (spill threshold), 
   the worker will start dumping unused data to disk.
         
   At 80% process memory load, currently executing tasks continue to run, but no additional tasks 
   in the worker's queue will be started.

   At 95% process memory load (terminate threshold), all workers will be terminated. Tasks will be cancelled 
   as well and data on the worker will be lost and need to be recomputed.
