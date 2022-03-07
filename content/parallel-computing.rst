Parallel computing
==================

.. objectives::

   - Understand the Global Interpreter Lock in Python
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
  - Running multiple threads is *only effective for I/O-bound tasks*

- **Distributed memory parallelism (multiprocessing):** Different processes manage their own memory segments and 
  share data by communicating (passing messages) as needed.

  - A process can contain one or more threads
  - Two processes can run on different CPU cores and different computers
  - Processes have more overhead than threads (creating and destroying processes takes more time)
  - Running multiple processes is *only effective for CPU-bound tasks*

In the next episode we will look at `Dask <https://dask.org/>`__, an array model extension and task scheduler, 
which goes beyond these three approaches.


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

Multithreading
--------------

TODO: write about how to multithread I/O

Multiprocessing
---------------

- https://aaltoscicomp.github.io/python-for-scicomp/parallel/#multiprocessing
- https://www.kth.se/blogs/pdc/2019/02/parallel-programming-in-python-multiprocessing-part-1/


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

.. challenge:: Using MPI

   We can do this as **exercise or as demo**. Note that this example requires ``mpi4py`` and a
   MPI installation such as for instance `OpenMPI <https://www.open-mpi.org/>`__.

   - Try to run this example on one core: ``$ python example.py``.
   - Then compare the output with a run on multiple cores (in this case 2): ``$ mpiexec -n 2 python example.py``.
   - Can you guess what the ``comm.gather`` function does by looking at the print-outs right before and after.
   - Why do we have the if-statement ``if rank == 0`` at the end?
   - Why did we use ``_, n_inside_circle = sample(n_task)`` and not ``n, n_inside_circle = sample(n_task)``?




See also
--------

- `More on the global interpreter lock
  <https://wiki.python.org/moin/GlobalInterpreterLock>`__
- `Parallel programming in Python: multiprocessing <https://www.kth.se/blogs/pdc/2019/02/parallel-programming-in-python-multiprocessing-part-1/>`__
- `Parallel programming in Python: mpi4py <https://www.kth.se/blogs/pdc/2019/08/parallel-programming-in-python-mpi4py-part-1/>`__






.. keypoints::

   - 1
   - 2
   - 3