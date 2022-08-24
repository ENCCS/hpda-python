.. _performance:

Profiling and optimising
========================

.. objectives::

   - Learn how to benchmark and profile Python code
   - Understand how optimisation can be algorithmic or based on CPU or memory usage

.. instructor-note::

   - 20 min teaching/type-along
   - 20 min exercises


Once your code is working reliably, you can start thinking of optimising it.


.. warning::

   Always measure the code before you start optimization. Don't base your optimization 
   on theoretical consideration, otherwise you'll have surprises. 


Profilers 
---------

time
^^^^

One of the easy way to profile the program is to use the time function:

.. code-block:: python

   import time
   # start the timer
   start_time=time.time()
   # here are the code you would like to profile
   a = np.arange(1000)
   a = a ** 2
   # stop the timer
   end_time=time.time()
   print("Runtime: {:.4f} seconds".format(end_time - start_time))
   # Runtime: 0.0001 seconds


Timeit
^^^^^^

If you're using a Jupyter notebook, the best choice will be to use 
`%timeit <https://docs.python.org/library/timeit.html>`__ to time a small piece of code:

.. code-block:: ipython

   import numpy as np

   a = np.arange(1000)

   %timeit a ** 2
   # 1.4 µs ± 25.1 ns per loop 

One can also use the cell magic ``%%timeit`` to benchmark a full cell.

.. note::

   For long running calls, using ``%time`` instead of ``%timeit``; it is
   less precise but faster


cProfile
^^^^^^^^

For more complex code, one can use the `built-in python profilers 
<https://docs.python.org/3/library/profile.html>`_, ``cProfile`` or ``profile``.

As a demo, let us consider the following code which simulates a random walk in one dimension
(we can save it as ``walk.py`` or download from :download:`here <example/walk.py>`):

.. literalinclude:: example/walk.py

We can profile it with ``cProfile``:

.. code-block:: console

   $  python -m cProfile -s time walk.py


The ``-s`` switch sorts the results by ``time``. Other options include 
e.g. function name, cumulative time, etc. However, this will print a lot of 
output which is difficult to read. 

.. code-block:: console

   $ python -m cProfile -o walk.prof walk.py


It's also possible to write the profile 
to a file with the ``-o`` flag and view it with `profile pstats module 
<https://docs.python.org/3/library/profile.html#module-pstats>`__
or profile visualisation tools like 
`Snakeviz <https://jiffyclub.github.io/snakeviz/>`__ 
or `profile-viewer <https://pypi.org/project/profile-viewer/>`__.

.. note::

   Similar functionality is available in interactive IPython or Jupyter sessions with the 
   magic command `%%prun <https://ipython.readthedocs.io/en/stable/interactive/magics.html>`__.


Line-profiler
^^^^^^^^^^^^^

The cProfile tool tells us which function takes most of the time but it does not give us a 
line-by-line breakdown of where time is being spent. For this information, we can use the 
`line_profiler <https://github.com/pyutils/line_profiler/>`__ tool. 

.. demo:: Demo: line profiling

   For line-profiling source files from the command line, we can add a decorator ``@profile`` 
   to the functions of interests. If we do this for the :meth:`step` and :meth:`walk` function 
   in the example above, we can then run the script using the `kernprof.py` program which comes with 
   ``line_profiler``, making sure to include the switches ``-l, --line-by-line`` and ``-v, --view``:

   .. code-block:: console

       $ kernprof -l -v walk.py

   ``line_profiler`` also works in a Jupyter notebook. First one needs to load the extension:

   .. code-block:: ipython

      %load_ext line_profiler

   If the :meth:`walk` and :meth:`step` functions are defined in code cells, we can get the line-profiling 
   information by:

   .. code-block:: ipython

      %lprun -f walk -f step walk(10000)


   - Based on the output, can you spot a mistake which is affecting performance?

   .. solution:: Line-profiling output

      .. code-block:: console

         Wrote profile results to walk.py.lprof
         Timer unit: 1e-06 s

         Total time: 0.113249 s
         File: walk.py
         Function: step at line 4

         Line #      Hits         Time  Per Hit   % Time  Line Contents
         ==============================================================
            4                                           @profile
            5                                           def step():
            6     99999      57528.0      0.6     50.8      import random
            7     99999      55721.0      0.6     49.2      return 1. if random.random() > .5 else -1.

         Total time: 0.598811 s
         File: walk.py
         Function: walk at line 9

         Line #      Hits         Time  Per Hit   % Time  Line Contents
         ==============================================================
            9                                           @profile
            10                                           def walk(n):
            11         1         20.0     20.0      0.0      x = np.zeros(n)
            12         1          1.0      1.0      0.0      dx = 1. / n
            13    100000      44279.0      0.4      7.4      for i in range(n - 1):
            14     99999     433303.0      4.3     72.4          x_new = x[i] + dx * step()
            15     99999      53894.0      0.5      9.0          if x_new > 5e-3:
            16                                                       x[i + 1] = 0.
            17                                                   else:
            18     99999      67313.0      0.7     11.2              x[i + 1] = x_new
            19         1          1.0      1.0      0.0      return x

   .. solution:: The mistake

      The mistake is that the ``random`` module is loaded inside the :meth:`step` function
      which is called thousands of times! Moving the module import to the top level saves 
      considerable time.

Performance optimization 
------------------------

Once we have identified the bottlenecks, we need to make the corresponding code go faster.

Algorithm optimization
^^^^^^^^^^^^^^^^^^^^^^

The first thing to look into is the underlying algorithm you chose: is it optimal?
To answer this question, a good understanding of the maths behind the algorithm helps. 
For certain algorithms, many of the bottlenecks will be linear 
algebra computations. In these cases, using the right function to solve 
the right problem is key. For instance, an eigenvalue problem with a 
symmetric matrix is much easier to solve than with a general matrix. Moreover, 
most often, you can avoid inverting a matrix and use a less costly 
(and more numerically stable) operation. However, it can be as simple as 
moving computation or memory allocation outside a loop, and this happens very often as well.

Singular Value Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Singular Value Decomposition <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_ (SVD)
is quite often used in climate model data analysis.  The computational cost of this algorithm is 
roughly :math:`n^3` where  :math:`n` is the size of the input matrix. 
However, in most cases, we are not using all the output of the SVD, 
but only the first few rows of its first returned argument. If
we use the ``svd`` implementation from SciPy, we can ask for an incomplete
version of the SVD. Note that implementations of linear algebra in
SciPy are richer then those in NumPy and should be preferred.
The following example demonstrates the performance benefit for a "slim" array
(i.e. much larger along one axis):

.. sourcecode:: ipython

   import numpy as np
   data = np.random.random((4000,100))

   %timeit np.linalg.svd(data)
   # 1.09 s ± 19.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   from scipy import linalg

   %timeit linalg.svd(data)
   # 1.03 s ± 24.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   %timeit linalg.svd(data, full_matrices=False)
   # 21.2 ms ± 716 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

   %timeit np.linalg.svd(data, full_matrices=False)
   # 23.8 ms ± 3.06 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

CPU usage optimization
^^^^^^^^^^^^^^^^^^^^^^

Vectorization
~~~~~~~~~~~~~

Arithmetic is one place where NumPy performance outperforms python list and 
the reason is that it uses vectorization. A lot of the data analysis involves 
a simple operation being applied to each element of a large dataset. 
In such cases, vectorization is key for better performance. 
In practice, a vectorised operation means reframing the code in a manner that
completely avoids a loop and instead uses e.g. slicing to apply the operation
on the whole array (slice) at one go. For example, the following code for 
calculating the difference of neighbouring elements in an array:


Consider the following code:

.. code-block:: ipython

   %%timeit

   import numpy as np
   a = np.arange(1000)
   a_dif = np.zeros(999, np.int64)
   for i in range(1, len(a)):
       a_dif[i-1] = a[i] - a[i-1]

   # 564 µs ± 25.2 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

How can the ``for`` loop be vectorized? We need to use clever indexing to get rid of the 
loop:

.. code-block:: ipython

   %%timeit

   import numpy as np
   a = np.arange(1000)
   a_dif = a[1:] - a[:-1]

   # 2.12 µs ± 25.8 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

The first brute force approach using a for loop is much slower than the second vectorised form!

So one should consider using *vectorized* operations whenever possible, not only for 
performance but also because the vectorized version can be more convenient. 

What if we have a function that only take scalar values as input, but we want to apply it 
element-by-element on an array? We can vectorize the function!  
Let's define a simple function ``f`` which takes scalars input: 

.. code-block:: python

   import math
   def f(x, y):
       return math.pow(x,3.0) + 4*math.sin(y) 

If we pass an array we get an error 
   
.. code-block:: python

   x = np.ones(10000, dtype=np.int8)
   f(x,x)
   
   # Traceback (most recent call last):
   #   File "<stdin>", line 1, in <module>
   #   File "<stdin>", line 2, in f
   # TypeError: only size-1 arrays can be converted to Python scalars

We could loop over the array:

.. code-block:: ipython

   %%timeit 
   for i in x:
       f(i,i)

   # 49.9 ms ± 3.84 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

However, in order to pass a NumPy array it is better to vectorize the function using :meth:`np.vectorize`
which takes a nested sequence of objects or NumPy arrays as inputs and returns a single 
NumPy array or a tuple of NumPy arrays:

.. code-block:: ipython

   import numpy as np
   import math

   def f(x, y):
       return math.pow(x,3.0) + 4*math.sin(y) 

   f_numpy = np.vectorize(f)

   # benchmark
   x = np.ones(10000, dtype=np.int8)
   %timeit f_numpy(x,x)
   # 4.84 ms ± 75.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


For high performance vectorization, another choice is to use Numba. 
Adding the decorator in a function, Numba will figure out the rest for you:

.. code-block:: ipython

   import numba
   import math

   def f(x, y):
       return math.pow(x,3.0) + 4*math.sin(y) 

   f_numba = numba.vectorize(f)

   # benchmark
   x = np.ones(10000, dtype=np.int8)
   %timeit f_numba(x,x)

   # 89.2 µs ± 1.74 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

We will learn more about Numba in the next episode.

Memory usage optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

Broadcasting
~~~~~~~~~~~~

Basic operations of NumPy are elementwise, and the shape of the arrays should be compatible.
However, in practice under certain conditions, it is possible to do operations on arrays of different shapes.
NumPy expands the arrays such that the operation becomes viable.

.. note:: Broadcasting Rules  

  - Dimensions match when they are equal, or when either is 1 or None.   
  - In the latter case, the dimension of the output array is expanded to the larger of the two.
  - Broadcasted arrays are never physically constructed, which saves memory.


.. challenge:: Broadcasting

   .. tabs:: 

      .. tab:: 1D

         .. code-block:: python

            import numpy as np
            a = np.array([1, 2, 3])
            b = 4 
            a + b


         .. figure:: img/bc_1d.svg 


      .. tab:: 2D

         .. code-block:: python

            import numpy as np
            a = np.array([[0, 0, 0],[10, 10, 10],[20, 20, 20],[30, 30, 30]])
            b = np.array([1, 2, 3])
            a + b                      


         .. figure:: img/bc_2d_1.svg 


         .. code-block:: python

            import numpy as np
            a = np.array([0, 10, 20, 30])
            b = np.array([1, 2, 3]) 
            a + b # this does not work
            a[:,None] +b 
            # or
            a[:,np.newaxis] +b
                 

         .. figure:: img/bc_2d_2.svg 




Cache effects
~~~~~~~~~~~~~

Memory access is cheaper when it is grouped: accessing a big array in a 
continuous way is much faster than random access. This implies amongst 
other things that **smaller strides are faster**:

.. code-block:: python

   c = np.zeros((10000, 10000), order='C')

   %timeit c.sum(axis=0)
   # 1 loops, best of 3: 3.89 s per loop

   %timeit c.sum(axis=1)
   # 1 loops, best of 3: 188 ms per loop

   c.strides
   # (80000, 8)

This is the reason why Fortran ordering or C ordering may make a big
difference on operations.


Temporary arrays
~~~~~~~~~~~~~~~~

- In complex expressions, NumPy stores intermediate values in
  temporary arrays
- Memory consumption can be higher than expected

.. code-block:: python

   a = np.random.random((1024, 1024, 50))
   b = np.random.random((1024, 1024, 50))
   
   # two temporary arrays will be created
   c = 2.0 * a - 4.5 * b
   
   # four temporary arrays will be created, and from which two are due to unnecessary parenthesis
   c = (2.0 * a - 4.5 * b) + (np.sin(a) + np.cos(b))

   # solution
   # apply the operation one by one for really large arrays
   c = 2.0 * a
   c = c - 4.5 * b
   c = c + np.sin(a)
   c = c + np.cos(b)

- Broadcasting approaches can lead also to hidden temporary arrays  

  - Input data M x 3 array
  - Output data M x M array 
  - There is a temporary M x M x 3 array

.. code-block:: python

   import numpy as np
   X = np.random.random((M, 3))
   D = np.sqrt(((X[:, np.newaxis, :] - X) ** 2).sum(axis=-1))


Numexpr
~~~~~~~

- Evaluation of complex expressions with one operation at a time can lead
  also into suboptimal performance
    
    - Effectively, one carries out multiple *for* loops in the NumPy C-code

- Numexpr package provides fast evaluation of array expressions

.. code-block:: ipython

   import numexpr as ne
   x = np.random.random((10000000, 1))
   y = np.random.random((10000000, 1))
   %timeit y = ((.25*x + .75)*x - 1.5)*x - 2
   %timeit y = ne.evaluate("((.25*x + .75)*x - 1.5)*x - 2")

- By default, Numexpr tries to use multiple threads
- Number of threads can be queried and set with
  ``numexpr.set_num_threads(nthreads)``
- Supported operators and functions:
  +,-,\*,/,\*\*, sin, cos, tan, exp, log, sqrt
- Speedups in comparison to NumPy are typically between 0.95 and 4
- Works best on arrays that do not fit in CPU cache






.. keypoints::

   - Measure and benchmark before you start optimizing
   - Optimization can be to change algorithms, optimise memory usage or add
     vectorization, or to convert performance-critical functions to Numba or Cython
