.. _performance:

Profiling and optimising
========================

.. objectives::

   - Learn how to benchmark and profile Python code
   - Understand how optimisation can be algorithmic or based on CPU or memory usage
   - Learn how to boost performance using Numba and Cython

.. instructor-note::

   - 40 min teaching/type-along
   - 40 min exercises


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
   print("Runtime: {} seconds".format(round(end_time - start_time, 2)))



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


we use the ``-s`` switch to sort the results by ``time``, other options include 
e.g. function name, cummulative time, etc. However, this will print a lot of 
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

   a = numpy.random.random((1024, 1024, 50))
   b = numpy.random.random((1024, 1024, 50))
   
   # two temporary arrays will be created
   c = 2.0 * a - 4.5 * b
   
   # four temporary arrays will be created, and from which two are due to unnecessary parenthesis
   c = (2.0 * a - 4.5 * b) + (numpy.sin(a) + numpy.cos(b))

   # solution
   # apply the operation one by one for really large arrays
   c = 2.0 * a
   c = c - 4.5 * b
   c = c + numpy.sin(a)
   c = c + numpy.cos(b)

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
   x = numpy.random.random((10000000, 1))
   y = numpy.random.random((10000000, 1))
   %timeit y = ((.25*x + .75)*x - 1.5)*x - 2
   %timeit y = ne.evaluate("((.25*x + .75)*x - 1.5)*x - 2")

- By default, Numexpr tries to use multiple threads
- Number of threads can be queried and set with
  ``numexpr.set_num_threads(nthreads)``
- Supported operators and functions:
  +,-,\*,/,\*\*, sin, cos, tan, exp, log, sqrt
- Speedups in comparison to NumPy are typically between 0.95 and 4
- Works best on arrays that do not fit in CPU cache




Performance boosting
--------------------

For many user cases, using NumPy or Pandas is sufficient. However, in some computationally heavy applications, 
it is possible to improve the performance by pre-compiling expensive functions.
`Cython <https://cython.org/>`__ and `Numba <https://numba.pydata.org/>`__ 
are among the popular choices and both of them have good support for NumPy arrays. 


Cython
^^^^^^

Cython is a superset of Python that additionally supports calling C functions and 
declaring C types on variables and class attributes. Under Cython, source code gets 
translated into optimized C/C++ code and compiled as Python extension modules. 

Developers can run the ``cython`` command-line utility to produce a ``.c`` file from 
a ``.py`` file which needs to be compiled with a C compiler to an ``.so`` library 
which can then be directly imported in a Python program. There is, however, also an easy 
way to use Cython directly from Jupyter notebooks through the ``%%cython`` magic 
command. We will restrict the discussion here to the Jupyter-way. For a full overview 
of the capabilities refer to the `documentation <https://cython.readthedocs.io/en/latest/>`__.


.. demo:: Demo: Cython

   Consider the following pure Python code which integrates a function:

   .. literalinclude:: example/integrate_python.py 

   We generate a dataframe and apply the :meth:`apply_integrate_f` function on its columns, timing the execution:

   .. code-block:: ipython

      import pandas as pd

      df = pd.DataFrame({"a": np.random.randn(1000),
                        "b": np.random.randn(1000),
                        "N": np.random.randint(100, 1000, (1000))})                

      %timeit apply_integrate_f(df['a'], df['b'], df['N'])
      # 321 ms ± 10.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   Now import the Cython extension:

   .. code-block:: ipython

      %load_ext cython

   As a first cythonization step we add the cython magic command with the 
   ``-a, --annotate`` flag, ``%%cython -a``, to the top of the Jupyter code cell.
   The yellow coloring in the output shows us the amount of pure Python:

   .. figure:: img/cython_annotate.png
       
   Our task is to remove as much yellow as possible by explicitly declaring variables and functions.
   We can start by simply compiling the code using Cython without any changes:

   .. literalinclude:: example/integrate_cython.py 

   .. code-block:: ipython

      %timeit apply_integrate_f_cython(df['a'], df['b'], df['N'])
      # 276 ms ± 20.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   Simply by using Cython and a copy-and-paste gives us about 10% increase in performance. 
   Now we can start adding data type annotation to the input variables:

   .. literalinclude:: example/integrate_cython_dtype0.py 
      :emphasize-lines: 6,9,16

   .. code-block:: ipython

      # this will not work
      #%timeit apply_integrate_f_cython_dtype0(df['a'], df['b'], df['N'])
      # but rather 
      %timeit apply_integrate_f_cython_dtype0(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
      # 41.4 ms ± 1.27 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

   .. warning::

      You can not pass a Series directly since the Cython definition is specific to an array. 
      Instead using the ``Series.to_numpy()`` to get the underlying NumPy array
      which works nicely with Cython.

      We use C data types like ``double``, ``long`` to define variables.


   Next step, we can start adding type annotation to the functions.
   There are three ways of declaring functions: 
   
   - ``def`` - Python style:

   Declaring the types of arguments and local types (thus return values) can allow Cython 
   to generate optimised code which speeds up the execution. If the types are declared then 
   a ``TypeError`` will be raised if the function is passed the wrong types.

   - ``cdef`` - C style:

   Cython treats the function as pure C functions. All types **must** be declared. 
   This will give you the best performance but there are a number of consequences. 
   One should really take care of the ``cdef`` declared functions, since you are actually writing in C.

   - ``cpdef`` - Python/C mixed:

   ``cpdef`` functions combine both ``def`` and ``cdef``: one can use ``cdef`` for C types and ``def`` for Python types. 
   In terms of performance, ``cpdef`` functions may be as fast as those using ``cdef`` and 
   might be as slow as ``def`` declared functions.  

   .. literalinclude:: example/integrate_cython_dtype1.py 
      :emphasize-lines: 6,9,16

   .. code-block:: ipython

      %timeit apply_integrate_f_cython_dtype1(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
      # 37.2 ms ± 556 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

   Last step, we can add type annotation to the local variables within the functions and output.

   .. literalinclude:: example/integrate_cython_dtype2.py 
      :emphasize-lines: 6,9,10,11,16,20,21

   .. code-block:: ipython

      %timeit apply_integrate_f_cython_dtype2(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
      # 696 µs ± 8.71 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
   
   Now it is over 400 times faster than the original Python implementation, and all we have done is to add 
   type declarations! If we add the ``-a`` annotation flag we indeed see much less Python interaction in the 
   code.

   .. figure:: img/cython_annotate_2.png



Numba
^^^^^

An alternative to statically compiling Cython code is to use a dynamic just-in-time (JIT) compiler with `Numba <https://numba.pydata.org/>`__. 
Numba allows you to write a pure Python function which can be JIT compiled to native machine instructions, 
similar in performance to C, C++ and Fortran, by simply adding the decorator ``@jit`` in your function. 
However, the ``@jit`` compilation will add overhead to the runtime of the function, 
i.e. the first time a function is run using Numba engine will be slow as Numba will have the function compiled. 
Once the function is JIT compiled and cached, subsequent calls will be fast. So the performance benefits may not be 
realized especially when using small datasets.

Numba supports compilation of Python to run on either CPU or GPU hardware and is designed to integrate with 
the Python scientific software stack. The optimized machine code is generated by the LLVM compiler infrastructure.


.. demo:: Demo: Numba

   Consider the integration example again using Numba this time:

   .. literalinclude:: example/integrate_numba.py 

   .. code-block:: ipython

      # try passing Pandas Series 
      %timeit apply_integrate_f_numba(df['a'],df['b'],df['N'])
      # 6.02 ms ± 56.5 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
      # try passing NumPy array
      %timeit apply_integrate_f_numba(df['a'].to_numpy(),df['b'].to_numpy(),df['N'].to_numpy())
      # 625 µs ± 697 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


   .. warning:: 
   
      Numba is best at accelerating functions that apply numerical functions to NumPy arrays. When used with Pandas, 
      pass the underlying NumPy array of :class:`Series` or :class:`DataFrame` (using ``to_numpy()``) into the function.
      If you try to @jit a function that contains unsupported Python or NumPy code, compilation will fall back to the object mode 
      which will mostly likely be very slow. If you would prefer that Numba throw an error for such a case, 
      you can do e.g. ``@numba.jit(nopython=True)`` or ``@numba.njit``. 


   We can further add date type, although in this case there is not much performance improvement:

   .. literalinclude:: example/integrate_numba_dtype.py 

   .. code-block:: ipython

      %timeit apply_integrate_f_numba_dtype(df['a'].to_numpy(),df['b'].to_numpy(),df['N'].to_numpy())
      # 625 µs ± 697 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)



Exercises
---------

.. exercise:: Profile the word-autocorrelation code

   Revisit the word-autocorrelation code. To clone the repository (if you haven't already):

   .. code-block:: console

      $ git clone https://github.com/ENCCS/word-count-hpda.git
   
   To run the code, type:

   .. code-block:: console

      $ python source/autocorrelation.py data/pg99.txt processed_data/pg99.dat results/acf_pg99.dat

   Add ``@profile`` to the :meth:`word_acf` function, and run ``kernprof.py`` (or just ``kernprof``) 
   from the command line. What lines of this function are the most expensive?

   .. solution:: 

      .. code-block:: console

         $ kernprof -l -v source/autocorrelation.py data/pg99.txt processed_data/pg99.dat

      Output: 

      .. code-block:: text

         Wrote profile results to autocorrelation.py.lprof
         Timer unit: 1e-06 s
         
         Total time: 15.5976 s
         File: source/autocorrelation.py
         Function: word_acf at line 24
         
         Line #      Hits         Time  Per Hit   % Time  Line Contents
         ==============================================================
             24                                           @profile
             25                                           def word_acf(word, text, timesteps):
             26                                               """
             27                                               Calculate word-autocorrelation function for given word 
             28                                               in a text. Each word in the text corresponds to one "timestep".
             29                                               """
             30        10       1190.0    119.0      0.0      acf = np.zeros((timesteps,))
             31        10      15722.0   1572.2      0.1      mask = [w==word for w in text]
             32        10       6072.0    607.2      0.0      nwords_chosen = np.sum(mask)
             33        10         14.0      1.4      0.0      nwords_total = len(text)
             34      1010        658.0      0.7      0.0      for t in range(timesteps):
             35  11373500    4675124.0      0.4     30.0          for i in range(1,nwords_total-t):
             36  11372500   10897305.0      1.0     69.9              acf[t] += mask[i]*mask[i+t]
             37      1000       1542.0      1.5      0.0          acf[t] /= nwords_chosen      
             38        10         10.0      1.0      0.0      return acf
         

.. exercise:: Is the :meth:`word_acf` function efficient?

   Have another look at the :meth:`word_acf` function from the word-count project. 

   .. literalinclude:: exercise/autocorrelation.py
      :pyobject: word_acf
      
   Do you think there is any room for improvement? How would you go about optimizing 
   this function?

   .. solution:: 

      The function uses a Python object (``mask``) inside a double for-loop, 
      which is guaranteed to be suboptimal. There are a number of ways to speed 
      it up. One is to use ``numba`` and just-in-time compilation, as we shall 
      see below. 

      Another is to find an in-built vectorized NumPy function which can calculate the 
      autocorrelation for us! Here's one way to do it:

      .. literalinclude:: exercise/autocorrelation_numba_numpy.py
         :pyobject: word_acf_numpy


.. exercise:: Pairwise distance

   Consider the following Python function:

   .. literalinclude:: example/dis_python.py

   Start by profiling it in Jupyter:

   .. code-block:: ipython

      X = np.random.random((1000, 3))
      %timeit dis_python(X)

   Now try to speed it up with NumPy (i.e. *vectorise* the function),
   Numba or Cython (depending on what you find most interesting).
   Make sure that you're getting the correct result, and then benchmark it 
   with ``%timeit``.

   .. solution::

      .. tabs:: 
   
         .. tab:: NumPy
   
                .. literalinclude:: example/dis_numpy.py 

                .. code-block:: ipython

                   X = np.random.random((1000, 3))
                   %timeit dis_numpy(X)

   
         .. tab:: Cython
   
                .. literalinclude:: example/dis_cython.py 

                .. code-block:: ipython

                   X = np.random.random((1000, 3))
                   %timeit dis_cython(X)
   
         .. tab:: Numba
   
                .. literalinclude:: example/dis_numba.py 

                .. code-block:: ipython

                   X = np.random.random((1000, 3))
                   %timeit dis_numba(X)


.. exercise:: Bubble sort

   To make a long story short, in the worse case the time taken by the Bubblesort algorithm is 
   roughly :math:`O(n^2)` where  :math:`n` is the number of items being sorted. 

   .. image:: img/Bubble-sort-example-300px.gif

   Here is a function that performs bubble-sort:

   .. literalinclude:: example/bs_python.py 

   And this is how you can benchmark it:

   .. code-block:: ipython

      import random
      l = [random.randint(1,1000) for num in range(1, 1000)]
      %timeit bs_python(l)

   Now try to speed it up with Numba or Cython (depending on what you find 
   most interesting). Make sure that you're getting the correct result, 
   and then benchmark it with ``%timeit``.

   .. solution:: 
   
      .. tabs:: 

         .. tab:: Cython

                .. literalinclude:: example/bs_cython.py 

                .. code-block:: ipython

                   import random
                   l = [random.randint(1,1000) for num in range(1, 1000)]
                   l_arr = np.asarray(l)
                   %timeit bs_cython(l_arr)

             
                We can further improve performance by using more C/C++ features: 

                .. literalinclude:: example/bs_cython_adv.py 

                .. code-block:: ipython

                   import random
                   l = [random.randint(1,1000) for num in range(1, 1000)]
                   %timeit bs_clist(l)


         .. tab:: Numba

                .. literalinclude:: example/bs_numba.py 

                .. code-block:: ipython

                   import random
                   l = [random.randint(1,1000) for num in range(1, 1000)]
                   # first try using a list as input
                   %timeit bs_numba(l)
                   # try using a NumPy array
                   l_arr = np.asarray(l)
                   %timeit bs_numba(l_arr)




.. note::

   Note that the results depend on what version of Python, Cython, Numba, and NumPy you are using. 
   In addition, the different compiler choices used for installing NumPy can account for differences in the results.
   
   NumPy is really good at what it does. For simple operations or small data, Numba is not going to outperform it, 
   but when things get more complex Numba will save the day. 
