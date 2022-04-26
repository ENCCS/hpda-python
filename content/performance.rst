.. _performance:

Optimization
============

.. objectives::

   - Learn how to benchmark and profile Python code
   - Understand how optimisation can be algorithmic or based on CPU or memory usage
   - Learn how to boost performance using Numba and Cython

.. instructor-note::

   - 40 min teaching/type-along
   - 40 min exercises


Once your code is working reliably, you can start thinking of optimizing it.


.. warning::

   Always measure the code before you start optimization. Don't base your optimization 
   on theoretical consideration, otherwise you'll have surprises. 


Profilers 
---------

Timeit
^^^^^^

If you're using a Jupyter notebook, the best choice will be to use 
`timeit <https://docs.python.org/library/timeit.html>`__ to time a small piece of code:

.. code-block:: ipython

   In [1]: import numpy as np

   In [2]: a = np.arange(1000)

   In [3]: %timeit a ** 2
   100000 loops, best of 3: 5.73 us per loop

.. note::

   For long running calls, using ``%time`` instead of ``%timeit``; it is
   less precise but faster


cProfile
^^^^^^^^

For more complex code, one can use the `built-in python profilers 
<https://docs.python.org/3/library/profile.html>`_, ``cProfile`` or ``profile``.

As a demo, let us consider the following code which simulates a random walk in one dimension
(we can save it as ``walk.py``):

.. code-block:: python

   import numpy as np

   def step():
       import random
       return 1. if random.random() > .5 else -1.
   
   def walk(n):
       x = np.zeros(n)
       dx = 1. / n
       for i in range(n - 1):
           x_new = x[i] + dx * step()
           if x_new > 5e-3:
               x[i + 1] = 0.
           else:
               x[i + 1] = x_new
       return x

   if __name__ == "__main__":
       n = 100000
       x = walk(n)

We can profile it with ``cProfile``:

.. code-block:: console

   $  python -m cProfile -s time walk.py


we use the ``-s`` switch to sort the results by ``time``, other options include 
e.g. function name, cummulative time, etc. However, this will print a lot of 
output which is difficult to read. 

.. code-block:: console

   $  python -m cProfile -o walk.prof walk.py


It's also possible to write the profile 
to a file with the ``-o`` flag and view it with `profile pstats module 
<https://docs.python.org/3/library/profile.html#module-pstats>`__
or profile visualisation tools like 
`Snakeviz <https://jiffyclub.github.io/snakeviz/>`__ 
or `profile-viewer <https://pypi.org/project/profile-viewer/>`__.

Similar functionality is available in interactive IPython or Jupyter sessions with the 
magic command `%%prun <https://ipython.readthedocs.io/en/stable/interactive/magics.html>`__.


Line-profiler
^^^^^^^^^^^^^

The cProfile tool tells us which function takes most of the time but it does not give us a 
line-by-line breakdown of where time is being spent. For this information, we can use the 
`line_profiler <https://github.com/pyutils/line_profiler/>`__ tool. 

.. demo:: Line profiling

   For line-profiling source files from the command line, we can add a decorator ``@profile`` 
   to the functions of interests. If we do this for the :meth:`step` and :meth:`walk` function 
   in the example above, we can then run the script using the `kernprof.py` program which comes with 
   ``line_profiler``, making sure to include the switches ``-l, --line-by-line`` and ``-v, --view``:

   .. code-block:: console

       $ kernprof.py -l -v walk.py

   ``line_profiler`` also works in a Jupyter notebook. First one needs to load the extension:

   .. code-block:: ipython

      In [1]: %load_ext line_profiler

   If the :meth:`walk` and :meth:`step` functions are defined in code cells, we can get the line-profiling 
   information by:

   .. code-block:: ipython

      In [2]: %lprun -f walk -f step walk(10000)


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

.. exercise:: Profile the word-autocorrelation code

   Revisit the word-autocorrelation code. Add ``@profile`` to the :meth:`word_autocorr` and 
   :meth:`word_autocorr_average` function, and run ``kernprof.py`` from the command line.

   .. solution:: autocorrelation.py



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
we use the ``svd`` implementation from scipy, we can ask for an incomplete
version of the SVD. Note that implementations of linear algebra in
scipy are richer then those in numpy and should be preferred.

.. sourcecode:: ipython

    In [3]: %timeit np.linalg.svd(data)
    1 loops, best of 3: 14.5 s per loop

    In [4]: from scipy import linalg

    In [5]: %timeit linalg.svd(data)
    1 loops, best of 3: 14.2 s per loop

    In [6]: %timeit linalg.svd(data, full_matrices=False)
    1 loops, best of 3: 295 ms per loop

    In [7]: %timeit np.linalg.svd(data, full_matrices=False)
    1 loops, best of 3: 293 ms per loop


CPU usage optimization
^^^^^^^^^^^^^^^^^^^^^^

Vectorization
~~~~~~~~~~~~~

Arithmetic is one place where numpy performance outperforms python list and the reason is that it uses vectorization.
A lot of the data analysis involves a simple operation being applied to each element of a large dataset.
In such cases, vectorization is key for better performance.

.. exercise::  vectorized operation vs for loop 

   Consider the following code:

   .. code-block:: python

      import numpy as np
      a = np.arange(1000)
      a_dif = np.zeros(999, np.int64)
      for i in range(1, len(a)):
            a_dif[i-1] = a[i] - a[i-1]

   Try to vectorize the ``for`` loop!

   .. solution::

      .. code-block:: python

			import numpy as np
         a = np.arange(1000)
			a_dif = a[1:] - a[:-1]

.. exercise:: Profile the word-autocorrelation code

   Use line-profiling on the word-autocorrelation code!

   .. solution:: 

      WRITEME

.. exercise:: Is the :meth:`word_autocorr` function efficient?

   Have another look at the :meth:`word_autocorr` function from the word-count project. 

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
      
   Do you think there is any room for improvement? How would you go about optimizing 
   this function?

   .. solution:: 

      The function uses a Python object (``mask``) inside a double for-loop, 
      which is guaranteed to be suboptimal. There are a number of ways to speed 
      it up. One is to use ``numba`` and just-in-time compilation, as we shall 
      see below. 

      Another is to find an in-built vectorized NumPy function which can calculate the 
      autocorrelation for us! Here's one way to do it:

      .. code-block:: python

         def word_autocorr_numpy(word, text, timesteps):
             """
             Calculate word-autocorrelation function for given word 
             in a text using numpy.correlate function. 
             Each word in the text corresponds to one "timestep".
             """
             acf = np.zeros((timesteps,))
             mask = np.array([w==word for w in text]).astype(np.float64)
             nwords_chosen = np.sum(mask)
             acf = np.correlate(mask, mask, mode='full') / nwords_chosen
             return acf[int(acf.size/2):int(acf.size/2)+timesteps]         


So one should consider use "vectorized" operations whenever possible.
Not only for performance, sometimes the vectorized function is also convenient. 

Let's define a simple function f which takes scalars as input only, 

.. code-block:: python

   import math
   def f(x, y):
       return x**3 + 4*math.sin(y) 

if we pass an array, 
   
.. code-block:: console

   >>> x = np.ones(10000, dtype=np.int8)
   >>> f(x,x)
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
     File "<stdin>", line 2, in f
   TypeError: only size-1 arrays can be converted to Python scalars


In order to pass an numpy array, we could vectorize it.
For universal functions (or ``ufunc`` for short), 
NumPy provides the ``vectorize`` function.

.. code-block:: python

   import numpy as np
   import math

   def f(x, y):
       return x**3 + 4*math.sin(y) 

   f_numpy = np.vectorize(f)

   # benchmark
   x = np.ones(10000, dtype=np.int8)
   %timeit f_numpy(x,x)


.. note:: 
   
   As stated in the NumPy document: 
   The vectorize function is provided primarily for convenience, not for performance. The implementation is essentially a for loop.



For high performance vectorization, one choice is to use Numba. 
Adding the decorator in a function, Numba will figure out the rest for you. 

.. code-block:: python

   import numba
   import math

   def f(x, y):
       return x**3 + 4*math.sin(y) 

   f_numba = numba.vectorize(f)

   # benchmark
   x = np.ones(10000, dtype=np.int8)
   %timeit f_numba(x,x)



Memory usage optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

Broadcasting
~~~~~~~~~~~~

Basic operations of numpy are elementwise, and the shape of the arrays should be compatible.
However, in practice under certain conditions, it is possible to do operations on arrays of different shapes.
NumPy expands the arrays such that the operation becomes viable.

.. note:: Broadcasting Rules  

  - Dimensions match when they are equal, or when either is 1 or None.   
  - In the latter case, the dimension of the output array is expanded to the larger of the two.
  - Broadcasted arrays are never physically constructed, which saves memory.


.. challenge:: broadcasting

   .. tabs:: 

      .. tab:: 1D

             .. code-block:: py

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
			     a = np.array([0, 10, 20,30])
			     b = np.array([1, 2, 3]) 
			     a + b                       # array([[11, 12, 13],
                                			 #        [14, 15, 16]]) 
				XXXXX fixing 

             .. figure:: img/bc_2d_2.svg 




Cache effects
~~~~~~~~~~~~~

Memory access is cheaper when it is grouped: accessing a big array in a 
continuous way is much faster than random access. This implies amongst 
other things that **smaller strides are faster**:

  .. sourcecode:: ipython

    In [1]: c = np.zeros((1e4, 1e4), order='C')

    In [2]: %timeit c.sum(axis=0)
    1 loops, best of 3: 3.89 s per loop

    In [3]: %timeit c.sum(axis=1)
    1 loops, best of 3: 188 ms per loop

    In [4]: c.strides
    Out[4]: (80000, 8)

  This is the reason why Fortran ordering or C ordering may make a big
  difference on operations:

  .. sourcecode:: ipython

    In [5]: a = np.random.rand(20, 2**18)

    In [6]: b = np.random.rand(20, 2**18)

    In [7]: %timeit np.dot(b, a.T)
    1 loops, best of 3: 194 ms per loop

    In [8]: c = np.ascontiguousarray(a.T)

    In [9]: %timeit np.dot(b, c)
    10 loops, best of 3: 84.2 ms per loop

  Note that copying the data to work around this effect may not be worth it:

  .. sourcecode:: ipython

    In [10]: %timeit c = np.ascontiguousarray(a.T)
    10 loops, best of 3: 106 ms per loop

  Using `numexpr <http://code.google.com/p/numexpr/>`_ can be useful to
  automatically optimize code for such effects.


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
   
   # four temporary arrays will be created due to unnecessary parenthesis
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
   D = npy.sqrt(((X[:, np.newaxis, :] - X) ** 2).sum(axis=-1))


Numexpr
~~~~~~~

- Evaluation of complex expressions with one operation at a time can lead
  also into suboptimal performance
    
    - Effectively, one carries out multiple *for* loops in the NumPy C-code

- Numexpr package provides fast evaluation of array expressions

.. code-block:: python

   import numexpr as ne
   x = numpy.random.random((1000000, 1))
   y = numpy.random.random((1000000, 1))
   poly = ne.evaluate("((.25*x + .75)*x - 1.5)*x - 2")

- By default, numexpr tries to use multiple threads
- Number of threads can be queried and set with
  `ne.set_num_threads(nthreads)`
- Supported operators and functions:
  +,-,\*,/,\*\*, sin, cos, tan, exp, log, sqrt
- Speedups in comparison to NumPy are typically between 0.95 and 4
- Works best on arrays that do not fit in CPU cache




Performance boosting
--------------------

For many user cases, using NumPy or Pandas is sufficient. However, in some computationally heavy applications, 
it is possible to improve the performance by pre-compiling expensive functions.
`Cython <https://cython.org/>`__ and `Numba <https://numba.pydata.org/>`__ 
are among the popular choices and both of them have good support for numpy arrays. 


Cython
^^^^^^

Cython is a superset of Python that additionally supports calling C functions and 
declaring C types on variables and class attributes. Under Cython, source code gets 
translated into optimized C/C++ code and compiled as Python extension modules. 

Developers can run the ``cython`` command-line utility to produce a ``.c`` file from 
a ``.py`` file which needs to be compiled with a C compiler to an ``.so`` library 
which can then be directly imported in a Python program. There is, however, also an easy 
way to use Cython directly from Jupyter notebooks through the ``%%cython`` magic 
command. We will restrict the discussion here to the Jupyter-way - for a full overview 
of the capabilities refer to the `documentation <https://cython.readthedocs.io/en/latest/>`__.


.. demo:: Cython

   Consider the following pure Python code which integrates a function:

   .. literalinclude:: example/integrate_python.py 

   We generate a dataframe and apply the :meth:`apply_integrate_f` function on its columns, timing the execution:

   .. code-block:: ipython

      df = pd.DataFrame({"a": np.random.randn(1000),
                        "b": np.random.randn(1000),
                        "N": np.random.randint(100, 1000, (1000))})                

      %timeit apply_integrate_f(df['a'], df['b'], df['N'])
      # 279 ms ± 1.21 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

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
      # 279 ms ± 1.21 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   Now we can start adding data type annotation to the input variables:

   .. literalinclude:: example/integrate_cython_dtype0.py 

   .. code-block:: ipython

      #this will not work
      #%timeit apply_integrate_f_cython_dtype0(df['a'], df['b'], df['N'])
      #but rather 
      %timeit apply_integrate_f_cython_dtype0(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
      # 279 ms ± 1.21 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   .. warning::

      You can not pass a Series directly since the Cython definition is specific to an array. 
      Instead using the ``Series.to_numpy()`` to get the underlying NumPy array
      which works nicely with Cython.

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

   .. code-block:: ipython

      %timeit apply_integrate_f_cython_dtype1(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
      # 279 ms ± 1.21 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


   Last step, we can add type annotation to the local variables within the functions and output.

   .. literalinclude:: example/integrate_cython_dtype2.py 

   .. code-block:: ipython

      %timeit apply_integrate_f_cython_dtype2(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
      # 279 ms ± 1.21 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   
   Now it is over 400 XXX times faster than the original Python implementation, and we haven't really modified the code. 


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


.. demo:: Numba

   Consider the integration example again using Numba this time:

   .. literalinclude:: example/integrate_numba.py 

   .. code-block:: ipython

      # try passing Pandas Series 
      %timeit apply_integrate_f_numba(df['a'],df['b'],df['N'])
      # 6.02 ms ± 56.5 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
      # try passing NumPy array
      %timeit apply_integrate_f_numba(df['a'].to_numpy(),df['b'].to_numpy(),df['N'].to_numpy())
      # 625 µs ± 697 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


   .. note:: 
   
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




**WRITEME: use word-count example here**


Exercises
^^^^^^^^^

Here we have two exercises: the starting point is a Python function. Try to speed it up with 
Numba or Cython (depending on what you find most interesting).

.. exercise:: Pairwise distance

   .. literalinclude:: example/dis_python.py

   .. code-block:: ipython

      X = np.random.random((1000, 3))
      %timeit dis_python(X)


   .. solution::

      .. tabs:: 
   
         .. tab:: numpy
   
                .. literalinclude:: example/dis_numpy.py 

                .. code-block:: ipython

                   X = np.random.random((1000, 3))
                   %timeit dis_numpy(X)

   
         .. tab:: cython
   
                .. literalinclude:: example/dis_cython.py 

                .. code-block:: ipython

                   X = np.random.random((1000, 3))
                   %timeit dis_cython(X)
   
         .. tab:: numba
   
                .. literalinclude:: example/dis_numba.py 

                .. code-block:: ipython

                   X = np.random.random((1000, 3))
                   %timeit dis_numba(X)


.. exercise:: Bubble sort

   To make a long story short, in the worse case the time taken by the Bubblesort algorithm is 
   roughly :math:`O(n^2)` where  :math:`n` is the number of items being sorted. 

   .. image:: img/Bubble-sort-example-300px.gif


   .. literalinclude:: example/bs_python.py 

   .. code-block:: ipython

      import random
      l = [random.randint(1,1000) for num in range(1, 1000)]
      %timeit bs_python(l)


   .. solution:: 
   
      .. tabs:: 

         .. tab:: cython

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


         .. tab:: numba

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
