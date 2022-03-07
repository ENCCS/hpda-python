.. include:: .special.rst

Optimization
============

Once your code is working reliably, you can start thinking of optimizing it.


.. warning::

Always measure the code before you start optimization. Don't base your optimization 
on theoretical consideration, otherwise you'll have surprises. 


Profilers 
---------

Timeit
******

If you use Jupyter-notebook, the best choice will be to use ``timeit`` (https://docs.python.org/library/timeit.html) to time a small piece of code:

.. sourcecode:: ipython

    In [1]: import numpy as np

    In [2]: a = np.arange(1000)

    In [3]: %timeit a ** 2
    100000 loops, best of 3: 5.73 us per loop

.. note::

   For long running calls, using ``%time`` instead of ``%timeit``; it is
   less precise but faster


cProfile
********

For more complex code, one could use the built-in python profilers 
<https://docs.python.org/3/library/profile.html>`_ ``cProfile``.

    .. sourcecode:: console

        $  python -m cProfile -o demo.prof demo.py

    Using the ``-o`` switch will output the profiler results to the file
    ``demo.prof`` to view with an external tool. 


Line-profiler
*************

The cprofile tells us which function takes most of the time, but not where it is called.

For this information, we use the `line_profiler <http://packages.python.org/line_profiler/>`_: in the
source file  by adding a decorator ``@profile`` in the functions of interests

.. sourcecode:: python

    @profile
    def test():
        data = np.random.random((5000, 100))
        u, s, v = linalg.svd(data)
        pca = np.dot(u[:, :10], data)
        results = fastica(pca.T, whiten=False)

Then we run the script using the `kernprof.py
<http://packages.python.org/line_profiler>`_ program, with switches ``-l, --line-by-line`` and ``-v, --view`` to use the line-by-line profiler and view the results in addition to saving them:

.. sourcecode:: console

    $ kernprof.py -l -v demo.py

    Wrote profile results to demo.py.lprof
    Timer unit: 1e-06 s

    File: demo.py
    Function: test at line 5
    Total time: 14.2793 s

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    =========== ============ ===== ========= ======= ==== ========
        5                                           @profile
        6                                           def test():
        7         1        19015  19015.0      0.1      data = np.random.random((5000, 100))
        8         1     14242163 14242163.0   99.7      u, s, v = linalg.svd(data)
        9         1        10282  10282.0      0.1      pca = np.dot(u[:10, :], data)
       10         1         7799   7799.0      0.1      results = fastica(pca.T, whiten=False)

It is clear from the profiling results: 
the SVD is taking all the time, we need to optimise it if possible.



performance optimization 
------------------------

Once we have identified the bottlenecks, we need to make the corresponding code go faster.

Algorithm optimization
**********************

The first thing to look into is the underlying algorithm you chose: is it optimal?
To answer this question,  a good understanding of the maths behind the algorithm helps. 
For certain algorithms, many of the bottlenecks will be linear 
algebra computations. In these cases, using the right function to solve 
the right problem is key. For instance, an eigenvalue problem with a 
symmetric matrix is much easier to solve than with a general matrix. Moreover, 
most often, you can avoid inverting a matrix and use a less costly 
(and more numerically stable) operation. However, it can be as simple as 
moving computation or memory allocation outside a loop, and this happens very often as well.

Singular Value Decomposition
............................

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

We can try this using the example above: (XXXX fixing the example)

.. sourcecode:: console

    $ kernprof.py -l -v demo_opt.py

    Wrote profile results to demo_opt.py.lprof
    Timer unit: 1e-06 s

    File: demo_opt.py
    Function: test at line 5
    Total time: 14.2793 s



XXXX add sparse matrix example here 

CPU usage optimization
**********************

Vectorization
.............

Arithmetic is one place where numpy performance outperforms python list and the reason is that it uses vectorization.
A lot of the data analysis involves a simple operation being applied to each element of a large dataset.
In such cases, vectorization is key for better performance.

.. challenge::  vectorized operation vs for loop 

   .. tabs::

      .. tab:: python

             .. code-block:: python

			import numpy as np
			a = np.arange(1000)
			a_dif = np.zeros(999, int)
			for i in range(1, len(a)):
			    a_dif[i-1] = a[i] - a[i-1]

      .. tab:: numpy

             .. code-block:: python

			import numpy as np
                        a = np.arange(1000)
			a_dif = a[1:] - a[:-1]




So one should consider use "vectorized" operations whenever possible.

For user-defined functions, one can use e.g. numba. 
XXXX add one example




Memory usage optimization
*************************

Broadcasting
............

Basic operations of numpy are elementwise, and the shape of the arrays should be compatible.
However, in practice under certain conditions, it is possible to do operations on arrays of different shapes.
NumPy expands the arrays such that the operation becomes viable.

.. note:: Broadcasting Rules  

  - Dimensions match when they are equal, or when either is 1 or None.   
  - In the latter case, the dimension of the output array is expanded to the larger of the two.

.. note:: the broadcasted arrays are never physically constructed




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
.............

Memory access is cheaper when it is grouped: accessing a big array in a 
continuous way is much faster than random access. This implies amongst 
other things that **smaller strides are faster** (see :ref:`cache_effects`):

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
................

- In complex expressions, NumPy stores intermediate values in
  temporary arrays
- Memory consumption can be higher than expected

.. code-block:: python

   a = numpy.random.random((1024, 1024, 50))
   b = numpy.random.random((1024, 1024, 50))
   
   # two temporary arrays will be created
   c = 2.0 * a - 4.5 * b
   
   # three temporary arrays will be created due to unnecessary parenthesis
   c = (2.0 * a - 4.5 * b) + 1.1 * (numpy.sin(a) + numpy.cos(b))

- Broadcasting approaches can lead also to hidden temporary arrays  XXXX add one example
- XXXX Not clear to me Example: pairwise distance of **M** points in 3 dimensions
    - Input data is M x 3 array
    - Output is M x M array containing the distance between points i
      and j
	- There is a temporary 1000 x 1000 x 3 array

.. code-block:: python

   X = numpy.random.random((1000, 3))
   D = numpy.sqrt(((X[:, numpy.newaxis, :] - X) ** 2).sum(axis=-1))


Numexpr
.......

- Evaluation of complex expressions with one operation at a time can lead
  also into suboptimal performance
    - Effectively, one carries out multiple *for* loops in the NumPy
      C-code

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
********************

For many user cases, using NumPy or Pandas is sufficient. Howevewr, in some computationally heavy applications, 
it is possible to improve the performance by using the compiled code.
Cython and Numba are among the popular choices and both of them have good support for numpy arrays. 


Cython
......

The source code gets translated into optimized C/C++ code and compiled as Python extension modules. 

There are three ways of declaring functions: 


``def`` - Python style:
Declaring the types of arguments and local types (thus return values) can allow Cython to generate optimised code which speeds up the execution. If the types are declared then a ``TypeError`` will be raised if the function is passed the wrong types.

``cdef`` - C style:
Cython treats the function as pure 'C' functions. All types *must* be declared. This will give you the best performance but there are a number of consequences. One should really take care of the ``cdef`` declared functions, since you are actually writing in C.

``cpdef`` - Python/C mixed
``cpdef`` functions combine both ``def`` and ``cdef`` by creating two functions; a ``cdef`` for C types and a ``def`` for Python types. This exploits early binding so that ``cpdef`` functions may be as fast as possible when using C fundamental types (by using ``cdef``). ``cpdef`` functions use dynamic binding when passed Python objects and this might much slower, perhaps as slow as ``def`` declared functions.   XXXX rewrite this part.


Numba
.....


An alternative to statically compiling Cython code is to use a dynamic just-in-time (JIT) compiler with `Numba <https://numba.pydata.org/>`__.

Numba allows you to write a pure Python function which can be JIT compiled to native machine instructions, similar in performance to C, C++ and Fortran, by simply adding the decorator ``@jit`` in your function.

Numba supports compilation of Python to run on either CPU or GPU hardware and is designed to integrate with the Python scientific software stack. The optimized machine code is generated by the LLVM compiler infrastructure.

.. note::

    The ``@jit`` compilation will add overhead to the runtime of the function, so performance benefits may not be realized especially when using small data sets. In general, the Numba engine is performant with a larger amount of data points (e.g. 1+ million).
    Consider `caching <https://numba.readthedocs.io/en/stable/developer/caching.html>`__ your function to avoid compilation overhead each time your function is run, i.e. the first time a function is run using the Numba engine will be slow as Numba will have some function compilation overhead. However, once the JIT compiled functions are cached, subsequent calls will be fast. 


Numba can be used in 2 ways with pandas:

#. Specify the ``engine="numba"`` keyword in select pandas methods
#. Define your own Python function decorated with ``@jit`` and pass the underlying NumPy array of :class:`Series` or :class:`DataFrame` (using ``to_numpy()``) into the function

If Numba is installed, one can specify ``engine="numba"`` in select pandas methods to execute the method using Numba.
Methods that support ``engine="numba"`` will also have an ``engine_kwargs`` keyword that accepts a dictionary that allows one to specify
``"nogil"``, ``"nopython"`` and ``"parallel"`` keys with boolean values to pass into the ``@jit`` decorator.
If ``engine_kwargs`` is not specified, it defaults to ``{"nogil": False, "nopython": True, "parallel": False}`` unless otherwise specified.


examples
........

Integration
^^^^^^^^^^^


Consider the following pure Python code:



.. challenge:: integration

	we first generate a dataframe and apply the integrate_f function on the dataframe.


   .. tabs:: 

      .. tab:: python

             .. literalinclude:: example/integrate_python.py 

      .. tab:: cython

             .. literalinclude:: example/integrate_cython.py 

      .. tab:: numba

             .. literalinclude:: example/integrate_numba.py 



   .. tabs:: benchmark

	.. code-block:: python

	  df = pd.DataFrame(
  		  {
        		"a": np.random.randn(1000),
		        "b": np.random.randn(1000),
		        "N": np.random.randint(100, 1000, (1000)),
		        "x": "x",
		    }
		)



Pairwise distance
^^^^^^^^^^^^^^^^^


.. challenge:: pairwise distance

	we first generate a dataframe and apply the integrate_f function on the dataframe.


   .. tabs:: 

      .. tab:: python

             .. literalinclude:: example/dis_python.py 

      .. tab:: numpy

             .. literalinclude:: example/dis_numpy.py 

      .. tab:: cython

             .. literalinclude:: example/dis_cython.py 

      .. tab:: numba

             .. literalinclude:: example/dis_numba.py 


Bubble sort
^^^^^^^^^^^

Long stroy short, in the worse case, the time Bubblesort algorithm takes is roughly :math:`O(n^2)` where  :math:`n` is the number of items being sorted. 

.. image:: img/Bubble-sort-example-300px.gif


.. challenge:: Bubble sort

   .. tabs:: 

      .. tab:: python

             .. literalinclude:: example/bs_python.py 

      .. tab:: cython

             .. literalinclude:: example/bs_cython.py 

      .. tab:: numba

             .. literalinclude:: example/bs_numba.py 





.. note:

Note that the relative results also depend on what version of Python, Cython, Numba, and NumPy you are using. Also, the compiler choice for installing NumPy can account for differences in the results.
NumPy is really good at what it does. For simple operations, Numba is not going to outperform it, but when things get more complex Numba will save the day. 
