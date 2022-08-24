.. _boosting:

Performance boosting
====================

.. objectives::

   - Learn how to boost performance using Numba and Cython

.. instructor-note::

   - 20 min teaching/type-along
   - 20 min exercises

After benchmarking and optimizing your code, you can start thinking of accelerating 
it further with libraries like Cython and Numba to pre-compile performance-critical functions.


Pre-compiling Python
--------------------

For many (or most) use cases, using NumPy or Pandas is sufficient. However, in some computationally heavy applications, 
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


.. callout:: Numba vs Cython

   Should you use Numba or Cython? Does it matter?

   - Performance is usually very similar and exact results depend on versions of 
     Python, Cython, Numba and NumPy.
   - Numba is generally easier to use (just add ``@jit``)
   - Cython is more stable and mature, Numba developing faster
   - Numba also works for GPUs
   - Cython can compile arbitrary Python code and directly call C libraries, 
     Numba has restrictions
   - Numba requires LLVM toolchain, Cython only C compiler.
   
   Finally:

   NumPy is really good at what it does. For simple operations or small data, 
   Numba or Cython is not going to outperform it. But when things get more complex 
   these frameworks can save the day!


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

   Try to implement one faster version!

   .. solution:: Hints

      - You can replace the double loop (the manual calculation of an ACF) with an 
        in-built NumPy function, :meth:`np.correlate`. NumPy gurus often know which 
        function to use for which algorithms, but searching the internet also helps.
        One typically needs to figure out how to use the in-built function for the 
        particular use case.
      - There are two ways of using Numba, one with ``nopython=False`` and one with 
        ``nopython=True``. The latter needs a rewrite of the :meth:`word_acf` function 
        to accept the ``mask`` array, since Numba cannot pre-compile the expression 
        defining ``mask``.

   .. solution:: 

      The function uses a Python object (``mask``) inside a double for-loop, 
      which is guaranteed to be suboptimal. There are a number of ways to speed 
      it up. One is to use ``numba`` and just-in-time compilation, as we shall 
      see below. 

      Another is to find an in-built vectorized NumPy function which can calculate the 
      autocorrelation for us! Here are the Numpy and Numba ``(nopython=False)`` versions:

      .. tabs:: 
   
         .. tab:: NumPy

            .. literalinclude:: exercise/autocorrelation_numba_numpy.py
               :pyobject: word_acf_numpy

         .. tab:: Numba

            .. literalinclude:: exercise/autocorrelation_numba_numpy.py
               :pyobject: word_acf_numba_py

         In the `autocorr-numba-numpy branch <https://github.com/enccs/word-count-hpda/tree/autocorr-numba-numpy>`__ 
         of the word-count-hpda repository you 
         can additionally find a ``nopython=True`` Numba version as well as benchmarking 
         of all the versions. Note that the Numba functions use ``cache=True`` to save the 
         precompiled code so that subsequent executions of the ``autocorrelation.py`` script 
         are faster than the first.


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


.. keypoints::

   - To squeeze the last drop of performance out of your Python code you can 
     convert performance-critical functions to Numba or Cython
   - Both Numba and Cython pre-compile Python code to make it run faster.
