NumPy and Pandas
================

.. objectives::

   - Understand limitations of Python's standard library for large data processing
   - Understand vectorization and basic array computing in NumPy
   - Learn to use several of NumPy's numerical computing tools 
   - Learn to use data structures and analysis tools from Pandas

> This episode is based on this 
> `repository on HPC-Python from CSC <https://github.com/csc-training/hpc-python>`__ and 
> this `Python for Scientific Computing lesson <https://aaltoscicomp.github.io/python-for-scicomp/>`__.

Why can Python be slow?
-----------------------

Python is very flexible and dynamic language, but the flexibility comes with
a price.

Computer programs are nowadays practically always written in a high-level
human readable programming language and then translated to the actual machine
instructions that a processor understands. There are two main approaches for
this translation:

 - For **compiled** programming languages, the translation is done by
   a compiler before the execution of the program
 - For **interpreted** languages, the translation is done by an interpreter
   during the execution of the program

Compiled languages are typically more efficient, but the behaviour of
the program during runtime is more static than with interpreted languages.
The compilation step can also be time consuming, so the software cannot
always be tested as rapidly during development as with interpreted
languages.

Python is an interpreted language, and many features that make development
rapid with Python are a result of that, with the price of reduced performance
in some cases.

Dynamic typing
^^^^^^^^^^^^^^

Python is a very dynamic language. As variables get type only during the
runtime as values (Python objects) are assigned to them, it is more difficult
for the interpreter to optimize the execution (in comparison, a compiler can
make extensive analysis and optimization before the execution). Even though,
in recent years, there has been a lot of progress in just-in-time (JIT)
compilation techniques that allow programs to be optimized at runtime, the
inherent, very dynamic nature of the Python programming language remains one
of its main performance bottlenecks.

Flexible data structures
^^^^^^^^^^^^^^^^^^^^^^^^

The built-in data structures of Python, such as lists and dictionaries,
are very flexible, but they are also very generic, which makes them not so
well suited for extensive numerical computations. Actually, the implementation
of the data structures (e.g. in the standard CPython interpreter) is often
quite efficient when one needs to process different types of data. However,
when one is processing only a single type of data, for example only
floating point numbers, there is a lot of unnecessary overhead due to the
generic nature of these data structures.

Multithreading
^^^^^^^^^^^^^^

The performance of a single CPU core has stagnated over the last ten years,
and as such most of the speed-up in modern CPUs is coming from using multiple
CPU cores, i.e. parallel processing. Parallel processing is normally based
either on multiple threads or multiple processes. Unfortunately, the memory
management of the standard CPython interpreter is not thread-safe, and it uses
something called Global Interpreter Lock (GIL) to safeguard memory integrity.
In practice, this limits the benefits of multiple threads only to some
special situations (e.g. I/O). Fortunately, parallel processing with multiple
processes is relatively straightforward also with Python.

In summary, the flexibility and dynamic nature of Python, that enhances
the programmer productivity greatly, is also the main cause for the
performance problems. Flexibility comes with a price! Fortunately, as we
discuss in the course, many of the bottlenecks can be circumvented.


NumPy
-----

- Standard Python is not well suitable for numerical computations
    - lists are very flexible but also slow to process in numerical
      computations

- Numpy adds a new **array** data type
    - static, multidimensional
    - fast processing of arrays
    - tools for linear algebra, random numbers, *etc.*

NumPy arrays
^^^^^^^^^^^^

- All elements of an array have the same type
- Array can have multiple dimensions
- The number of elements in the array is fixed, shape can be changed

.. figure:: img/list-vs-array.svg
   :align: center
   :scale: 100 %


Array computing and vectorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For loops in Python are slow. If one needs to apply a mathematical operation
on multiple (consecutive) elements of an array, it is always better to use a
vectorised operation if possible.

In practice, a vectorised operation means reframing the code in a manner that
completely avoids a loop and instead uses e.g. slicing to apply the operation
on the whole array (slice) at one go.

For example, the following code for calculating the difference of neighbouring
elements in an array:

.. code-block:: python

   # brute force using a for loop
   arr = numpy.arange(1000)
   dif = numpy.zeros(999, int)
   for i in range(1, len(arr)):
       dif[i-1] = arr[i] - arr[i-1]

can be re-written as a vectorised operation:

.. code-block:: python

   # vectorised operation
   arr = numpy.arange(1000)
   dif = arr[1:] - arr[:-1]

.. figure:: img/vectorised-difference.png
   :align: center

The first brute force approach using a for loop is approx. 80 times slower
than the second vectorised form!


Creating numpy arrays
^^^^^^^^^^^^^^^^^^^^^

From a list:

.. code-block:: python

   import numpy
   a = numpy.array((1, 2, 3, 4), float)
   a
   # array([ 1., 2., 3., 4.])

   list1 = [[1, 2, 3], [4,5,6]]
   mat = numpy.array(list1, complex)
   mat
   # array([[ 1.+0.j, 2.+0.j, 3.+0.j],
   #       [ 4.+0.j, 5.+0.j, 6.+0.j]])

   mat.shape
   # (2, 3)

   mat.size
   # 6


Helper functions for creating arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`arange` and `linspace` can generate ranges of numbers:

.. code-block:: python

    a = numpy.arange(10)
    a
    # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    b = numpy.arange(0.1, 0.2, 0.02)
    b
    # array([0.1 , 0.12, 0.14, 0.16, 0.18])

    c = numpy.linspace(-4.5, 4.5, 5)
    c
    # array([-4.5 , -2.25, 0. , 2.25, 4.5 ])


Array with given shape initialized to `zeros`, `ones` or arbitrary value (`full`):

.. code-block:: python

   a = numpy.zeros((4, 6), float)
   a.shape
   # (4, 6)

   b = numpy.ones((2, 4))
   b
   # array([[ 1., 1., 1., 1.],
   #       [ 1., 1., 1., 1.]])
	   
   c = numpy.full((2, 3), 4.2)
   c
   # array([[4.2, 4.2, 4.2],
   #       [4.2, 4.2, 4.2]])

Empty array (no values assigned) with `empty`.

Similar arrays as an existing one with `zeros_like`, `ones_like`, 
`full_like` and `empty_like`:

.. code-block:: python

   a = numpy.zeros((4, 6), float)
   b = numpy.empty_like(a)
   c = numpy.ones_like(a)
   d = numpy.full_like(a, 9.1)

Non-numeric data
~~~~~~~~~~~~~~~~

NumPy supports also storing non-numerical data e.g. strings (largest
element determines the item size)

.. code-block:: python

   a = numpy.array(['foo', 'foo-bar'])
   a
   # array(['foo', 'foo-bar'], dtype='|U7')

Character arrays can, however, be sometimes useful

.. code-block:: python

   dna = 'AAAGTCTGAC'
   a = numpy.array(dna, dtype='c')
   a
   # array([b'A', b'A', b'A', b'G', b'T', b'C', b'T', b'G', b'A', b'C'],
   #       dtype='|S1')


Accessing arrays
~~~~~~~~~~~~~~~~

Simple indexing:

.. code-block:: python

   mat = numpy.array([[1, 2, 3], [4, 5, 6]])
   mat[0,2]
   #  3

   mat[1,-2]
   # 5

Slicing:

.. code-block:: python

   a = numpy.arange(10)
   a[2:]
   # array([2, 3, 4, 5, 6, 7, 8, 9])

   a[:-1]
   # array([0, 1, 2, 3, 4, 5, 6, 7, 8])

   a[1:3] = -1
   a
   # array([0, -1, -1, 3, 4, 5, 6, 7, 8, 9])

   a[1:7:2]
   # array([1, 3, 5])

Slicing of arrays in multiple dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Multidimensional arrays can be sliced along multiple dimensions
- Values can be assigned to only part of the array

.. code-block:: python

   a = numpy.zeros((4, 4))
   a[1:3, 1:3] = 2.0
   a
   # array([[ 0., 0., 0., 0.],
   #       [ 0., 2., 2., 0.],
   #       [ 0., 2., 2., 0.],
   #       [ 0., 0., 0., 0.]])
```


Views and copies of arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Simple assignment creates references to arrays
- Slicing creates "views" to the arrays
- Use `copy()` for real copying of arrays

.. code-block:: python

   a = numpy.arange(10)
   b = a              # reference, changing values in b changes a
   b = a.copy()       # true copy

   c = a[1:4]         # view, changing c changes elements [1:4] of a
   c = a[1:4].copy()  # true copy of subarray


Array manipulation
~~~~~~~~~~~~~~~~~~

- `reshape` : change the shape of array

.. code-block:: python

   mat = numpy.array([[1, 2, 3], [4, 5, 6]])
   mat
   #array([[1, 2, 3],
   #      [4, 5, 6]])

   mat.reshape(3,2)
   # array([[1, 2],
   #       [3, 4],
   #       [5, 6]])

- `ravel` : flatten array to 1-d

.. code-block:: python

    mat.ravel()
    # array([1, 2, 3, 4, 5, 6])


Array manipulation
~~~~~~~~~~~~~~~~~~

- `concatenate` : join arrays together

.. code-block:: python

    mat1 = numpy.array([[1, 2, 3], [4, 5, 6]])
    mat2 = numpy.array([[7, 8, 9], [10, 11, 12]])
    numpy.concatenate((mat1, mat2))
    # array([[ 1, 2, 3],
    #       [ 4, 5, 6],
    #       [ 7, 8, 9],
    #       [10, 11, 12]])

    numpy.concatenate((mat1, mat2), axis=1)
    # array([[ 1, 2, 3,  7,  8,  9],
    #       [ 4, 5, 6, 10, 11, 12]])

`split` : split array to N pieces

.. code-block:: python

    numpy.split(mat1, 3, axis=1)
    # [array([[1], [4]]), array([[2], [5]]), array([[3], [6]])]


Array operations
~~~~~~~~~~~~~~~~

Most operations for numpy arrays are done element-wise
(`+`, `-`,  `*`,  `/`,  `**`)

.. code-block:: python

    a = numpy.array([1.0, 2.0, 3.0])
    b = 2.0
    a * b
    # array([ 2., 4., 6.])

    a + b
    # array([ 3., 4., 5.])

    a * a
    # array([ 1., 4., 9.])

Numpy has special functions which can work with array arguments
(sin, cos, exp, sqrt, log, ...)

.. code-block:: python

    import numpy, math
    a = numpy.linspace(-math.pi, math.pi, 8)
    a
    # array([-3.14159265, -2.24399475, -1.34639685, -0.44879895,
    #        0.44879895,  1.34639685,  2.24399475,  3.14159265])

    numpy.sin(a)
    # array([ -1.22464680e-16, -7.81831482e-01, -9.74927912e-01,
    #         -4.33883739e-01,  4.33883739e-01,  9.74927912e-01,
    #          7.81831482e-01,  1.22464680e-16])

    math.sin(a)
    # Traceback (most recent call last):
    # File "<stdin>", line 1, in ?
    # TypeError: only length-1 arrays can be converted to Python scalars


I/O with numpy
~~~~~~~~~~~~~~

- Numpy provides functions for reading data from file and for writing data
  into the files
- Simple text files
    - `numpy.loadtxt`
    - `numpy.savetxt`
    - Data in regular column layout
    - Can deal with comments and different column delimiters


Random numbers
~~~~~~~~~~~~~~

- The module `numpy.random` provides several functions for constructing
  random arrays
    - `random`: uniform random numbers
    - `normal`: normal distribution
    - `choice`: random sample from given array
    - ...

.. code-block:: python

    import numpy.random as rnd
    rnd.random((2,2))
    # array([[ 0.02909142, 0.90848 ],
    #       [ 0.9471314 , 0.31424393]])

    rnd.choice(numpy.arange(4), 10)
    # array([0, 1, 1, 2, 1, 1, 2, 0, 2, 3])


Polynomials
~~~~~~~~~~~

- Polynomial is defined by an array of coefficients p
  $p(x, N) = p[0] x^{N-1} + p[1] x^{N-2} + ... + p[N-1]$
- For example:
    - Least square fitting: `numpy.polyfit`
    - Evaluating polynomials: `numpy.polyval`
    - Roots of polynomial: `numpy.roots`

.. code-block:: python

    x = numpy.linspace(-4, 4, 7)
    y = x**2 + rnd.random(x.shape)

    p = numpy.polyfit(x, y, 2)
    p
    # array([ 0.96869003, -0.01157275, 0.69352514])


Linear algebra
~~~~~~~~~~~~~~

- Numpy can calculate matrix and vector products efficiently: `dot`,
  `vdot`, ...
- Eigenproblems: `linalg.eig`, `linalg.eigvals`, ...
- Linear systems and matrix inversion: `linalg.solve`, `linalg.inv`

.. code-block:: python

    A = numpy.array(((2, 1), (1, 3)))
    B = numpy.array(((-2, 4.2), (4.2, 6)))
    C = numpy.dot(A, B)

    b = numpy.array((1, 2))
    numpy.linalg.solve(C, b) # solve C x = b
    # array([ 0.04453441, 0.06882591])

- Normally, NumPy utilises high performance libraries in linear algebra
  operations
- Example: matrix multiplication C = A * B matrix dimension 1000
    - pure python:           522.30 s
    - naive C:                 1.50 s
    - numpy.dot:               0.04 s
    - library call from C:     0.04 s


Anatomy of NumPy array
~~~~~~~~~~~~~~~~~~~~~~

- **ndarray** type is made of
    - one dimensional contiguous block of memory (raw data)
    - indexing scheme: how to locate an element
    - data type descriptor: how to interpret an element

.. figure:: img/ndarray-in-memory.svg
   :align: center
   


NumPy indexing
~~~~~~~~~~~~~~

- There are many possible ways of arranging items of N-dimensional
  array in a 1-dimensional block
- NumPy uses **striding** where N-dimensional index ($n_0, n_1, ..., n_{N-1}$)
  corresponds to offset from the beginning of 1-dimensional block
  
$$
offset = \sum_{k=0}^{N-1} s_k n_k, s_k \text{ is stride in dimension k}
$$


.. figure:: img/ndarray-in-memory-offset.svg
   :align: center

ndarray attributes
~~~~~~~~~~~~~~~~~~

`a = numpy.array(...)`
  : `a.flags`
    : various information about memory layout

    `a.strides`
    : bytes to step in each dimension when traversing

    `a.itemsize`
    : size of one array element in bytes

    `a.data`
    : Python buffer object pointing to start of arrays data

    `a.__array_interface__`
    : Python internal interface


Advanced indexing
~~~~~~~~~~~~~~~~~

- Numpy arrays can be indexed also with other arrays (integer or
  boolean)

.. code-block:: python

    x = numpy.arange(10,1,-1)
    x
    # array([10, 9, 8, 7, 6, 5, 4, 3, 2])

    x[numpy.array([3, 3, 1, 8])]
    # array([7, 7, 9, 2])

Boolean "mask" arrays:

.. code-block:: python

    m = x > 7
    m
    # array([ True, True, True, False, False, ...

    x[m]
    # array([10, 9, 8])

Advanced indexing creates copies of arrays.


Vectorized operations
~~~~~~~~~~~~~~~~~~~~~

- `for` loops in Python are slow
- Use "vectorized" operations when possible
- Example: difference
    - for loop is ~80 times slower!

.. code-block:: python

   # brute force using a for loop
   arr = numpy.arange(1000)
   dif = numpy.zeros(999, int)
   for i in range(1, len(arr)):
       dif[i-1] = arr[i] - arr[i-1]
   
   # vectorized operation
   arr = numpy.arange(1000)
   dif = arr[1:] - arr[:-1]

.. figure:: img/vectorised-difference.png
   :align: center


Broadcasting
~~~~~~~~~~~~

- If array shapes are different, the smaller array may be broadcasted
  into a larger shape

.. code-block:: python

    from numpy import array
    a = array([[1,2],[3,4],[5,6]], float)
    a
    #array([[ 1., 2.],
    #      [ 3., 4.],
    #      [ 5., 6.]])

    b = array([[7,11]], float)
    b
    # array([[ 7., 11.]])

    a * b
    # array([[ 7., 22.],
    #       [ 21., 44.],
    #       [ 35., 66.]])


Example: calculate distances from a given point

.. code-block:: python

   # array containing 3d coordinates for 100 points
   points = numpy.random.random((100, 3))
   origin = numpy.array((1.0, 2.2, -2.2))
   dists = (points - origin)**2
   dists = numpy.sqrt(numpy.sum(dists, axis=1))
   
   # find the most distant point
   i = numpy.argmax(dists)
   print(points[i])


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
   
   # three temporary arrays will be created due to unnecessary parenthesis
   c = (2.0 * a - 4.5 * b) + 1.1 * (numpy.sin(a) + numpy.cos(b))

- Broadcasting approaches can lead also to hidden temporary arrays
- Example: pairwise distance of **M** points in 3 dimensions
    - Input data is M x 3 array
    - Output is M x M array containing the distance between points i
      and j
	- There is a temporary 1000 x 1000 x 3 array

.. code-block:: python

   X = numpy.random.random((1000, 3))
   D = numpy.sqrt(((X[:, numpy.newaxis, :] - X) ** 2).sum(axis=-1))


Numexpr
~~~~~~~

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





Pandas
------

Pandas is a Python package that provides high-performance and easy to use 
data structures and data analysis tools.  
This page provides a brief overview of pandas, but the open source community 
developing the pandas package has also created excellent documentation and training 
material, including: 

- a  `Getting started guide <https://pandas.pydata.org/getting_started.html>`__ 
  (including tutorials and a 10 minute flash intro)
- a `"10 minutes to pandas" <https://pandas.pydata.org/docs/user_guide/10min.html#min>`__
  tutorial
- thorough `Documentation <https://pandas.pydata.org/docs/>`__ containing a user guide, 
  API reference and contribution guide
- a `cheatsheet <https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf>`__ 
- a `cookbook <https://pandas.pydata.org/docs/user_guide/cookbook.html#cookbook>`__.

Let's get a flavor of what we can do with pandas. We will be working with an
example dataset containing the passenger list from the Titanic, which is often used in Kaggle competitions and data science tutorials. First step is to load pandas::

    import pandas as pd

We can download the data from `this GitHub repository <https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv>`__
by visiting the page and saving it to disk, or by directly reading into 
a **dataframe**::

    url = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
    titanic = pd.read_csv(url, index_col='Name')

We can now view the dataframe to get an idea of what it contains and
print some summary statistics of its numerical data::

    # print the first 5 lines of the dataframe
    titanic.head()  
    
    # print summary statistics for each column
    titanic.describe()  

Ok, so we have information on passenger names, survival (0 or 1), age, 
ticket fare, number of siblings/spouses, etc. With the summary statistics we see that the average age is 29.7 years, maximum ticket price is 512 USD, 38\% of passengers survived, etc.

Let's say we're interested in the survival probability of different age groups. With two one-liners, we can find the average age of those who survived or didn't survive, and plot corresponding histograms of the age distribution::

    print(titanic.groupby("Survived")["Age"].mean())

::

    titanic.hist(column='Age', by='Survived', bins=25, figsize=(8,10), 
                 layout=(2,1), zorder=2, sharex=True, rwidth=0.9);
    

Clearly, pandas dataframes allows us to do advanced analysis with very few commands, but it takes a while to get used to how dataframes work so let's get back to basics.

.. callout:: Getting help

    Series and DataFrames have a lot functionality, but
    how can we find out what methods are available and how they work? One way is to visit 
    the `API reference <https://pandas.pydata.org/docs/reference/frame.html>`__ 
    and reading through the list. 
    Another way is to use the autocompletion feature in Jupyter and type e.g. 
    ``titanic["Age"].`` in a notebook and then hit ``TAB`` twice - this should open 
    up a list menu of available methods and attributes.

    Jupyter also offers quick access to help pages (docstrings) which can be 
    more efficient than searching the internet. Two ways exist:

    - Write a function name followed by question mark and execute the cell, e.g.
      write ``titanic.hist?`` and hit ``SHIFT + ENTER``.
    - Write the function name and hit ``SHIFT + TAB``.


What's in a dataframe?
----------------------

As we saw above, pandas dataframes are a powerful tool for working with tabular data. 
A pandas 
`DataFrame object <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame>`__ 
is composed of rows and columns:

.. image:: img/01_table_dataframe.svg

Each column of a dataframe is a 
`series object <https://pandas.pydata.org/docs/user_guide/dsintro.html#series>`__ 
- a dataframe is thus a collection of series::

    # print some information about the columns
    titanic.info()

Unlike a NumPy array, a dataframe can combine multiple data types, such as
numbers and text, but the data in each column is of the same type. So we say a
column is of type ``int64`` or of type ``object``.

Let's inspect one column of the Titanic passanger list data (first downloading
and reading the titanic.csv datafile into a dataframe if needed, see above)::

    titanic["Age"]
    titanic.Age          # same as above
    type(titanic["Age"])

The columns have names. Here's how to get them::

    titanic.columns

However, the rows also have names! This is what Pandas calls the **index**::

    titanic.index

We saw above how to select a single column, but there are many ways of
selecting (and setting) single or multiple rows, columns and values. We can
refer to columns and rows either by number or by their name::

    titanic.loc['Lam, Mr. Ali',"Age"]          # select single value by row and column
    titanic.loc[:'Lam, Mr. Ali',"Name":"Age"]  # slice the dataframe by row and column *names*
    titanic.iloc[0:2,3:6]                      # same slice as above by row and column *numbers*

    titanic.at['Lam, Mr. Ali',"Age"] = 42      # set single value by row and column *name* (fast)
    titanic.at['Lam, Mr. Ali',"Age"]           # select single value by row and column *name* (fast)
    titanic.at['Lam, Mr. Ali',"Age"] = 42      # set single value by row and column *name* (fast)
    titanic.iat[0,5]                           # select same value by row and column *number* (fast)

    titanic["foo"] = "bar"                     # set a whole column

Dataframes also support boolean indexing, just like we saw for ``numpy`` 
arrays::

    titanic[titanic["Age"] > 70]
    # ".str" creates a string object from a column
    titanic[titanic["Name"].str.contains("Margaret")]

What if your dataset has missing data? Pandas uses the value ``np.nan`` 
to represent missing data, and by default does not include it in any computations.
We can find missing values, drop them from our dataframe, replace them
with any value we like or do forward or backward filling::

    titanic.isna()                    # returns boolean mask of NaN values
    titanic.dropna()                  # drop missing values
    titanic.dropna(how="any")         # or how="all"
    titanic.dropna(subset=["Cabin"])  # only drop NaNs from one column
    titanic.fillna(0)                 # replace NaNs with zero
    titanic.fillna(method='ffill')    # forward-fill NaNs



Exercises 1
-----------

.. challenge:: Exploring dataframes

    - Have a look at the available methods and attributes using the 
      `API reference <https://pandas.pydata.org/docs/reference/frame.html>`__ 
      or the autocomplete feature in Jupyter. 
    - Try out a few methods using the Titanic dataset and have a look at 
      the docstrings (help pages) of methods that pique your interest
    - Compute the mean age of the first 10 passengers by slicing and the ``mean`` method
    - (Advanced) Using boolean indexing, compute the survival rate 
      (mean of "Survived" values) among passengers over and under the average age.
    
.. solution:: 

    - Mean age of the first 10 passengers: ``titanic.iloc[:10,:]["Age"].mean()`` 
      or ``titanic.loc[:9,"Age"].mean()`` or ``df.iloc[:10,5].mean()``.
    - Survival rate among passengers over and under average age: 
      ``titanic[titanic["Age"] > titanic["Age"].mean()]["Survived"].mean()`` and 
      ``titanic[titanic["Age"] < titanic["Age"].mean()]["Survived"].mean()``.


Tidy data
---------

The above analysis was rather straightforward thanks to the fact 
that the dataset is *tidy*.

.. image:: img/tidy_data.png

In short, columns should be variables and rows should be measurements, 
and adding measurements (rows) should then not require any changes to code 
that reads the data.

What would untidy data look like? Here's an example from 
some run time statistics from a 1500 m running event::

    runners = pd.DataFrame([
                  {'Runner': 'Runner 1', 400: 64, 800: 128, 1200: 192, 1500: 240},
                  {'Runner': 'Runner 2', 400: 80, 800: 160, 1200: 240, 1500: 300},
                  {'Runner': 'Runner 3', 400: 96, 800: 192, 1200: 288, 1500: 360},
              ])

What makes this data untidy is that the column names `400, 800, 1200, 1500`
indicate the distance ran. In a tidy dataset, this distance would be a variable
on its own, making each runner-distance pair a separate observation and hence a
separate row.

To make untidy data tidy, a common operation is to "melt" it, 
which is to convert it from wide form to a long form::

    runners = pd.melt(df, id_vars="Runner", 
                  value_vars=[400, 800, 1200, 1500], 
                  var_name="distance", 
                  value_name="time"
              )

In this form it's easier to **filter**, **group**, **join** 
and **aggregate** the data, and it's also easier to model relationships 
between variables.

The opposite of melting is to *pivot* data, which can be useful to 
view data in different ways as we'll see below.

For a detailed exposition of data tidying, have a look at 
`this article <http://vita.had.co.nz/papers/tidy-data.pdf>`__.



Working with dataframes
-----------------------

We saw above how we can read in data into a dataframe using the ``read_csv`` method.
Pandas also understands multiple other formats, for example using ``read_excel``,  
``read_hdf``, ``read_json``, etc. (and corresponding methods to write to file: 
``to_csv``, ``to_excel``, ``to_hdf``, ``to_json``, etc.)  

But sometimes you would want to create a dataframe from scratch. Also this can be done 
in multiple ways, for example starting with a numpy array::

    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

or a dictionary::

    df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                       'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                       'C': np.array([3] * 8, dtype='int32'),
                       'D': np.random.randn(8),
                       'E': np.random.randn(8)})

There are many ways to operate on dataframes. Let's look at a 
few examples in order to get a feeling of what's possible
and what the use cases can be.

We can easily split and concatenate or append dataframes::

    sub1, sub2, sub3 = df[:2], df[2:4], df[4:]
    pd.concat([sub1, sub2, sub3])
    sub1.append([sub2, sub3])      # same as above

When pulling data from multiple dataframes, a powerful ``merge()`` method is
available that acts similarly to merging in SQL. Say we have a dataframe containing the age of some athletes::

    age = pd.DataFrame([
        {"Runner": "Runner 4", "Age": 18},
        {"Runner": "Runner 2", "Age": 21},
        {"Runner": "Runner 1", "Age": 23},
        {"Runner": "Runner 3", "Age": 19},
    ])

We now want to use this table to annotate the original ``runners`` table from
before with their age. Note that the ``runners`` and ``age`` dataframes have a
different ordering to it, and ``age`` has an entry for ``Dave`` which is not
present in the ``runners`` table. We can let Pandas deal with all of it using
the ``.merge()`` method::

    # Add the age for each runner
    runners.merge(age, on="Runner")

In fact, much of what can be done in SQL 
`is also possible with pandas <https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html>`__.

``groupby()`` is a powerful method which splits a dataframe and aggregates data
in groups. To see what's possible, let's return to the Titanic dataset. Let's
test the old saying "Women and children first". We start by creating a new
column ``Child`` to indicate whether a passenger was a child or not, based on
the existing ``Age`` column. For this example, let's assume that you are a
child when you are younger than 12 years::

    titanic["Child"] = titanic["Age"] < 12

Now we can test the saying by grouping the data on ``Sex`` and then creating further sub-groups based on ``Child``::

    titanic.groupby(["Sex", "Child"])["Survival"].mean()

Here we chose to summarize the data by its mean, but many other common
statistical functions are available as dataframe methods, like
``std()``, ``min()``, ``max()``, ``cumsum()``, ``median()``, ``skew()``,
``var()`` etc. 



Exercises 2
-----------

.. challenge:: Analyze the Titanic passenger list dataset

    In the Titanic passenger list dataset, 
    investigate the family size of the passengers (i.e. the "SibSp" column).

    - What different family sizes exist in the passenger list? Hint: try the `unique` method 
    - What are the names of the people in the largest family group?
    - (Advanced) Create histograms showing the distribution of family sizes for 
      passengers split by the fare, i.e. one group of high-fare passengers (where 
      the fare is above average) and one for low-fare passengers 
      (Hint: instead of an existing column name, you can give a lambda function
      as a parameter to ``hist`` to compute a value on the fly. For example
      ``lambda x: "Poor" if df["Fare"].loc[x] < df["Fare"].mean() else "Rich"``).

.. solution:: Solution

    - Existing family sizes: ``df["SibSp"].unique()``
    - Names of members of largest family(ies): ``df[df["SibSp"] == 8]["Name"]``
    - ``df.hist("SibSp", lambda x: "Poor" if df["Fare"].loc[x] < df["Fare"].mean() else "Rich", rwidth=0.9)``




Time series superpowers
-----------------------

An introduction of pandas wouldn't be complete without mention of its 
special abilities to handle time series. To show just a few examples, 
we will use a new dataset of Nobel prize laureates::

    nobel = pd.read_csv("http://api.nobelprize.org/v1/laureate.csv")
    nobel.head()

This dataset has three columns for time, "born"/"died" and "year". 
These are represented as strings and integers, respectively, and 
need to be converted to datetime format::

    # the errors='coerce' argument is needed because the dataset is a bit messy
    nobel["born"] = pd.to_datetime(nobel["born"], errors ='coerce')
    nobel["died"] = pd.to_datetime(nobel["died"], errors ='coerce')
    nobel["year"] = pd.to_datetime(nobel["year"], format="%Y")

Pandas knows a lot about dates::

    print(nobel["born"].dt.day)
    print(nobel["born"].dt.year)
    print(nobel["born"].dt.weekday)
    
We can add a column containing the (approximate) lifespan in years rounded 
to one decimal::

    nobel["lifespan"] = round((nobel["died"] - nobel["born"]).dt.days / 365, 1)

and then plot a histogram of lifespans::

    nobel.hist(column='lifespan', bins=25, figsize=(8,10), rwidth=0.9)
    
Finally, let's see one more example of an informative plot 
produced by a single line of code::

    nobel.boxplot(column="lifespan", by="category")



Exercises 3
-----------

.. challenge:: Analyze the Nobel prize dataset

    - What country has received the largest number of Nobel prizes, and how many?
      How many countries are represented in the dataset? Hint: use the `describe()` method
      on the ``bornCountryCode`` column.
    - Create a histogram of the age when the laureates received their Nobel prizes.
      Hint: follow the above steps we performed for the lifespan. 
    - List all the Nobel laureates from your country.

    Now more advanced steps:
    
    - Now define an array of 4 countries of your choice and extract 
      only laureates from these countries::
      
          countries = np.array([COUNTRY1, COUNTRY2, COUNTRY3, COUNTRY4])
          subset = nobel.loc[nobel['bornCountry'].isin(countries)]

    - Use ``groupby`` to compute how many nobel prizes each country received in
      each category. The ``size()`` method tells us how many rows, hence nobel
      prizes, are in each group::

          nobel.groupby(['bornCountry', 'category']).size()

    - (Optional) Create a pivot table to view a spreadsheet like structure, and view it

        - First add a column “number” to the nobel dataframe containing 1’s 
          (to enable the counting below).          

        - Then create the pivot table::

            table = subset.pivot_table(values="number", index="bornCountry", columns="category", aggfunc=np.sum)
        
    - (Optional) Install the **seaborn** visualization library if you don't 
      already have it, and create a heatmap of your table::
      
          import seaborn as sns
          sns.heatmap(table,linewidths=.5);

    - Play around with other nice looking plots::
    
        sns.violinplot(y="year", x="bornCountry",inner="stick", data=subset);

      ::

        sns.swarmplot(y="year", x="bornCountry", data=subset, alpha=.5);

      ::

        subset_physchem = nobel.loc[nobel['bornCountry'].isin(countries) & (nobel['category'].isin(['physics']) | nobel['category'].isin(['chemistry']))]
        sns.catplot(x="bornCountry", y="year", col="category", data=subset_physchem, kind="swarm");

      ::
      
        sns.catplot(x="bornCountry", col="category", data=subset_physchem, kind="count");


Beyond the basics
-----------------

There is much more to Pandas than what we covered in this lesson. Whatever your
needs are, chances are good there is a function somewhere in its `API
<https://pandas.pydata.org/docs/>`__. And when there is not, you can always
apply your own functions to the data using `.apply`::

    from functools import lru_cache

    @lru_cache
    def fib(x):
        """Compute Fibonacci numbers. The @lru_cache remembers values we
        computed before, which speeds up this function a lot."""
        if x < 0:
            raise NotImplementedError('Not defined for negative values')
        elif x < 2:
            return x
        else:
            return fib(x - 2) + fib(x - 1)

    df = pd.DataFrame({'Generation': np.arange(100)})
    df['Number of Rabbits'] = df['Generation'].apply(fib)


.. keypoints::

   - Numpy provides a static array data structure, fast mathematical operations for 
     arrays and tools for linear algebra and random numbers
   - pandas dataframes are a good data structure for tabular data
   - Dataframes allow both simple and advanced analysis in very compact form 



