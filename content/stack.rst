Python software stack
=====================

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

XXXX maybe move this to parallel part? 

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

Being one of the most fundemental part of python scientific computing ecosystem, 
NumPy offers comprehensive mathematical functions, random number generators, 
linear algebra routines, Fourier transforms, and more. Moreover, 
NumPy is based on well-optimized C code, which gives much better performace than Python. 
(XXXX add vectorization, for this reason)

- Standard Python is not well suitable for numerical computations
    - lists are very flexible but also slow to process in numerical
      computations

- Numpy adds a new **array** data type
    - static, multidimensional
    - fast processing of arrays
    - tools for linear algebra, random numbers, *etc.*

NDArray
^^^^^^^

The core of numpy is the numpy ndarray (n-dimensional array).
Compared to a python list, the numpy array is simialr in terms of serving as a data container.
Some differences between the two are: 

- numpy array can have multi dimensions, e.g. a 1-D array is a vector, a 2-D array is a matrix 
- numpy array can work fast only when all data elements are of the same type  
- numpy array can be fast when vectorized  
- numpy array is slower for certain operations, e.g. appending elements 


.. figure:: img/list-vs-array.svg
   :align: center
   :scale: 100 %


Numpy Data Type
^^^^^^^^^^^^^^^

The most common used data types (dtype) for numerical data (integer and floating-point) are listed here, 

For integers:

+-------------+----------------------------------+
| data type   | data range                       |
+=============+==================================+
| int8        | -2**7 to  2**7 -1                |
+-------------+----------------------------------+
| int16       | -32768 to 32767                  |
+-------------+----------------------------------+
| int32       | -2147483648 to 2147483647        |
+-------------+----------------------------------+
| int64       |    fff                           |
+-------------+----------------------------------+

For unsigned intergers:

+-------------+----------------------------------+
| data type   | data range                       |
+=============+==================================+
| uint8       | ffff                             |
+-------------+----------------------------------+
| uint16      | ffff                             |
+-------------+----------------------------------+
| uint32      | ffff                             |
+-------------+----------------------------------+
| uint64      | ffff                             |
+-------------+----------------------------------+

Be careful, once the data value is beyond the lower or upper bound of a certain data type, 
the value will be wrapped around and there is no warning:

.. code:: python

	np.array([255], np.uint8) + 1   # 2**8-1 is INT_MAX for uint8  
	array([0], dtype=uint8)



For floating-point numbers:

+-------------+----------------------------------+
| data type   | data range                       |
+=============+==================================+
| float16     | fff	                         |
+-------------+----------------------------------+
| float32     | fff     			 |
+-------------+----------------------------------+
| float64     | fff                              |
+-------------+----------------------------------+


Creating numpy arrays
^^^^^^^^^^^^^^^^^^^^^

One way to create a numpy array is to convert from a python list, but make sure that the list is homogeneous (same data type) 
otherwise you will downgrade the performace of numpy array. 
Since appending elements to an existing array is slow, it is a common practice to preallocate the necessary space with np.zeros or np.empty
when converting from a python list is not possible.


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

XXXX may add this to exercise and find out what is the differences between arange and linspace

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



Array Operations and Manipulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All the familiar arithemtic operators are applied on an element-by-element basis.

.. challenge:: Arithmetic

   .. tabs:: 

      .. tab:: 1D

             .. code-block:: py

			import numpy as np
                        a = np.array([1, 3, 5])
                        b = np.array([4, 5, 6])

             .. code-block:: py

			a + b

             .. figure:: img/np_add_1d_new.svg 

             .. code-block:: py

			a/b

             .. figure:: img/np_div_1d_new.svg 


      .. tab:: 2D

             .. code-block:: python

			import numpy as np
		        a = np.array([[1, 2, 3],
	               	   [4, 5, 6]])
		        b = np.array([10, 10, 10],
	               	   [10, 10, 10]])

			a + b                       # array([[11, 12, 13],
                                			 #        [14, 15, 16]]) 

             .. figure:: img/np_add_2d.svg 


Array Indexing
^^^^^^^^^^^^^^

Basic indexing is similar to python lists.
Note that advanced indexing creates copies of arrays.

.. challenge:: index


   .. tabs:: 

      .. tab:: 1D

             .. code-block:: py

			import numpy as np
                        data = np.array([1,2,3,4,5,6,7,8])

             .. figure:: img/np_ind_0.svg 

             .. code-block:: py

			     # integer indexing 

             .. figure:: img/np_ind_integer.svg 

             .. code-block:: py

			     # fancy indexing 

             .. figure:: img/np_ind_fancy.svg 

             .. code-block:: python

			     # boolean indexing 

             .. figure:: img/np_ind_boolean.svg 


      .. tab:: 2D

             .. code-block:: python

			     import numpy as np
			     data = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])

             .. figure:: img/np_ind2d_data.svg 

             .. code-block:: python

			     # integer indexing

             .. figure:: img/np_ind2d_integer.svg 

             .. code-block:: python

			     # fancy indexing 

             .. figure:: img/np_ind2d_fancy.svg 

             .. code-block:: python

			     # boolean indexing 


             .. figure:: img/np_ind2d_boolean.svg 


Array Aggregation
^^^^^^^^^^^^^^^^^
.. challenge:: aggregation

Apart from aggregate all values, one can also aggregate across the rows or columns by using the axis parameter:

   .. tabs:: 


      .. tab:: 2D

             .. code-block:: py

			     # max 

             .. figure:: img/np_max_2d.svg 


             .. code-block:: py

			     # sum 

             .. figure:: img/np_sum_2d.svg 

 
             .. code-block:: py

			     # axis 

             .. figure:: img/np_min_2d_ax0.svg 
             .. figure:: img/np_min_2d_ax1.svg 


Array Reshaping
^^^^^^^^^^^^^^^

.. challenge:: reshape

Sometimes, you need to change the dimension of an array. 
One of the most common need is to trasnposing the matrix 
during the dot product. Switching the dimensions of 
a numpy array is also quite common in more advanced cases.

             .. code-block:: py

			import numpy as np
                        data = np.array([1,2,3,4,6,7,8,9,10,11,12])

             .. figure:: img/np_reshape0.svg 

             .. code-block:: py

			    data.reshape(4,3)

             .. figure:: img/np_reshape43.svg 

             .. code-block:: py

			     data.reshape(3,4)
 
             .. figure:: img/np_reshape34.svg 


XXX put the fowlloing in exercices

add example, T of 1d array is not working
Use flatten as an alternative to ravel. What is the difference? (Hint: check which one returns a view and which a copy)

Views and copies of arrays
- Simple assignment creates references to arrays
- Slicing creates "views" to the arrays
- Use `copy()` for real copying of arrays

.. code-block:: python

   a = numpy.arange(10)
   b = a              # reference, changing values in b changes a
   b = a.copy()       # true copy

   c = a[1:4]         # view, changing c changes elements [1:4] of a
   c = a[1:4].copy()  # true copy of subarray


I/O with numpy
Random numbers

XXX put the above in exercices



Anatomy of NumPy array
^^^^^^^^^^^^^^^^^^^^^^

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












Pandas
------

Pandas is a Python package that provides high-performance and easy to use 
data structures and data analysis tools. Build on numpy array, pandas is 
particularly well suited to analyze tabular and time series data. 
Although numpy could deal with structured array (array with mixed data types), 
it is not efficient. 

The core data structures of pandas are Series and Dataframe. 
A pandas series is a one-dimensional numpy array with an index 
which we could use to access the data, while dataframe consists of 
a table of values with lables for each row and column.  
A dataframe can combine multiple data types, such as numbers and text, 
but the data in each column is of the same type. Each column of a dataframe is a 
`series object <https://pandas.pydata.org/docs/user_guide/dsintro.html#series>`__ - 
a dataframe is thus a collection of series.

.. image:: img/01_table_dataframe.svg




The open source community developing the pandas package has also created 
excellent documentation and training material, including: 

- a  `Getting started guide <https://pandas.pydata.org/getting_started.html>`__ 
  (including tutorials and a 10 minute flash intro)
- a `"10 minutes to pandas" <https://pandas.pydata.org/docs/user_guide/10min.html#min>`__
  tutorial
- thorough `Documentation <https://pandas.pydata.org/docs/>`__ containing a user guide, 
  API reference and contribution guide
- a `cheatsheet <https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf>`__ 
- a `cookbook <https://pandas.pydata.org/docs/user_guide/cookbook.html#cookbook>`__.


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



tidy data format
^^^^^^^^^^^^^^^^

Let's first look at the following two tables:

.. challenge:: 1500m Running event

   .. tabs:: 

      .. tab:: untidy data format

             .. code-block:: py

			     Runner  400  800  1200  1500
			0  Runner 1   64  128   192   240
			1  Runner 2   80  160   240   300
			2  Runner 3   96  192   288   360

      .. tab:: tidy data format

             .. code-block:: python

			      Runner distance  time
			0   Runner 1      400    64
			1   Runner 2      400    80
			2   Runner 3      400    96
			3   Runner 1      800   128
			4   Runner 2      800   160
			5   Runner 3      800   192
			6   Runner 1     1200   192
			7   Runner 2     1200   240
			8   Runner 3     1200   288
			9   Runner 1     1500   240
			10  Runner 2     1500   300
			11  Runner 3     1500   360


Most tabular data is either in a tidy format or a untidy format 
(some people refer them as the long format or the wide format). 

In short, in an untidy (wide) format, each row represents an observation 
consisting of multiple variables and each variable has its own column. 
This is very intuitive and easy for us (human beings) to understand 
and  make comparisons across different variables, calculate statistics, etc.  
In a tidy (long) format , i.e. column-oriented format, each row represents 
only one variable of the observation, and can be considered "computer readable".

Both formats have their own merits and you need to know 
which one suits your analysis. For example, if you are dealing with matrices, 
you would not want to store them as rows and columns, but as 
a two-dimensional array using untidy format. On the other hand, 
if you need to add new data or remove old data frequently from the table 
in a relational database, the tidy format may be the choice. 
Another case is that there are certain visualization tools 
which take data in the tidy format, e,g, ggplot, seaborn.

When it comes to data analysis using pandas, the tidy format is recommended: 

 - each column can be stored as a vector and this not only saves memory 
   but also allows for vectorized calculations which are much faster.
 - it's easier to filter, group, join and aggregate the data

.. note:: 

The name "tidy data" comes from Wickham’s paper (2014) which describes the ideas in great detail.


data pre-processing
^^^^^^^^^^^^^^^^^^^

In real applications, some data pre-processing have to be performed 
before one can perform useful analysis. There is no fixed list of 
what these pre-processings are, but in general the following steps are involved:

- data cleaning
- data reshaping
- data 


data cleaning
~~~~~~~~~~~~~

A couple of essential  data cleaning processes include 
but not limited to the following:

- data renaming
- data reordering
- data type converting
- handling of duplicating data, missing data, invalid data


add examples 
https://pandas.pydata.org/docs/user_guide/missing_data.html


data Reshaping
~~~~~~~~~~~~~~

Once data cleaning is done, we will reach the data reshaping phase. 
By reorganising the data, one could make the subsequent data operations easier.

pivoting
********

Create a data frame first

.. code:: python

	df = pd.DataFrame(
    	{
       	 "foo": ["one", "one", "one", "two", "two", "two"] ,
       	 "bar": ["A", "B", "C"] * 2,
       	 "baz": np.linspace(1,6,6).astype(int),
       	 "zoo": ["x","y","z","q","w","t"]
    	}
	)


To select out everything for variable ``A`` we could do:

.. code:: python

   filtered = df[df["bar"] == "A"]
   filtered

But suppose we would like to represent the table in such a way that
the ``columns`` are the unique variables from 'bar' and the ``index`` from 'foo'. 
To reshape the data into this form, we use the :meth:`DataFrame.pivot` 
method (also implemented as a top level function :func:`~pandas.pivot`):

.. code:: python

   pivoted = df.pivot(index="foo", columns="bar", values="baz")
   pivoted

.. image:: img/reshaping_pivot.png

If the ``values`` argument is omitted, and the input :class:`DataFrame` has more than
one column of values which are not used as column or index inputs to :meth:`~DataFrame.pivot`,
then the resulting "pivoted" :class:`DataFrame` will have :ref:`hierarchical columns
<advanced.hierarchical>` whose topmost level indicates the respective value
column:

.. code:: python

   df["value2"] = df["value"] * 2
   pivoted = df.pivot(index="date", columns="variable")
   pivoted

You can then select subsets from the pivoted :class:`DataFrame`:

.. code:: python

   pivoted["value2"]

Note that this returns a view on the underlying data in the case where the data
are homogeneously-typed.

.. note::
   :func:`~pandas.pivot` will error with a ``ValueError: Index contains duplicate
   entries, cannot reshape`` if the index/column pair is not unique. In this
   case, consider using :func:`~pandas.pivot_table` which is a generalization
   of pivot that can handle duplicate values for one index/column pair.

stacking and unstacking
***********************

Closely related to the pivot() method are the related 
stack() and unstack() methods available on Series and DataFrame. 
These methods are designed to work together with MultiIndex objects.

The stack() function "compresses" a level in 
the DataFrame columns to produce either:

 - A Series, in the case of a simple column Index.
 - A DataFrame, in the case of a MultiIndex in the columns.

If the columns have a MultiIndex, you can choose which level to stack. 
The stacked level becomes the new lowest level in a MultiIndex on the columns:

.. code:: python

	tuples = list(
    	zip(
        	*[
            	["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            	["one", "two", "one", "two", "one", "two", "one", "two"],
        	]
    	)
	)

	columns = pd.MultiIndex.from_tuples(
    	[
        	("bar", "one"),
	        ("bar", "two"),
        	("baz", "one"),
	        ("baz", "two"),
        	("foo", "one"),
	        ("foo", "two"),
	        ("qux", "one"),
        	("qux", "two"),
	    ],
	    names=["first", "second"]
	)

	index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])


Note: there are other ways to generate MultiIndex, e.g. 

.. code:: python

	index = pd.MultiIndex.from_product(
    	[("bar", "baz", "foo", "qux"), ("one", "two")], names=["first", "second"]
	)

	df = pd.DataFrame(np.linspace(1,16,16).astype(int).reshape(8,2), index=index, columns=["A", "B"])
	df
	df2 = df[:4]
	df2
	stacked=df2.stack()

.. image:: img/reshaping_stack.png 

The unstack() method performs the inverse operation of stack(), 
and by default unstacks the last level. If the indexes have names, 
you can use the level names instead of specifying the level numbers.



stacked.unstack()

.. image:: img/reshaping_unstack.png 


stacked.unstack(1)
or 
stacked.unstack("second")

.. image:: img/reshaping_unstack_1.png 
.. image:: img/reshaping_unstack_0.png 



groupby
^^^^^^^

As we know, when it is about  mathematical oprations on arrays of numerical data, Numpy does best.
Pandas works very well with numpy when aggregating dataframes.

Pandas has a strong built-in understanding of time. With datasets indexed by a pandas DateTimeIndex, 
we can easily group and resample the data using common time units.

The groupby() method is an amazingly powerful function in pandas. But it is also complicated to use and understand.
Together with pivot() / stack() / unstack() and the basic Series and DataFrame statistical functions, 
groupby can produce some very expressive and fast data manipulations.

.. image:: img/groupby.png 

The workflow of groubpy method can be divided into three general steps:

- Splitting: Partition the data into different groups based on some criterion.
- Applying: Do some caclulation within each group. 
  Different kinds of calulations might be aggregation, transformation, filtration
- Combining: Put the results back together into a single object.

data aggregation
~~~~~~~~~~~~~~~~

Here we will go through the following example 

.. code:: python

	import urllib.request
	import pandas as pd

	header_url = 'ftp://ftp.ncdc.noaa.gov/pub/data/uscrn/products/daily01/HEADERS.txt'
	with urllib.request.urlopen(header_url) as response:
	    data = response.read().decode('utf-8')
	lines = data.split('\n')
	headers = lines[1].split(' ')

	ftp_base = 'ftp://ftp.ncdc.noaa.gov/pub/data/uscrn/products/daily01/'
	dframes = []
	for year in range(2016, 2019):
	    data_url = f'{year}/CRND0103-{year}-NY_Millbrook_3_W.txt'               
	    df = pd.read_csv(ftp_base + data_url, parse_dates=[1],
	                     names=headers,header=None, sep='\s+',
        	             na_values=[-9999.0, -99.0])
	    dframes.append(df)

	df = pd.concat(dframes)
	df = df.set_index('LST_DATE')
	df.head()
	df['T_DAILY_MEAN'] # or df.T_DAILY_MEAN
	df['T_DAILY_MEAN'].aggregate([np.max,np.min,np.mean])
	df.index   # df.index is a pandas DateTimeIndex object.

.. code:: python

	gbyear=df.groupby(df.index.year)
	gbyear.T_DAILY_MEAN.head()
	gbyear.T_DAILY_MEAN.max()
	gbyear.T_DAILY_MEAN.aggregate(np.max)
	gbyear.T_DAILY_MEAN.aggregate([np.min, np.max, np.mean, np.std])


now let us calculate the monthly mean values

.. code:: python

	gb=df.groupby(df.index.month)
	df.groupby('T_DAILY_MEAN')  # or  df.groupby(df.T_DAILY_MEAN)
	monthly_climatology = df.groupby(df.index.month).mean()
	monthly_climatology

Each row in this new dataframe respresents the average values for the months (1=January, 2=February, etc.)

.. code:: python

	monthly_T_climatology = df.groupby(df.index.month).aggregate({'T_DAILY_MEAN': 'mean',
                                                              'T_DAILY_MAX': 'max',
                                                              'T_DAILY_MIN': 'min'})
	monthly_T_climatology.head()
	daily_T_climatology = df.groupby(df.index.dayofyear).aggregate({'T_DAILY_MEAN': 'mean',
                                                            'T_DAILY_MAX': 'max',
                                                            'T_DAILY_MIN': 'min'})
	def standardize(x):
	    return (x - x.mean())/x.std()
	anomaly = df.groupby(df.index.month).transform(standardize)


data transfromation
~~~~~~~~~~~~~~~~~~~

The key difference between aggregation and transformation is that 
aggregation returns a smaller object than the original, 
indexed by the group keys, while transformation returns an object 
with the same index (and same size) as the original object. 

In this example, we standardize the temperature so that 
the distribution has zero mean and unit variance. 
We do this by first defining a function called standardize 
and then passing it to the transform method.


.. code:: python

	transformed = df.groupby(lambda x: x.year).transform(
	    lambda x: (x - x.mean()) / x.std()
	)
	grouped = df.groupby(lambda x: x.year)
	grouped_trans = transformed.groupby(lambda x: x.year)




Exercises 1
^^^^^^^^^^^

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


Exercises 2
^^^^^^^^^^^

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


Exercises 3
^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^

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





Scipy
-----

.. seealso::

   * Main article: `SciPy documentation <https://docs.scipy.org/doc/scipy/reference/>`__



SciPy is a library that builds on top of NumPy. It contains a lot of
interfaces to battle-tested numerical routines written in Fortran or
C, as well as python implementations of many common algorithms.

What's in SciPy?
^^^^^^^^^^^^^^^^

Briefly, it contains functionality for

- Special functions (Bessel, Gamma, etc.)
- Numerical integration
- Optimization
- Interpolation
- Fast Fourier Transform (FFT)
- Signal processing
- Linear algebra (more complete than in NumPy)
- Sparse matrices
- Statistics
- More I/O routine, e.g. Matrix Market format for sparse matrices,
  MATLAB files (.mat), etc.

Many (most?) of these are not written specifically for SciPy, but use
the best available open source C or Fortran libraries.  Thus, you get
the best of Python and the best of compiled languages.

Most functions are documented ridiculously well from a scientific
standpoint: you aren't just using some unknown function, but have a
full scientific description and citation to the method and
implementation.



.. keypoints::

   - When you need advance math or scientific functions, let's just
     admit it: you do a web search first.
   - But when you see something in SciPy come up, you know your
     solutions are in good hands.
