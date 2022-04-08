.. _stack:

Python software stack
=====================

.. objectives::

   - Understand limitations of Python's standard library for large data processing
   - Understand vectorization and basic array computing in NumPy
   - Learn to use several of NumPy's numerical computing tools 
   - Learn to use data structures and analysis tools from Pandas

.. 

  This episode is inspired by and derived from this 
  `repository on HPC-Python from CSC <https://github.com/csc-training/hpc-python>`__ and 
  this `Python for Scientific Computing lesson <https://aaltoscicomp.github.io/python-for-scicomp/>`__.

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

In summary, the flexibility and dynamic nature of Python, that enhances
the programmer productivity greatly, is also the main cause for the
performance problems. Flexibility comes with a price! Fortunately, as we
discuss in the course, many of the bottlenecks can be circumvented.


NumPy
-----

Being one of the most fundamental part of python scientific computing ecosystem, 
NumPy offers comprehensive mathematical functions, random number generators, 
linear algebra routines, Fourier transforms, and more. NumPy is based on well-optimized 
C code, which gives much better performace than Python. In particular, by using homogeneous 
data structures, NumPy *vectorizes* mathematical operations where fast pre-compiled code 
can be applied to a sequence of data instead of using traditional ``for`` loops.

Arrays
^^^^^^

The core of numpy is the numpy ``ndarray`` (n-dimensional array).
Compared to a python list, the numpy array is similar in terms of serving as a data container.
Some differences between the two are: 

- ndarrays can have multi dimensions, e.g. a 1-D array is a vector, a 2-D array is a matrix 
- ndarrays are fast only when all data elements are of the same type 
- ndarrays are fast when vectorized  
- ndarrays are slower for certain operations, e.g. appending elements 


.. figure:: img/list-vs-array.svg
   :align: center
   :scale: 100 %



Data types
^^^^^^^^^^

NumPy supports a much greater variety of numerical types (``dtype``) than Python does.
There are 5 basic numerical types representing booleans (``bool``), integers (``int``), 
unsigned integers (``uint``) floating point (``float``) and complex (``complex``). 

.. code-block:: python

   import numpy as np

   # create float32 variable
   x = np.float32(1.0)
   # array with uint8 unsigned integers
   z = np.arange(3, dtype=np.uint8)
   # convert array to floats
   z.astype(float)

Creating numpy arrays
^^^^^^^^^^^^^^^^^^^^^

One way to create a numpy array is to convert from a python list, but make sure that the list is homogeneous 
(same data type) otherwise you will downgrade the performace of numpy array. 
Since appending elements to an existing array is slow, it is a common practice to preallocate the necessary space 
with ``np.zeros`` or ``np.empty`` when converting from a python list is not possible.

.. code-block:: python

   import numpy as np
   a = np.array((1, 2, 3, 4), float)
   a
   # array([ 1., 2., 3., 4.])

   list1 = [[1, 2, 3], [4, 5, 6]]
   mat = np.array(list1, complex)
   mat
   # array([[ 1.+0.j, 2.+0.j, 3.+0.j],
   #       [ 4.+0.j, 5.+0.j, 6.+0.j]])

   mat.shape
   # (2, 3)

   mat.size
   # 6

Helper functions
~~~~~~~~~~~~~~~~

``arange`` and ``linspace`` can generate ranges of numbers:

.. code-block:: python

    a = np.arange(10)
    a
    # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    b = np.arange(0.1, 0.2, 0.02)
    b
    # array([0.1 , 0.12, 0.14, 0.16, 0.18])

    c = np.linspace(-4.5, 4.5, 5)
    c
    # array([-4.5 , -2.25, 0. , 2.25, 4.5 ])


Array with given shape initialized to ``zeros``, ``ones``, arbitrary value (``full``)
or unitialized (``empty``):

.. code-block:: python

   a = np.zeros((4, 6), float)
   a.shape
   # (4, 6)

   b = np.ones((2, 4))
   b
   # array([[ 1., 1., 1., 1.],
   #       [ 1., 1., 1., 1.]])
	   
   c = np.full((2, 3), 4.2)
   c
   # array([[4.2, 4.2, 4.2],
   #       [4.2, 4.2, 4.2]])

   d = np.empty((2, 2))
   # array([[0.00000000e+000, 1.03103236e-259],
   #       [0.00000000e+000, 9.88131292e-324]])

Similar arrays as an existing array:

.. code-block:: python

   a = numpy.zeros((4, 6), float)
   b = numpy.empty_like(a)
   c = numpy.ones_like(a)
   d = numpy.full_like(a, 9.1)



Array Operations and Manipulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All the familiar arithemtic operators are applied on an element-by-element basis.

.. tabs:: 

   .. tab:: 1D

      .. code-block:: python

         import numpy as np
         a = np.array([1, 3, 5])
         b = np.array([4, 5, 6])
         
         a + b

      .. figure:: img/np_add_1d_new.svg 

      .. code-block:: py

         a/b
         
      .. figure:: img/np_div_1d_new.svg


   .. tab:: 2D
    
      .. code-block:: python

         import numpy as np
	      a = np.array([[1, 2, 3], [4, 5, 6]])
         b = np.array([10, 10, 10], [10, 10, 10]])

         a + b                       # array([[11, 12, 13],
                                     #        [14, 15, 16]]) 

      .. figure:: img/np_add_2d.svg 


Array Indexing
^^^^^^^^^^^^^^

Basic indexing is similar to python lists.
Note that advanced indexing creates copies of arrays.

.. tabs:: 

   .. tab:: 1D

      .. code-block:: py

         import numpy as np
         data = np.array([1,2,3,4,5,6,7,8])

      .. figure:: img/np_ind_0.svg 

      **Integer indexing:**

      .. figure:: img/np_ind_integer.svg 

      **Fancy indexing:**

      .. figure:: img/np_ind_fancy.svg 

      **Boolean indexing:**

      .. figure:: img/np_ind_boolean.svg 


   .. tab:: 2D

      .. code-block:: python

         import numpy as np
         data = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])

      .. figure:: img/np_ind2d_data.svg 

      **Integer indexing:**

      .. figure:: img/np_ind2d_integer.svg 

      **Fancy indexing:**

      .. figure:: img/np_ind2d_fancy.svg 

      **Boolean indexing:**

      .. figure:: img/np_ind2d_boolean.svg 


Array Aggregation
^^^^^^^^^^^^^^^^^

Apart from aggregating values, one can also aggregate across rows or columns by using the ``axis`` parameter:

.. code-block:: py

   import numpy as np
   data = np.array([[0, 1, 2], [3, 4, 5]])

.. figure:: img/np_max_2d.svg 

.. figure:: img/np_sum_2d.svg 

.. figure:: img/np_min_2d_ax0.svg 

.. figure:: img/np_min_2d_ax1.svg 


Array Reshaping
^^^^^^^^^^^^^^^

Sometimes, you need to change the dimension of an array. 
One of the most common need is to trasnposing the matrix 
during the dot product. Switching the dimensions of 
a NumPy array is also quite common in more advanced cases.

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


Views and copies of arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Simple assignment creates **references** to arrays
- Slicing creates **views** to the arrays
- Use ``copy`` for real copying of arrays

.. code-block:: python

   a = numpy.arange(10)
   b = a              # reference, changing values in b changes a
   b = a.copy()       # true copy

   c = a[1:4]         # view, changing c changes elements [1:4] of a
   c = a[1:4].copy()  # true copy of subarray


I/O with numpy
^^^^^^^^^^^^^^

- Numpy provides functions for reading data from file and for writing data
  into the files
- Simple text files

    - :meth:`numpy.loadtxt`
    - :meth:`numpy.savetxt`
    - Data in regular column layout
    - Can deal with comments and different column delimiters


Random numbers
^^^^^^^^^^^^^^

- The module ``numpy.random`` provides several functions for constructing
  random arrays

   - :meth:`random`: uniform random numbers
   - :meth:`normal`: normal distribution
   - :meth:`choice`: random sample from given array
   - ...

.. code-block:: python

    import numpy.random as rnd
    rnd.random((2,2))
    # array([[ 0.02909142, 0.90848 ],
    #       [ 0.9471314 , 0.31424393]])

    rnd.choice(numpy.arange(4), 10)
    # array([0, 1, 1, 2, 1, 1, 2, 0, 2, 3])

Polynomials
^^^^^^^^^^^

- Polynomial is defined by an array of coefficients p
  ``p(x, N) = p[0] x^{N-1} + p[1] x^{N-2} + ... + p[N-1]``
- For example:

    - Least square fitting: :meth:`numpy.polyfit`
    - Evaluating polynomials: :meth:`numpy.polyval`
    - Roots of polynomial: :meth:`numpy.roots`

.. code-block:: python

    x = np.linspace(-4, 4, 7)
    y = x**2 + rnd.random(x.shape)

    p = np.polyfit(x, y, 2)
    p
    # array([ 0.96869003, -0.01157275, 0.69352514])


Linear algebra
^^^^^^^^^^^^^^

- Numpy can calculate matrix and vector products efficiently: :meth:`dot`,
  :meth:`vdot`, ...
- Eigenproblems: :meth:`linalg.eig`, :meth:`linalg.eigvals`, ...
- Linear systems and matrix inversion: :meth:`linalg.solve`, :meth:`linalg.inv`

.. code-block:: python

    A = np.array(((2, 1), (1, 3)))
    B = np.array(((-2, 4.2), (4.2, 6)))
    C = np.dot(A, B)

    b = np.array((1, 2))
    np.linalg.solve(C, b) # solve C x = b
    # array([ 0.04453441, 0.06882591])

- Normally, NumPy utilises high performance libraries in linear algebra
  operations
- Example: matrix multiplication C = A * B matrix dimension 1000

    - pure python:           522.30 s
    - naive C:                 1.50 s
    - numpy.dot:               0.04 s
    - library call from C:     0.04 s



Pandas
------

Pandas is a Python package that provides high-performance and easy to use 
data structures and data analysis tools. Built on NumPy arrays, pandas is 
particularly well suited to analyze tabular and time series data. 
Although NumPy could in principle deal with structured arrays 
(arrays with mixed data types), it is not efficient. 

The core data structures of pandas are Series and Dataframes.

- A pandas **series** is a one-dimensional numpy array with an index 
  which we could use to access the data 
- A **dataframe** consist of a table of values with labels for each row and column.  
  A dataframe can combine multiple data types, such as numbers and text, 
  but the data in each column is of the same type. 
- Each column of a dataframe is a series object - a dataframe is thus a collection of series.

.. image:: img/01_table_dataframe.svg


Tidy vs untidy data
^^^^^^^^^^^^^^^^^^^

Let's first look at the following two tables:

.. challenge:: 1500m Running event

   .. tabs:: 

      .. tab:: Untidy data format

         .. code-block:: py

	   		     Runner  400  800  1200  1500
	   		0  Runner 1   64  128   192   240
	   		1  Runner 2   80  160   240   300
	   		2  Runner 3   96  192   288   360

      .. tab:: Tidy data format

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

In short: 

- in an untidy (wide) format, each row represents an observation 
  consisting of multiple variables and each variable has its own column. 
  This is very intuitive and easy for us (human beings) to understand 
  and  make comparisons across different variables, calculate statistics, etc.  

- In a tidy (long) format , i.e. column-oriented format, each row represents 
  only one variable of the observation, and can be considered "computer readable".

When it comes to data analysis using pandas, the tidy format is recommended: 

- Each column can be stored as a vector and this not only saves memory 
  but also allows for vectorized calculations which are much faster.
- It's easier to filter, group, join and aggregate the data.

The name "tidy data" comes from `Wickham’s paper (2014) <https://vita.had.co.nz/papers/tidy-data.pdf>`__ 
which describes the ideas in great detail.
This image from Hadley Wickham’s book *R for Data Science* visualizes the idea:

.. figure:: img/tidy_data.png

Data analysis workflow
^^^^^^^^^^^^^^^^^^^^^^

Pandas is a powerful tool for all steps of a data analysis pipeline, 
including 

- Downloading and reading in data sets
- Exploring the data
- Pre-processing and cleaning data

  - renaming, reshaping, reordering, type conversion, handling duplicate/missing/invalid data

- Analysis


Let us explore some of the capabilities.
We begin by creating a dataframe:
Let's get a flavor of what we can do with pandas. We will be working with an
example dataset containing the passenger list from the Titanic, which is often used in 
Kaggle competitions and data science tutorials. First step is to load pandas and download 
the dataset into a dataframe:

.. code-block:: python

   import pandas as pd

   url = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
   titanic = pd.read_csv(url, index_col='Name')

We can now view the dataframe to get an idea of what it contains and
print some summary statistics of its numerical data::

    # print the first 5 lines of the dataframe
    titanic.head()  
    
    # print summary statistics for each column
    titanic.describe()  


Ok, so we have information on passenger names, survival (0 or 1), age, 
ticket fare, number of siblings/spouses, etc. With the summary statistics we 
see that the average age is 29.7 years, maximum ticket price is 512 USD, 
38\% of passengers survived, etc.

Let's say we're interested in the survival probability of different age groups. 
With two one-liners, we can find the average age of those who survived or didn't survive, 
and plot corresponding histograms of the age distribution:

.. code-block:: python

   print(titanic.groupby("Survived")["Age"].mean())

::

    titanic.hist(column='Age', by='Survived', bins=25, figsize=(8,10), 
                 layout=(2,1), zorder=2, sharex=True, rwidth=0.9);
    


.. code-block:: python

	df = pd.DataFrame(
    	{
       	 "foo": ["one", "one", "one", "two", "two", "two"] ,
       	 "bar": ["A", "B", "C"] * 2,
       	 "baz": np.linspace(1,6,6).astype(int),
       	 "zoo": ["x","y","z","q","w","t"]
    	}
	)
   df

Summary statistics of numerical columns:

.. code-block:: python

   df.describe()

To select out everything for variable ``A`` we could do:

.. code-block:: python

   filtered = df[df["bar"] == "A"]
   filtered

But suppose we would like to represent the table in such a way that
the ``columns`` are the unique variables from "bar" and the ``index`` from "foo". 
To reshape the data into this form, we use the :meth:`DataFrame.pivot` 
method (also implemented as a top level function :func:`~pandas.pivot`):

.. code:: python

   pivoted = df.pivot(index="foo", columns="bar", values="baz")
   pivoted

.. image:: img/reshaping_pivot.png

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

See also
--------

- Pandas  `getting started guide <https://pandas.pydata.org/getting_started.html>`__ 
  (including tutorials and a 10 minute flash intro)
- Pandas `documentation <https://pandas.pydata.org/docs/>`__ containing a user guide, 
  API reference and contribution guide.
- Pandas `cheatsheet <https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf>`__ 
- Pandas `cookbook <https://pandas.pydata.org/docs/user_guide/cookbook.html#cookbook>`__.

.. keypoints::


