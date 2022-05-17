.. _motivation:

Motivation
==========

.. objectives::

   - Become familiar with the term "big data"
   - Know what to expect from this course

.. instructor-note::

   - 10 min teaching/type-along


Big data
--------

.. discussion:: How large is your data?

   How large is the data you are working with? Are you experiencing performance bottlenecks 
   when you try to analyse it?

*"Big data refers to data sets that are too large or complex to be dealt with by 
traditional data-processing application software. [...]
Big data analysis challenges include capturing data, data storage, data analysis, 
search, sharing, transfer, visualization, querying, updating, information privacy, 
and data source."* (from `Wikipedia <https://en.wikipedia.org/wiki/Big_data>`__)

"Big data" is a current buzzword used heavily in the tech industry, but many scientific 
research communities are increasingly adopting high-throughput data production methods 
which lead to very large datasets. One driving force behind this development is the advent 
of powerful machine learning methods which enable researchers to derive novel scientific 
insights from large datasets. Another is the strong development of high perfomance 
computing (HPC) hardware and the accompanying development of software libraries and 
packages which can efficiently take advantage of the hardware.

This course focuses on high-performace data analytics (HPDA), a subset of high-performance 
computing which focuses on working with large data. 
The data can come from either computer models and simulations or from experiments and 
observations, and the goal is to preprocess, analyse and visualise it to generate 
scientific results.


Python
------

.. discussion:: Performance bottlenecks in Python

   Have you ever written Python scripts that look something like this?

   .. code-block:: python

      f = open("mydata.dat", "r")
      for line in f.readlines():
          fields = line.split(",")
          x, y, z = fields[1], fields[2], fields[3]
          # some analysis with x, y and z
      f.close()

   Compared to C/C++/Fortran, this for-loop will probably be orders of magnitude slower!
   
Despite early design choices of the Python language which made it significantly slower 
than conventional HPC languages, a rich and growing ecosystem of open source libraries 
have established Python as an industry-standard programming language for working with 
data on all levels of the data analytics pipeline.
These range from generic numerical libraries to special-purpose and/or domain-specific 
packages. This lesson is focused on introducing modern packages from the Python 
ecosystem to work with large data. Specifically, we will learn to use:

- Numpy 
- Scipy
- Pandas
- Xarray
- Numba
- Cython
- Snakemake
- multithreading
- multiprocessing
- MPI4Py
- Dask


What you will learn
-------------------

This lesson provides a broad overview of methods to work with large 
datasets using tools and libraries from the Python ecosystem. Since this field is fairly 
extensive we will not have time to go into much depth. Instead, the objective is to expose 
just enough details on each topic for you to get a good idea of the big picture and an 
understanding of what combination of tools and libraries will work well for your particular 
use case.

Specifically, the lesson covers:

- Tools for efficiently storing data and writing/reading data to/from disk
- How to share datasets and mint digital object identifiers (DOI)
- Main methods of efficiently working with tabular data and multidimensional arrays
- How to measure performance and boost performance of time consuming Python functions
- Various methods to parallelise Python code
- How to port Python code to run on graphical processing units (GPUs)

The lesson does not cover the following:

- Visualisation techniques
- Machine learning 


.. keypoints::

   - Datasets are getting larger across nearly all scientific and engineering domains
   - The Python ecosystem has many libraries and packages for working with big data efficiently
