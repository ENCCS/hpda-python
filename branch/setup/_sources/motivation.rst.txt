.. _motivation:

Motivation
==========

.. objectives::

   - Understand why high-performance data analytics techniques are needed

.. instructor-note::

   - 10 min teaching/type-along


Big data
--------

HPC has become an indispensable tool for  climate science community, with the advance of mordern computing sytems (especially with accerlarators like GPUs), more and more data is produced even faster rate and legacy software tools for data analysis can not handle them efficiently. This even becomes a obstacle to scientific progress in some cases. This course focuses on high performace data analysis, a subset of computing in which the raw data from either climate model simulation or observation is to be transformed into understanding following the steps below:

    1. read the raw data
    2. perform some operations on the data, from very simple (e.g. take the mean) to very complex (e.g. train a deep neural network)
    3. visualize the results or write the output into a file

The bulk of the content is devoted to the very basics of earth science data analysis using the modern scientific Python ecosystem, including Numpy, Scipy, Pandas, Xarray and  performace enhancement using numba, dask, cuPy.


Performance bottlenecks
-----------------------













.. keypoints::

   - 1
   - 2
   - 3
