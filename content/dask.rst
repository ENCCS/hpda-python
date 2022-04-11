.. _dask:

Dask for scalable analytics
===========================

.. objectives::

   - Understand how Dask achieves parallelism
   - Learn a few common workflows with Dask


Intro
-----

An increasingly common problem we are facing today is that
the data we are analyzing is getter biger and bigger. So becomes 
the modern data analysis more and more expensive computationally 
as the data volume grows. The first difficult situation to deal with
is when the volume of data exceeds one's computer's RAM. 
Modern laptops/desktops have RAM about 10 GB. Beyond this threshold, 
some special care is required to carry out the analysis. 
The next threshold of difficulty is when the data can not even 
fit on the hard drive, which is about a couple of TB on a modern laptop.
In this situation, it is better to use HPC system, or cloud-based solution, 
and Dask is the tool that helps us easily extend our familiar data analysis 
tools to work with big data. In addition, dask also speed up 
our analysis by using mutiple CPU cores and making our work 
more efficiently on laptop, HPC and cloud platforms.


Dask Clusters
-------------

Dask needs a certain amount of computing resources in order to 
perform parallel computations. Dask Clusters have different names corresponding 
to different computing environments. for example: 

  - LocalCluster on laptop/desktop
  - PBSCluster on HPC
  - Kubernetes Cluster on the Cloud
 
Each cluster will be allocated with a certain number of "workers" associated with 
CPU and RAM and the dask scheduling system maps jobs to each worker for you.


We will focus on LocalCluster only during the course. XXXX
We can start a LocalCluster scheduler which makes use of all the cores and RAM 
we have on the machine. by: 

.. code-block:: python
    
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster()
    # explicitly connect to the cluster we just created
    client = Client(cluster)
    client


Or you can simply lauch a Client() call which is shorthand for what is described above.

.. code-block:: python

    from dask.distributed import Client
    client = Client()
    client



This sets up a scheduler in your local process along with a number of workers and threads per worker 
related to the number of cores in your machine.

Instantiating a cluster class like LocalCluster and then passing it to the Client is a common pattern. 
Cluster managers also provide useful utilities: for example if a cluster manager supports scaling, 
you can modify the number of workers manually or automatically based on workload.

.. code-block:: python
   
   cluster.scale(10)  # Sets the number of workers to 10
   cluster.adapt(minimum=1, maximum=10)  # Allows the cluster to auto scale to 10 when tasks are computed



Dask Collections
----------------

Dask provides dynamic parallel task scheduling and 
three main high-level collections:
  
  - dask.array: Parallel NumPy arrays
  - dask.dataframe: Parallel Pandas DataFrames
  - dask.bag: Parallel Python Lists 


Dask Arrays
^^^^^^^^^^^

A dask array looks and feels a lot like a numpy array. 
However, a dask array uses the so-called "lazy" execution mode, 
which allows one to build up complex, large calculations symbolically 
before turning them over the scheduler for execution. 


.. note::

   Not like a normal computation, in a lazy execution mode, all the computations needed to generate the data are symbolically represented, forming a queue of tasks mapped over data blocks. Nothing is actually computed until the actual numerical values are needed, e.g., to print results to your screen or write to disk. At that point, data is loaded into memory and computation proceeds in a streaming fashion, block-by-block. The actual computation is controlled by a multi-processing or thread pool, which allows Dask to take full advantage of multiple processors available on the computers.


.. code-block:: python

    import numpy as np
    shape = (1000, 4000)
    ones_np = np.ones(shape)
    ones_np
    ones_np.nbytes / 1e6


Now let's create the same array using dask's array interface.

.. code-block:: python

    import dask.array as da
    shape = (1000, 4000)
    ones = da.ones(shape)


This did not work, because a crucal difference with dask is that 
we must specify the "chunks" argument, which describes 
how the array is split up into sub-arrays.

.. code-block:: python

    import dask.array as da
    shape = (4000, 4000)
    chunk_shape = (1000, 1000)
    ones = da.ones(shape, chunks=chunk_shape)
    ones


So far, it is only a symbolic represetnation of the array. 
One way to trigger the computation is to call ``.compute()``:

.. code-block:: python

    ones.compute()


.. note::

   Plotting also triggers computation, since the actual values are needed.


We can visualize the symbolic operations by calling ``.visualize()``:

.. code-block:: python

    ones.visualize()

Let us calculate the sum of the dask array and visualize again:

.. code-block:: python

    sum_da = ones.sum()
    sum_da.visualize()



Dask Dataframe
^^^^^^^^^^^^^^

Dask Dataframes split a dataframe into partitions along an index. 
You can find additional details and examples here 
https://examples.dask.org/dataframe.html.


.. code-block:: python

    import dask.dataframe as dd
    server = 'https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?'
    query = 'service=WFS&version=2.0.0&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_Volcanoes&outputFormat=csv'

    # blocksize=None means use a single partion
    df = dd.read_csv(server+query, blocksize=None)

    # We only see the metadata, the actual data are only computed when requested.
    df

    # We can break up the table into 4 partions to map out to each core:
    df = df.repartition(npartitions=4)
    df

    # Let's say we want to know the minimum last eruption year for all volcanoes
    last_eruption_year_min = df.Last_Eruption_Year.min()
    last_eruption_year_min

    # Instead of getting the actual value we see dd.Scalar, which represents a recipe for actually calculating this value
    last_eruption_year_min.visualize()

    # To get the value call the 'compute method'
    # NOTE: this was slower than using pandas directly,,, for small data you often don't need to use parallel computing!
    last_eruption_year_min.compute()




Dask Delayed
^^^^^^^^^^^^

Sometimes problems don't fit into one of the collections like 
dask.array or dask.dataframe. In these cases, we can parallelize custom algorithms 
using dask.delayed interface. dask.delay allows users to delay function calls 
into a task graph with dependencies. If you have a problem that is paralellizable, 
but isn't as simple as just a big array or a big dataframe, then dask.delayed 
may be the right choice for you.


Consider the following example, these functions are very simple, and they sleep 
for a prescribed time to simulate real work.

#.. tab:: python

#    .. literalinclude:: example/delay.py 


Let us run the example first, one after the other in sequence:

.. sourcecode:: ipython

    %%time
    x = inc(1)
    y = dec(2)
    z = add(x, y)
    z


Note that the first two functions inc and dec don't depend on each other, 
we could have called them in parallel. We can call dask.delayed on these funtions 
to make them lazy and tasks into a graph which we will run later on parallel hardware.

.. sourcecode:: ipython

    import dask
    inc = dask.delayed(inc)
    dec = dask.delayed(dec)
    add = dask.delayed(add)

    %%time
    x = inc(1)
    y = dec(2)
    z = add(x, y)
    z

    z.visualize(rankdir='LR')

    %%time
    z.compute()


Let us extend the example a little bit more by 
applying the function on a data array using for loop:

.. code-block:: ipython

    def inc(x):
        time.sleep(4)
        return x + 1

    def dec(x):
        time.sleep(3)
        return x - 1

    def add(x, y):
        time.sleep(1)
        return x + y

    data = [1, 2, 3, 4, 5]

    output = []
    for x in data:
        a = inc(x)
        b = dec(x)
        c = add(a, b)
        output.append(c)

    total = sum(output)



.. challenge:: chunk size

    The following example calculate the mean value of a ramdom generated array. 
    Run the example and see the performance improvement by using dask.
    But what happens if we use different chunk sizes?

    - Try out with different chunk sizes:
      What happens if the dask chunks=(20000,20000)
      What happens if the dask chunks=(200,200)

   .. tabs::

      .. tab:: numpy

         .. literalinclude:: example/chunk_np.py
            :language: python

      .. tab:: dask

         .. literalinclude:: example/chunk_dask.py
            :language: python



.. challenge:: Data from climate simulation

    There are a couple of data in NetCDF files containing monthly global 2m air temperature. 


.. code-block:: python

    ds=xr.open_mfdataset('/home/x_qiali/qiang/hpda/airdata/tas*.nc', parallel=True)
    ds
    ds.tas
    dask.visualize(ds.tas) 
    
    tas_mean=ds.tas.mean(axis=0) 
    fig = plt.figure
    plt.imshow(tas_mean, cmap='RdBu_r');



SVD 
---

We can use dask to compute SVD of certain matrix.

.. code-block:: python

    import dask
    import dask.array as da
    X = da.random.random((200000, 100), chunks=(10000, 100))
    u, s, v = da.linalg.svd(X)
    dask.visualize(u, s, v)


We could also use approximate algorithm

.. code-block:: python

    import dask
    import dask.array as da
    X = da.random.random((10000, 10000), chunks=(2000, 2000)).persist()
    u, s, v = da.linalg.svd_compressed(X, k=5)
    dask.visualize(u, s, v)





How does Dask work?
-------------------


Common use cases
----------------



Comparison to Spark
-------------------

Dask has much in common with the 
[Apache Spark](https://spark.apache.org/).

- ref: https://docs.dask.org/en/stable/spark.html









.. keypoints::

   - 1
   - 2
   - 3
