.. _GPU-computing:

GPU computing
=============

.. questions::

   - Why use GPUs?
   - What is different about GPUs?
   - What is the programming model?

.. objectives::

   - Understand GPU architecture (resources available to programmer) 
   - Understand execution model 
   - Get an overview of different options for GPU computing in Python
   - Understand what types of computation is suitable for GPUs
   - Learn the basics of Numba for GPUs and PyCUDA

.. instructor-note::

   - 70 min teaching/type-along
   - 40 min exercises


.. prereq::

   1. Basic C or FORTRAN
   2. Basic knowledge about processes and threads


GPU Intro
---------

Several important terms in the topic of GPU programming are listed here:

- *host*: the CPU
- *device*: the GPU
- *host memory*: the system main memory of the CPU
- *device memory*: GPU onboard memory
- *kernels*: a GPU function launched by the host and executed on the device
- *device function*: a GPU function executed on the device which can only be
  called from the device (i.e. from a kernel or another device function)



Moore's law
~~~~~~~~~~~

The number of transistors in a dense integrated circuit doubles about every two years.
More transistors means smaller size of a single element, so higher core frequency can be achieved.
However, power consumption scales as frequency in third power, so the growth in the core frequency has slowed down significantly.
Higher performance of a single node has to rely on its more complicated structure and still can be achieved with SIMD, branch prediction, etc.

.. figure:: img/microprocessor-trend-data.png
   :align: center

   The evolution of microprocessors.
   The number of transistors per chip increase every 2 years or so.
   However it can no longer be explored by the core frequency due to power consumption limits.
   Before 2000, the increase in the single core clock frequency was the major source of the increase in the performance.
   Mid 2000 mark a transition towards multi-core processors.

Achieving performance has been based on two main strategies over the years:

    - Increase the single processor performance: 

    - More recently, increase the number of physical cores.


Why use GPUs?
~~~~~~~~~~~~~

The Graphics processing units (GPU) have been the most common accelerators during the last few years. The term *GPU* sometimes is used interchangeably with the term *accelerator*. The Graphics Processing Unit (GPU) provides much higher instruction throughput and memory bandwidth than the CPU within a similar price and power envelope.



How do GPUs differ from CPUs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPUs and GPUs were designed with different goals in mind. While the CPU is designed to excel at executing a sequence of operations, called a thread, as fast as possible and can execute a few tens of these threads in parallel, the GPU is designed to excel at executing many thousands of them in parallel. GPUs were initially developed for highly-parallel task of graphic processing and therefore designed such that more transistors are devoted to data processing rather than data caching and flow control. More transistors dedicated to data processing is beneficial for highly parallel computations; the GPU can hide memory access latencies with computation, instead of relying on large data caches and complex flow control to avoid long memory access latencies, both of which are expensive in terms of transistors.



.. figure:: img/gpu_vs_cpu.png
   :align: center

   A comparison of the CPU and GPU architecture.
   CPU (left) has complex core structure and pack several cores on a single chip.
   GPU cores are very simple in comparison, they also share data and control between each other.
   This allows to pack more cores on a single chip, thus achieving very hich compute density.

.. list-table::  
   :widths: 100 100
   :header-rows: 1

   * - CPU
     - GPU
   * - General purpose
     - Highly specialized for parallelism
   * - Good for serial processing
     - Good for parallel processing
   * - Great for task parallelism
     - Great for data parallelism
   * - Low latency per thread
     - High-throughput
   * - Large area dedicated cache and control
     - Hundreds of floating-point execution units


Summary
~~~~~~~

- GPUs are highly parallel devices that can execute certain parts of the program in many parallel threads.

- CPU controls the works flow and makes all the allocations and data transfers.

- In order to use the GPU efficiently, one has to split their the problem  in many parts that can run simultaneously.


Supported Python features in CUDA Python
--------------------------------------------------------

This page lists the Python features supported in the CUDA Python. This includes all kernel and device functions compiled with @cuda.jit and other higher level Numba decorators that targets the CUDA GPU.

Execution Model

CUDA Python maps directly to the single-instruction multiple-thread execution (SIMT) model of CUDA. Each instruction is implicitly executed by multiple threads in parallel. With this execution model, array expressions are less useful because we don't want multiple threads to perform the same task. Instead, we want threads to perform a task in a cooperative fashion.


Numba for GPUs
--------------


Numba supports GPU programming by directly compiling a restricted subset of Python code 
into kernels and device functions following the execution model. 
Kernels written in Numba appear to have direct access to NumPy arrays. 
NumPy arrays are transferred between the CPU and the GPU automatically.
Numba supports GPUs from both Nvidia and AMD, but we will use Nvidia GPU as examples in the rest of the course. 

.. note:: newer GPU devices from NVIDIA support device-side kernel launching; this feature is called dynamic parallelism but Numba does not support it currently



Kernel declaration

A kernel function is a GPU function that is meant to be called from CPU code (*). It gives it two fundamental characteristics:

    kernels cannot explicitly return a value; all result data must be written to an array passed to the function (if computing a scalar, you will probably pass a one-element array);

    kernels explicitly declare their thread hierarchy when called: i.e. the number of thread blocks and the number of threads per block (note that while a kernel is compiled once, it can be called multiple times with different block sizes or grid sizes).





ufunc (gufunc) decorator
~~~~~~~~~~~~~~~~~~~~~~~~

Using ufuncs (and generalized ufuncs) is the easist way to run on a GPU with Numba, and it requires minimal understanding of GPU programming.
Numba @vectroize will produce a ufunc-like object. This object is a close analog but not fully compatible with a regular NumPy ufunc.
Generating a ufunc for GPU requires the explicit type signature and  target attribute.

.. typealong:: ufunc 

   .. tabs::

      .. tab:: python

         .. literalinclude:: example/math_cpu.py
            :language: python

      .. tab:: cpu

         .. literalinclude:: example/math_numba_cpu.py
            :language: python

      .. tab:: gpu

         .. literalinclude:: example/math_numba_gpu.py
            :language: python


benchmark

   .. tabs::

      .. tab:: cpu

	.. code-block:: python

		a = np.random.rand(10000000)
		b = np.random.rand(10000000)
		c = np.random.rand(10000000)

		%timeit c=func_numba_cpu(a, b)

      .. tab:: gpu

	.. code-block:: python

		a = np.random.rand(10000000)
		b = np.random.rand(10000000)
		c = np.random.rand(10000000)

		%timeit c=func_numba_gpu(a, b)



numba.vectorize() is limited to scalar arguments in the core function, for multi-dimensional arrays arguments, GUVectorize is used.

.. typealong::  

   .. tabs::

      .. tab:: python

         .. literalinclude:: example/matmul_cpu.py
            :language: python

      .. tab:: cpu

         .. literalinclude:: example/matmul_numba_cpu.py
            :language: python

      .. tab:: gpu

         .. literalinclude:: example/matmul_numba_gpu.py
            :language: python


benchmark

   .. tabs::

      .. tab:: numpy

	.. code-block:: python

		N = 500
		A = np.random.rand(N,N)
		B = np.random.rand(N,N)
		C = np.random.rand(N,N)
		%timeit np.matmul(A,B)

      .. tab:: cpu

	.. code-block:: python

		N = 500
		A = np.random.rand(N,N)
		B = np.random.rand(N,N)
		C = np.random.rand(N,N)
		%timeit matmul_numba_cpu(A,B,C)
		

      .. tab:: gpu

	.. code-block:: python

		N = 500
		A = np.random.rand(N,N)
		B = np.random.rand(N,N)
		C = np.random.rand(N,N)
		%timeit matmul_numba_gpu(A,B,C)



.. note:: Numba automatically did a lot of things for us:

  - Memory was allocated on GPU
  - Data was copied from CPU and GPU
  - The kernel was configured and launched
  - Data was copied back from GPU to CPU



CUDA kernel: CUDA JIT decorator 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alough it is simple to use ufuncs(gfuncs) to run on GPU, the performance is the price we have to pay. 
In addition, not all functions can be written as ufuncs in practice. To have much more flexibility, 
one needs to write a kernel on GPU or device function, which requires more understanding of the GPU programming. 



GPU Programming Model
~~~~~~~~~~~~~~~~~~~~~

Accelerators are a separate main circuit board with the processor, memory, power management, etc., but they can not operate by themselves. They are always part of a system (host) in which the CPUs run the operating systems and control the programs execution. This is reflected in the programming model. CPU (host) and GPU (device) codes are mixed. CPU acts as a main processor, controlling the execution workflow.  The host makes all calls, allocates the memory,  and  handles the memory transfers between CPU and GPU. GPUs run tens of thousands of threads simultaneously on thousands of cores and does not do much of the data management. The device code is executed by doing calls to functions (kernels) written specifically to take advantage of the GPU. The kernel calls are asynchronous, the control is returned to the host after a kernel calls. All kernels are executed sequentially. 

GPU Autopsy. Volta GPU
^^^^^^^^^^^^^^^^^^^^^^

.. figure:: img/volta-architecture.png
    :align: center

    A scheme of NVIDIA Volta GPU.

The NVIDIA GPU  architecture is built upon a number of multithreaded Streaming Multiprocessors (SMs), 
each SM contains a number of compute units. NVIDIA Volta GPU has 80 SMs.

NVIDIA Volta streaming multiprocessor (SM):

- 64 single precision cores

- 32 double precision cores

- 64 integer cores

- 8 Tensore cores

- 128 KB memory block for L1 and shared memory

  - 0 - 96 KB can be set to user managed shared memory

  - The rest is L1

- 65536 registers - enables the GPU to run a very large number of threads

.. figure:: img/volta-sm-architecture.png
    :align: center

    A scheme of NVIDIA Volta streaming multiprocessor.


thread hierarchy
^^^^^^^^^^^^^^^^

In order to take advantage of the accelerators it is needed to use parallelism.  
When a kernel is launched,  tens of thousands of threads are created. 
All threads execute the given kernel with each thread executing the same 
instructions but on different data (Single Iinstruction Multiple Data 
parallel programming model). It is therefore crucial  to know which thread 
operates on which array element(s). 

.. note:: All loops in which the individual iterations are independent of each other can be parallelized.




We just mentioned a little bit of the hardware.  To reflect this hierarchy on a software level, when CPU invokes a kernel grid, all the threads launched in the given kernel are partitioned/grouped into the so-called thread blocks, and the thread blocks of the grid are enumerated and distributed to SMs with available execution capacity. Thread blocks are required to execute independently, i.e. it must be possible to execute them in any order: in parallel or in series. Moreover, each thread block can be scheduled on any of the available SM within a GPU, in any order, concurrently or sequentially, so that they can be executed on any number of SMs.   However, a thread block can not be splitted among the SMs, but in a SM several blocks can be active at any given moment. As thread blocks terminate, new blocks are launched on the vacated SMs. Within a thread block, the threads execute concurrently on the same SM, and they can exchange data via the so called shared memory and can be explicitly synchronized.  The blocks can not interact with other blocks.

.. figure:: img/thread-hierarchy.png
   :align: center


Threads can be identified using a one-dimensional, two-dimensional, 
or three-dimensional thread index through the buit-in threadIdx variable,  
and this provides a natural way to invoke computation across the elements 
in a domain such as a vector, matrix, or volume.  Each block within the grid 
can be identified by  a one-dimensional, two-dimensional, or three-dimensional 
unique index accessible within the kernel through the built-in blockIdx variable. 
The dimension of the thread block is accessible within the kernel 
through the built-in blockDim variable.  The global index of a thread should be 
computed from its in-block index, the index of execution block and the block size. 
For 1D, it is threadIdx.x + blockIdx.x*blockDim.x.

.. note: Compared to an one-dimensional declarations of equivalent sizes, using multi-dimensional blocks does not change anything to the efficiency or behaviour of generated code, but can help you write your code in a more natural way.



.. figure:: img/MappingBlocksToSMs.png
   :align: center

A simple example of the division of threads (green squares) in blocks (cyan rectangles). The equally-sized blocks contain four threads each. The thread index starts from zero in each block. Hence the “global” thread index should be computed from the thread index, block index and block size. This is explained for the thread #3 in block #2 (blue numbers). The thread blocks are mapped to SMs for execution, with all threads within a block executing on the same device. The number of threads within one block does not have to be equal to the number of execution units within multiprocessor. In fact, GPUs can switch between software threads very efficiently, putting threads that currently wait for the data on hold and releasing the resources for threads that are ready for computations. For efficient GPU utilization, the number of threads per block has to be couple of factors higher than the number of computing units on the multiprocessor. Same is true for the number of thread blocks, which can and should be higher than the number of available multiprocessor in order to use the GPU computational resources efficiently.  XXX less text




To obtain the best choice of the thread grid is not a simple task, since it depends on the specific implemented algorithm and GPU computing capability. 
The total number of threads is equal to the number of threads per block times the number of blocks per grid.
The number of thread blocks per grid is usually dictated by the size of the data being processed, and it should be large enough to fully utilize the GPU.

  - start with 20-100 blocks, the number of blocks is usually chosen to be 2x-4x the number of SMs
 
The size of the number of threads per block should be a multiple of 32, values like 128, 256 or 512 are frequently used
  
  - it should be lower than 1024 since it determines how many threads share a limited size of the shared memory 

  - it must be large than the number of available (single precision, double precision or integer operation) cores in a SM to fully occupy the SM

The CUDA kernel launch overhead does depend on the number of blocks, so we find it best not to launch a grid where the number of threads equals the number of input elements when the input size is very big. We'll show a pattern for dealing with large inputs below.   XXX reformulate it


Because of the design, a GPU with more SMs will automatically execute the program in less time than a GPU with fewer SMs. 



 



Memory management
~~~~~~~~~~~~~~~~~

With many cores trying to access the memory simultaneously and with little cache available, 
the accelerator can run out of memory very quickly. This makes the memory management an essential task on the GPU.

Data transfer
^^^^^^^^^^^^^

Although Numba could transfer data automatically from/to the device, these data transfers are slow, 
sometimes even more than the actual on-device computation. 
Therefore explicitly transfering the data is necessary and should be minimised in real applications.

Using numba.cuda functions, one can transfer data from/to device. To transfer data from cpu to gpu, 
one could use to_device() method: 

.. code-block:: py

	d_x = numba.cuda.to_device(x)
	d_y = numba.cuda.to_device(y)

the resulting d_x is a DeviceNDArray. 
To transfer data on the device back to the host, one can use the copy_to_host() method:

.. code-block:: py

	h_x = numba.cuda.copy_to_host(d_x)
	h_y = numba.cuda.copy_to_host(d_y)


Memory hierarchy
^^^^^^^^^^^^^^^^

.. figure:: img/memory-hierarchy.png
   :align: center

As shown in the figure,  CUDA threads may access data from different memory spaces 
during kernel execution: 

  - local memory: Each thread has private local memory.
  - shared memory: Each thread block has shared memory visible to all threads of the thread block and with the same lifetime as the block.
  - global memory: All threads have access to the same global memory. 
  
Both local and global memory resides in device memory, so memory accesses have high latency and low bandwidth, i.e. slow access time.
On the other hand, shared memory has much higher bandwidth and much lower latency than local or global memory.
However, only a limited amount of shared memory can be allocated on the device for better performance. One can think it as a manually-managed data cache.


CUDA JIT decorator 
~~~~~~~~~~~~~~~~~~

Kernel and device functions are created with the numba.cuda.jit decorator on Nvidia GPUs.
Numba provides function i.e. numba.cuda.grid(ndim),  to calculate the global thread positions.



.. typealong:: CUDA kernel

   .. tabs::


      .. tab:: numba gpu

         .. literalinclude:: example/math_numba_gpu.py
            :language: python

      .. tab:: CUDA kernel

         .. literalinclude:: example/math_kernel.py
            :language: python

      .. tab:: CUDA kernel with device function

         .. literalinclude:: example/math_kernel_devicefunction.py
            :language: python


.. typealong:: benchmark

   .. tabs::

      .. tab:: CUDA kernel

	.. code-block:: python

		a = np.random.rand(10000000)
		b = np.random.rand(10000000)
		c = np.random.rand(10000000)
	        threadsperblock = 32
		blockspergrid = (100 + 31) // 32 # blockspergrid = (array.size + (threadsperblock - 1)) // threadsperblock
		%timeit math_kernel[threadsperblock, blockspergrid](a, b, result)

      .. tab:: CUDA kernel with device function

	.. code-block:: python

		a = np.random.rand(10000000)
		b = np.random.rand(10000000)
		c = np.random.rand(10000000)
	        threadsperblock = 32
		blockspergrid = (100 + 31) // 32 # blockspergrid = (array.size + (threadsperblock - 1)) // threadsperblock
		%timeit math_kernel_devicefunction[threadsperblock, blockspergrid](a, b, result)



.. typealong:: gufunc to kernel

   .. tabs::

      .. tab:: numba gpu

         .. literalinclude:: example/matmul_numba_gpu.py
            :language: python

      .. tab:: CUDA kernel

         .. literalinclude:: example/matmul_kernel.py
            :language: python

	test benchmark


benchmark

   .. tabs::

      .. tab:: numpy


	.. code-block:: ipython

		N = 500
		A = np.random.rand(N,N)
		B = np.random.rand(N,N)
		C = np.random.rand(N,N)
		%timeit np.matmul(A,B)


      .. tab:: gufunc

         .. literalinclude:: example/matmul_gu_benchmark.py
            :language: ipython

      .. tab:: CUDA kernel

         .. literalinclude:: example/matmul_kernel_benchmark.py
            :language: ipython



new benchmark

.. code-block:: python

	N = 500
	A = np.random.rand(N,N)
	B = np.random.rand(N,N)

	TPB = 16
	threadsperblock = (TPB, TPB)
	blockspergrid_x = int(math.ceil(C.shape[0] / threadsperblock[0]))
	blockspergrid_y = int(math.ceil(C.shape[1] / threadsperblock[1]))
	blockspergrid = (blockspergrid_x, blockspergrid_y)


	%timeit C = np.dot(A, B)
	%timeit matmul_gu(A, B, C)
	%timeit matmul_kernel[blockspergrid, threadsperblock](A, B, C)


There are times when the gufunc kernel uses too many of a GPU’s resources, which can cause the kernel launch to fail. The user can explicitly control the maximum size of the thread block by setting the max_blocksize attribute on the compiled gufunc object.



Optimization
------------

GPU can be easily misused and which leads to a low performance. One should condiser the following points when programming with GPU:

  - Maximize GPU utilization 
	- input data size to keep GPU busy
        - high arithmetic intensity
  - Maximize memory throughput
	- minimizing data transfers between the host and the device
	- minimizing data transfers between global memory and the device by using shared memory and cache
  - Maximize instruction throughput
	- Asynchronous execution
	- data types: 64bit data types (integer and floating point) have a significant cost when running on GPU compared to 32bit.



.. exercise:: Discrete Laplace Operator

In this exercise, we will work with the discrete Laplace operator.
It has a wide applications including numerical analysis, physics problems, image processing and machine learning as well.
Here we consider a simple two-dimensional implementation with finite-difference formula i.e. the five-point stencil, which reads:

.. math::
   u_{out}(i,j) = 0.25*[ u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) ]
               

where :math:`u(i,j)` refers to the input at location with
integer index :math:`i` and :math:`j` within the domain.


You will start with a naive implenmentation in python and we would like you to 
optimize it to run on both CPU and GPU using what we learned so far.


.. challenge:: lap2d

   .. tabs::

      .. tab:: python

	.. literalinclude:: exercise/lap2d.py
            :language: python

      .. tab:: benchmark

	.. literalinclude:: exercise/lap2d_benchmark.py
            :language: python


.. solution::  

   Optimization on CPU 

   .. tabs::

      .. tab:: numpy

	.. literalinclude:: exercise/lap2d_numpy.py
            :language: python

      .. tab:: numba gufunc

         .. literalinclude:: exercise/lap2d_numba_gu_cpu.py
            :language: python

      .. tab:: numba JIT

         .. literalinclude:: exercise/lap2d_numba_jit_cpu.py
            :language: python


   Optimization on GPU 

   .. tabs:: 
   
      .. tab:: numba gufunc

         .. literalinclude:: exercise/lap2d_numba_gu_gpu.py
            :language: python

      .. tab:: numba CUDA kernel

         .. literalinclude:: exercise/lap2d_cuda.py
            :language: python

cuPy
------





move data from the CPU to the GPU using the cp.asarray() function:

ary_cpu = np.arange(10)
ary_gpu = cp.asarray(ary_cpu)
print('cpu:', ary_cpu)
print('gpu:', ary_gpu)
print(ary_gpu.device)



Most of the NumPy methods are supported in CuPy with identical function names and arguments:



.. keypoints::

   - 1
   - 2
   - 3





The index of a thread and its "global" thread ID relate to each other in a straightforward way: For a one-dimensional block, they are the same; for a two-dimensional block of size (Dx, Dy),the thread ID of a thread of index (x, y) is (x + y Dx); for a three-dimensional block of size (Dx, Dy, Dz), the thread ID of a thread of index (x, y, z) is (x + y Dx + z Dx Dy). 


There is a limit to the number of threads per block, since all threads of a block 
are expected to reside on the same processor core and must share the limited memory resources of that core. 
On current GPUs, a thread block may contain up to 1024 threads.

GPUs like to be overloaded with threads, because they can switch among threads very quickly. 
This allows to hide the memory operations: while some threads wait, others can compute. 



Unless you are sure the block size and grid size is a divisor of your array size, you must check boundaries as shown above.
