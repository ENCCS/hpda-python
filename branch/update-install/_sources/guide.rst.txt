Instructor's guide
------------------

Why teach this lesson
^^^^^^^^^^^^^^^^^^^^^

Python has traditionally not been used in high performance computing environments 
due to its inherent performance bottlenecks - i.e. being an interpreted language 
and having an in-built mechanism (the global interpreter lock) which prohibits many 
forms of parallelisation. At the same time, Python is enormously popular across scientific disciplines and 
application domains in both academia in industry, and there are by now a number of mature 
libraries for accelerating and parallelising Python code. Although better performance 
and parallel scalability will invariably be obtained by writing code in traditional HPC 
languages (C and Fortran), teaching researchers and engineers to write performant, parallel 
and GPU-ported code in a high-level language like Python can save substantial development time, 
make HPC resources more accessible to a wider range of researchers, and lead to better 
overall utilisation of available HPC computing power.

Intended learning outcomes
^^^^^^^^^^^^^^^^^^^^^^^^^^

By the end of a workshop covering this lesson, learners should:

- Have a good overview of available tools and libraries for improving performance in Python
- Know what libraries are available for efficiently storing, reading and writing large data
- Be comfortable working with NumPy arrays and Pandas dataframes 
- Be able to explain why Python code is often slow 
- Understand the concept of vectorisation 
- Understand the importance of measuring performance and profiling code before optimizing
- Be able to describe the difference between "embarrasing", shared-memory and distributed-memory parallelism
- Know the basics of parallel workflows, multiprocessing, multithreading and MPI
- Understand pre-compilation and know basic usage of Numba and Cython
- Have a mental model of how Dask achieves parallelism
- Remember key hardware differences between CPUs and GPUs
- Be able to create simple GPU kernels with Numba


Schedule for 3 half-day workshop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Day 1**

+-------------------+------------------------------------+
| Time              | Episode                            |
+===================+====================================+
| 09:00 - 09:20     | :doc:`motivation`                  | 
+-------------------+------------------------------------+
| 09:20 - 10:00     | :doc:`scientific-data`             |
+-------------------+------------------------------------+
| 10:00 - 10:20     | Break                              |
+-------------------+------------------------------------+
| 10:20 - 11:00     | :doc:`stack`                       | 
+-------------------+------------------------------------+
| 11:00 - 11:20     | Break                              |
+-------------------+------------------------------------+
| 11:20 - 12:00     | :doc:`stack`                       | 
+-------------------+------------------------------------+


**Day 2:**

+-------------------+------------------------------------+
| Time              | Episode                            |
+===================+====================================+
| 09:00 - 09:40     | :doc:`parallel-computing`          |
+-------------------+------------------------------------+
| 09:40 - 09:50     | Break                              |
+-------------------+------------------------------------+
| 09:50 - 10:20     | :doc:`parallel-computing`          |
+-------------------+------------------------------------+
| 10:20 - 10:40     | Break                              |
+-------------------+------------------------------------+
| 10:40 - 11:20     | :doc:`optimization`                |
+-------------------+------------------------------------+
| 11:20 - 11:30     | Break                              |
+-------------------+------------------------------------+
| 11:30 - 12:00     | :doc:`optimization`                |
+-------------------+------------------------------------+


**Day 3:**

+-------------------+------------------------------------+
| Time              | Episode                            |
+===================+====================================+
| 09:00 - 09:40     | :doc:`performance-boosting`        | 
+-------------------+------------------------------------+
| 09:40 - 09:50     | Break                              |
+-------------------+------------------------------------+
| 09:50 - 10:20     | :doc:`performance-boosting`        | 
+-------------------+------------------------------------+
| 10:20 - 10:40     | Break                              |
+-------------------+------------------------------------+
| 10:40 - 11:20     | :doc:`dask`                        |
+-------------------+------------------------------------+
| 11:20 - 11:30     | Break                              |
+-------------------+------------------------------------+
| 11:30 - 12:00     | :doc:`dask`                        |
+-------------------+------------------------------------+








