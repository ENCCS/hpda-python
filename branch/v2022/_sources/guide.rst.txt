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
- Understand the importance of measuring performance and profiling code before optimising
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
| 9:00 - 9:15       | :doc:`motivation`                  | 
+-------------------+------------------------------------+
| 9:15 - 9:50       | :doc:`scientific-data`             |
+-------------------+------------------------------------+
| 9:50 - 10:00      | Break                              |
+-------------------+------------------------------------+
| 10:00 - 10:50     | :doc:`stack`                       | 
+-------------------+------------------------------------+
| 10:50 - 11:00     | Break                              |
+-------------------+------------------------------------+
| 11:00 - 12:20     | :doc:`parallel-computing`          | 
+-------------------+------------------------------------+


**Day 2:**

+-------------------+------------------------------------+
| Time              | Episode                            |
+===================+====================================+
| 09:00 - 10:20     | :doc:`performance`                 |
+-------------------+------------------------------------+
| 10:20 - 10:40     | Break                              |
+-------------------+------------------------------------+
| 10:40 - 12:00     | :doc:`dask`                        |
+-------------------+------------------------------------+

**Day 3:**

+-------------------+------------------------------------+
| Time              | Episode                            |
+===================+====================================+
| 09:00 - 10:00     | :doc:`GPU-computing` I             | 
+-------------------+------------------------------------+
| 10:00 - 10:10     | Break                              |
+-------------------+------------------------------------+
| 10:10 - 11:00     | :doc:`GPU-computing` II            | 
+-------------------+------------------------------------+
| 11:00 - 11:10     | Break                              |
+-------------------+------------------------------------+
| 11:10 - 12:00     | Free exercise time                 |
+-------------------+------------------------------------+







