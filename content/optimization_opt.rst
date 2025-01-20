More on benchmarking, profiling and optimizing
==============================================

Profiling
---------

.. _scalene:
Scalene
~~~~~~~

Scalene is a sampling profiler. In addition to timings , it can also give insight into:

- CPU time spent in Python (interpreted), native (compiled) and system function calls
- Memory usage and copy
- GPU utilization
- Memory leak detection

Moreover, it adds minimal overhead due to profiling. The downside is the results are
less reproducible, because it is a sampling profiler.

Scalene can be used as a :ref:`scalene-cli`, or using :ref:`scalene-ipy` or in
:ref:`scalene-web` as an interactive widget. Here are some examples profiling
:download:`walk.py <example/walk.py>` with Scalene. 

.. _scalene-cli:
CLI tool
^^^^^^^^

.. code-block:: console

   $ scalene --cli walk.py

.. _scalene-ipy:
IPython magic
^^^^^^^^^^^^^
This allows for profiling a specific function. For example to profile just `walk`, we do as follows:

.. code-block:: ipython

   In [1]: %load_ext scalene

   In [2]: %run walk.py

   In [3]: %scrun --cli walk(n)

Gives the following output::
   
    SCRUN MAGIC
                                  /home/ashwinmo/Sources/enccs/hpda-python/content/example/walk.py: % of time = 100.00% (1.933s) out of 1.933s.                               
           ╷       ╷       ╷       ╷       ╷                                                                                                                                  
           │Time   │–––––– │–––––– │–––––– │                                                                                                                                  
      Line │Python │native │system │GPU    │/home/ashwinmo/Sources/enccs/hpda-python/content/example/walk.py                                                                  
    ╺━━━━━━┿━━━━━━━┿━━━━━━━┿━━━━━━━┿━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸
         1 │       │       │       │       │"""A 1-D random walk.                                                                                                             
         2 │       │       │       │       │                                                                                                                                  
         3 │       │       │       │       │See also:                                                                                                                         
         4 │       │       │       │       │- https://lectures.scientific-python.org/intro/numpy/auto_examples/plot_randomwalk.html                                           
         5 │       │       │       │       │                                                                                                                                  
         6 │       │       │       │       │"""                                                                                                                               
         7 │       │       │       │       │import numpy as np                                                                                                                
         8 │       │       │       │       │                                                                                                                                  
         9 │       │       │       │       │                                                                                                                                  
        10 │    6% │       │       │       │def step():                                                                                                                       
        11 │       │       │       │       │    import random                                                                                                                 
        12 │    7% │   64% │  13%  │       │    return 1.0 if random.random() > 0.5 else -1.0                                                                                 
        13 │       │       │       │       │                                                                                                                                  
        14 │       │       │       │       │                                                                                                                                  
        15 │       │       │       │       │def walk(n: int, dx: float = 1.0):                                                                                                
        16 │       │       │       │       │    """The for-loop version.                                                                                                      
        17 │       │       │       │       │                                                                                                                                  
        18 │       │       │       │       │    Parameters                                                                                                                    
        19 │       │       │       │       │    ----------                                                                                                                    
        20 │       │       │       │       │    n: int                                                                                                                        
        21 │       │       │       │       │        Number of time steps                                                                                                      
        22 │       │       │       │       │                                                                                                                                  
        23 │       │       │       │       │    dx: float                                                                                                                     
        24 │       │       │       │       │        Step size. Default step size is unity.                                                                                    
        25 │       │       │       │       │                                                                                                                                  
        26 │       │       │       │       │    """                                                                                                                           
        27 │       │       │       │       │    xs = np.zeros(n)                                                                                                              
        28 │       │       │       │       │                                                                                                                                  
        29 │       │       │       │       │    for i in range(n - 1):                                                                                                        
        30 │       │       │       │       │        x_new = xs[i] + dx * step()                                                                                               
        31 │    7% │       │       │       │        xs[i + 1] = x_new                                                                                                         
        32 │       │       │       │       │                                                                                                                                  
        33 │       │       │       │       │    return xs                                                                                                                     
        34 │       │       │       │       │                                                                                                                                  
        35 │       │       │       │       │                                                                                                                                  
        36 │       │       │       │       │def walk_vec(n: int, dx: float = 1.0):                                                                                            
        37 │       │       │       │       │    """The vectorized version of :func:`walk` using numpy functions."""                                                           
        38 │       │       │       │       │    import random                                                                                                                 
        39 │       │       │       │       │    steps = np.array(random.sample([1, -1], k=n, counts=[10 * n, 10 * n]))                                                        
        40 │       │       │       │       │                                                                                                                                  
        41 │       │       │       │       │    # steps = np.random.choice([1, -1], size=n)                                                                                   
        42 │       │       │       │       │                                                                                                                                  
        43 │       │       │       │       │    dx_steps = dx * steps                                                                                                         
        44 │       │       │       │       │                                                                                                                                  
        45 │       │       │       │       │    # set initial condition to zero                                                                                               
        46 │       │       │       │       │    dx_steps[0] = 0                                                                                                               
        47 │       │       │       │       │    # use cumulative sum to replicate time evolution of position x                                                                
        48 │       │       │       │       │    xs = np.cumsum(dx_steps)                                                                                                      
        49 │       │       │       │       │                                                                                                                                  
        50 │       │       │       │       │    return xs                                                                                                                     
        51 │       │       │       │       │                                                                                                                                  
        52 │       │       │       │       │                                                                                                                                  
        53 │       │       │       │       │if __name__ == "__main__":                                                                                                        
        54 │       │       │       │       │    n = 1_000_000                                                                                                                 
        55 │       │       │       │       │    _ = walk(n)                                                                                                                   
        56 │       │       │       │       │    _ = walk_vec(n)                                                                                                               
        57 │       │       │       │       │                                                                                                                                  
           │       │       │       │       │                                                                                                                                  
    ╶──────┼───────┼───────┼───────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╴
           │       │       │       │       │function summary for /home/ashwinmo/Sources/enccs/hpda-python/content/example/walk.py                                             
        10 │   14% │   69% │   9%  │       │step                                                                                                                              
        15 │    7% │       │       │       │walk                                                                                                                              
           ╵       ╵       ╵       ╵       

If you run the magic command in Jupyter you can use `%scrun walk(n)` instead and it should an output similar to the :ref:`scalene-web` below.

.. _scalene-web:
Web interface
^^^^^^^^^^^^^

Running

.. code-block:: console

   $ scalene walk.py

gives

.. figure:: example/scalene_web.png