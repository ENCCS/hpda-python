High Performance Data Analytics in Python
=========================================

Scientists, engineers and professionals from many sectors are seeing an enormous 
growth in the size and number of datasets relevant to their domains. 
Professional titles have emerged to describe specialists working with data, 
such as data scientists and data engineers, but also other experts are finding 
it necessary to learn tools and techniques to work with big data. Typical tasks 
include preprocessing, analysing, modeling and visualising data.

Python is an industry-standard programming language for working with data on 
all levels of the data analytics pipeline. This is in large part because of the rich 
ecosystem of libraries ranging from generic numerical libraries to 
special-purpose and/or domain-specific packages, often supported by large developer 
communities and stable funding sources.

This lesson is meant to give an overview of working with research data in 
Python using general libraries for storing, processing, analysing and sharing data. 
The focus is on high performance. After covering tools for performant 
processing on single workstations the focus shifts to profiling and optimising, parallel 
and distributed computing and finally GPU computing.



.. prereq::

   - Basic experience with Python
   - Basic experience in working in a Linux-like terminal
   - Some prior experience in working with large or small datasets



.. csv-table::
   :widths: auto
   :delim: ;

   15 min ; :doc:`motivation`
   35 min ; :doc:`scientific-data`
   50 min ; :doc:`stack`
   80 min ; :doc:`parallel-computing`
   40 min ; :doc:`optimization`
   40 min ; :doc:`performance-boosting`
   80 min ; :doc:`dask`
   110 min ; :doc:`GPU-computing`


.. toctree::
   :maxdepth: 1
   :caption: Preparation

   setup

.. toctree::
   :maxdepth: 1
   :caption: The lesson

   motivation
   scientific-data
   stack
   parallel-computing
   optimization
   performance-boosting
   dask

   


.. toctree::
   :maxdepth: 1
   :caption: Optional material

   pandas-extra
   GPU-computing


.. toctree::
   :maxdepth: 1
   :caption: Reference

   guide



.. _learner-personas:

Who is the course for?
----------------------

This material is for all researchers and engineers who work with large or small 
datasets and who want to learn powerful tools and best practices for writing more 
performant, parallelised, robust and reproducible data analysis pipelines.




About the course
----------------

This lesson material is developed by the `EuroCC National Competence Center
Sweden (ENCCS) <https://enccs.se/>`_ and taught in ENCCS workshops. 
Each lesson episode has clearly defined objectives that will be addressed and 
includes multiple exercises along with solutions, and is therefore also useful for
self-learning. The lesson material is licensed under `CC-BY-4.0
<https://creativecommons.org/licenses/by/4.0/>`_ and can be reused in any form
(with appropriate credit) in other courses and workshops.
Instructors who wish to teach this lesson can refer to the :doc:`guide` for
practical advice.




See also
--------

Each lesson episode has a "See also" section at the end which lists 
recommended further learning material.



Credits
-------

The lesson file structure and browsing layout is inspired by and derived from
`work <https://github.com/coderefinery/sphinx-lesson>`_ by `CodeRefinery
<https://coderefinery.org/>`_ licensed under the `MIT license
<http://opensource.org/licenses/mit-license.html>`__. We have copied and adapted
most of their license text.

Several examples and formulations are inspired by other open source 
educational material, in particular:

- `Python for Scientific Computing <https://aaltoscicomp.github.io/python-for-scicomp/>`__
- `Data analysis workflows with R and Python <https://aaltoscicomp.github.io/data-analysis-workflows-course/>`__
- `Python in High Performance Computing <https://github.com/csc-training/hpc-python>`__
- `Code examples from High Performance Python <https://github.com/mynameisfiber/high_performance_python_2e>`__
- `HPC Carpentry's Introduction to High-Performance Computing in Python <http://www.hpc-carpentry.org/hpc-python/>`__
- `Python Data Science Handbook <https://jakevdp.github.io/PythonDataScienceHandbook/index.html/>`__
- `An Introduction to Earth and Environmental Data Science <https://earth-env-data-science.github.io/>`__
- `Parallel and GPU Programming in Python <https://github.com/vcodreanu/SURFsara-PTC-Python-Parallel-and-GPU-Programming/tree/master/gpu_programming/>`__
- `Python in HPC <https://git.ichec.ie/training/studentresources/hpc-python/march-2021/>`__
- `High-performance computing with Python <https://gitlab.jsc.fz-juelich.de/sdlbio-courses/hpc-python/>`__
- `Advanced Python for data science in biology <https://github.com/NBISweden/workshop-advanced-python/>`__
- `Introduction to Numba <https://nyu-cds.github.io/python-numba/>`__
- `Python for Data Analysis <https://github.com/wesm/pydata-book/>`__
- `GTC2017-numba <https://github.com/ContinuumIO/gtc2017-numba/>`__
- `IPython Cookbook <https://ipython-books.github.io/>`__
- `Scipy Lecture Notes <https://scipy-lectures.org/>`__
- `Machine Learning and Data Science Notebooks <https://sebastianraschka.com/notebooks/ml-notebooks/>`__
- `Elegant SciPy <https://github.com/elegant-scipy/notebooks/>`__
- `A Comprehensive Guide to NumPy Data Types <https://axil.github.io/a-comprehensive-guide-to-numpy-data-types.html/>`__

Instructional Material
^^^^^^^^^^^^^^^^^^^^^^

This instructional material is made available under the
`Creative Commons Attribution license (CC-BY-4.0) <https://creativecommons.org/licenses/by/4.0/>`_.
The following is a human-readable summary of (and not a substitute for) the
`full legal text of the CC-BY-4.0 license
<https://creativecommons.org/licenses/by/4.0/legalcode>`_.
You are free to:

- **share** - copy and redistribute the material in any medium or format
- **adapt** - remix, transform, and build upon the material for any purpose,
  even commercially.

The licensor cannot revoke these freedoms as long as you follow these license terms:

- **Attribution** - You must give appropriate credit (mentioning that your work
  is derived from work that is Copyright (c) HPDA-Python and individual contributors and, where practical, linking
  to `<https://enccs.se>`_), provide a `link to the license
  <https://creativecommons.org/licenses/by/4.0/>`_, and indicate if changes were
  made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.
- **No additional restrictions** - You may not apply legal terms or
  technological measures that legally restrict others from doing anything the
  license permits.

With the understanding that:

- You do not have to comply with the license for elements of the material in
  the public domain or where your use is permitted by an applicable exception
  or limitation.
- No warranties are given. The license may not give you all of the permissions
  necessary for your intended use. For example, other rights such as
  publicity, privacy, or moral rights may limit how you use the material.



Software
^^^^^^^^

Except where otherwise noted, the example programs and other software provided
with this repository are made available under the `OSI <http://opensource.org/>`_-approved
`MIT license <https://opensource.org/licenses/mit-license.html>`_.

