Setup
=====

This lesson requires Python3 and a number of Python libraries. 
If you already have a preferred way to manage Python versions and 
libraries, you can stick to that. If not, we recommend that you 
install Python3 and all libraries using 
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`__, 
a free minimal installer for the package, dependency and environment manager 
`conda <https://docs.conda.io/en/latest/index.html>`__.

Please follow the installation instructions on 
https://docs.conda.io/en/latest/miniconda.html to install Miniconda3.

Make sure that both Python and conda are correctly installed:

.. code-block:: bash

   python --version
   # should give something like Python 3.9.7
   conda --version
   # should give something like conda 4.10.2

With conda installed, create a new conda environment named `hpda` by 

.. code-block:: bash

   conda create -n hpda python   

Then copy the following into a new file `requirements.txt`:

.. code-block:: bash

   WRITEME

and install the required dependencies by running:

.. code-block:: bash

   pip install -r requirements.txt --user
