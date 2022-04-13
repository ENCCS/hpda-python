.. _scientific-data:

Scientific data
===============

.. objectives::

   - Get an overview of different formats for scientific data
   - Understand performance pitfalls when working with big data
   - Learn how to work with the HDF5 and NetCDF formats
   - Discuss the pros and cons of open science
   - Learn how to mint a DOI for your project   

What is a data?
---------------

bit and byte
^^^^^^^^^^^^

The smallest building block of storage in the computer is a **bit**, 
which stores either a 0 or 1.
Normally a number of 8 bits are combined in a group to make a **byte**. 
One byte (8 bits) can represent/hold at most :math:`2^8` distint values.
Organising bytes in different ways could further represent 
different types of information, i.e. data.


Types of scientific data 
^^^^^^^^^^^^^^^^^^^^^^^^

Numerical Data
**************

Different numerial data types (integer and floating-point) can be encoded as bytes. 
The more bytes we use for each value, the more range or precision we get, 
however the more memory it takes. For example, integers stored with 1 byte (8 bits) 
have a range from [-128, 127], while with 2 bytes (16 bits), the ranges becomes  [-32768, 32767].
Integers are whole numbers and can be represented precisely given enough bytes. 
However, for floating-point numbers the decimal fractions simply 
can not be represented exactly as binary (base 2) fractions in most cases 
which is known as the representation error. Arithmetic operations will 
further propagate this error. That is why in scienctific computing, 
numerical algorithms have to be carefully chosen and 
floating-point numbers are usally allocated with 8 bytes  
to make sure the inaccuracy is under control and does not lead to unsteady solutions.

.. note:: In climate community, it is common practice to use single precision in some part of the model to achieve better performance at a small cost to the accuracy.

Text Data
*********

When it comes to text data, the simplest character encoding 
is ASCII (American Standard Code for Information Interchange) and was the most 
common character encodings until 2008 when UTF-8 took over.
The orignal ASCII uses only 7 bits for representing each character/letter and 
therefore encodes only 128 specified characters. Later  it became common 
to use an 8-bit byte to store each character in memory, providing an extended ASCII. 
As computer becomes more powerful and  there is need for including more characters 
from other languages like Chinese, Greek, Arabic, etc. UTF-8  becomes 
the most common encoding nowadays and it uses minimum one byte up to four bytes per character. 


Data and storage format
^^^^^^^^^^^^^^^^^^^^^^^

In real applications, the scientific data is more complex and usually contains both numerical and text data. 
Here we list a few of the data and file storage formats commonly used:

Tabular Data
************

A very common type of data is the so-called "tabular data". The data is structured 
typically into rows and columns. Each column usually have a name and a specific data type 
while each row is a distinct sample which provides data according to each column including missing value.
The simplest and most common way to save tablular data is via the so-called CSV (comma-separated values) file.

Grided Data
***********

Grided data is another very common type, and usually the numerical data is saved 
in a multi-dimentional rectangular grid. Most probably it is saved in one of the following formats:

- Hierarchical Data Format (HDF5) - Container for many arrays
- Network Common Data Form (NetCDF) - Container for many arrays which conform to the NetCDF data model
- Zarr - New cloud-optimized format for array storage

Metadata
********

Metadata consists of the information about the data. 
Different types of data may have different metadata conventions. 

In Earth and Environmental science, there are widespread robust practices around metdata. 
For NetCDF files, metadata can be embedded directly into the data files. 
The most common metadata convention is Climate and Forecast (CF) Conventions, commonly used with NetCDF data


When it comes to data storage, there are many types of data storage format used 
in scietific computing and data analysis. There isn't one data storage format that 
works in all cases, choose a file format that best suits you.


CSV (comma-separated values)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Key features

   - **Type:** Text format
   - **Packages needed:** numpy, pandas
   - **Space efficiency:** Bad
   - **Good for sharing/archival:** Yes
   - Tidy data:
       - Speed: Bad
       - Ease of use: Great
   - Array data:
       - Speed: Bad
       - Ease of use: Ok for one or two dimensional data. Bad for anything higher.
   - **Best use cases:** Sharing data. Small data. Data that needs to be human-readable. 

CSV is by far the most popular file format, as it is human-readable and easily shareable.
However, it is not the best format to use when you're working with big data.

.. important::

    When working with floating point numbers, you should be careful to save the data 
    with enough decimal places so that you won't lose precision.

1. you may lose data precision simply because you do not save the data with enough decimals
2. CSV writing routines in Pandas and numpy try to avoid problems such as these 
   by writing the floating point numbers with enough precision, but even they are not infallible.
3. Storage of these high-precision CSV files is usually very inefficient storage-wise.
4. Binary files, where floating point numbers are represented in their native binary format, do not suffer from such problems.

HDF5 (Hierarchical Data Format version 5)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Key features

   - **Type:** Binary format
   - **Packages needed:** pandas, PyTables, h5py
   - **Space efficiency:** Good for numeric data.
   - **Good for sharing/archival:** Yes, if datasets are named well.
   - Tidy data:
       - Speed: Ok
       - Ease of use: Good
   - Array data:
       - Speed: Great
       - Ease of use: Good
   - **Best use cases:** Working with big datasets in array data format.

HDF5 is a high performance storage format for storing large amounts of data in multiple datasets in a single file.
It is especially popular in fields where you need to store big multidimensional arrays such as physical sciences.


NetCDF4 (Network Common Data Form version 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important::

    A great NetCDF4 interface is provided by a `xarray-package <https://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html#read-write-netcdf-files>`__.
    
  
.. admonition:: Key features

   - **Type**: Binary format
   - **Packages needed:** pandas, netCDF4/h5netcdf, xarray
   - **Space efficiency:** Good for numeric data.
   - **Good for sharing/archival:** Yes.
   - Tidy data:
       - Speed: Ok
       - Ease of use: Good
   - Array data:
       - Speed: Good
       - Ease of use: Great
   - **Best use cases:** Working with big datasets in array data format. Especially useful if the dataset contains spatial or temporal dimensions. Archiving or sharing those datasets.

NetCDF4 is a data format that uses HDF5 as its file format, but it has standardized structure of datasets and metadata related to these datasets.
This makes it possible to be read from various different programs.

NetCDF4 is by far the most common format for storing large data from big simulations in physical sciences.


The advantage of NetCDF4 compared to HDF5 is that one can easily add other metadata e.g. spatial dimensions (``x``, ``y``, ``z``) or timestamps (``t``) that tell where the grid-points are situated.
As the format is standardized, many programs can use this metadata for visualization and further analysis.



Sharing data
------------


The Open Science movement encourages researchers
to share research output beyond the contents of a
published academic article (and possibly supplementary information).

.. figure:: img/Open_Science_Principles.png
   :scale: 80 %
   :align: center

Pros and cons of sharing data (`from Wikipedia <https://en.wikipedia.org/wiki/Open_science>`__)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In favor:

- Open access publication of research reports and data allows for rigorous peer-review
- Science is publicly funded so all results of the research should be publicly available
- Open Science will make science more reproducible and transparent
- Open Science has more impact
- Open Science will help answer uniquely complex questions

Against:

- Too much unsorted information overwhelms scientists
- Potential misuse
- The public will misunderstand science data
- Increasing the scale of science will make verification of any discovery more difficult
- Low-quality science


FAIR principles
^^^^^^^^^^^^^^^

.. figure:: img/8-fair-principles.jpg
   :scale: 15 %
   :align: center

(This image was created by `Scriberia <http://www.scriberia.co.uk>`__ for `The
Turing Way <https://the-turing-way.netlify.com>`__ community and is used under a
CC-BY licence. The image was obtained from 
https://zenodo.org/record/3332808)

"FAIR" is the current buzzword for data management. You may be asked
about it in, for example, making data management plans for grants:

- Findable
 
  - Will anyone else know that your data exists?
  - Solutions: put it in a standard repository, or at least a
    description of the data. Get a digital object identifier (DOI).

- Accessible

  - Once someone knows that the data exists, can they get it?
  - Usually solved by being in a repository, but for non-open data,
    may require more procedures.

- Interoperable

  - Is your data in a format that can be used by others, like csv
    instead of PDF?
  - Or better than csv. Example: `5-star open data <https://5stardata.info/en/>`__

- Reusable

  - Is there a license allowing others to re-use?

Even though this is usually referred to as "open data", it means
considering and making good decisions, even if non-open.

FAIR principles are usually discussed in the context of data,
but they apply also for research software.

Note that FAIR principles do not require data/software to be open.

.. discussion:: Discuss open science

   - Do you share any other research outputs besides published articles and possibly source code?
   - Discuss pros and cons of sharing research data.

 

Services for sharing and collaborating on research data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To find a research data repository for your data, you can search on the
`Registry of Research Data Repositories re3data <https://www.re3data.org/>`__
platform and filter by country, content type, discipline, etc.

**International:**

- `Zenodo <https://zenodo.org/>`__: A general-purpose open access repository
  created by OpenAIRE and CERN. Integration with GitHub, allows
  researchers to upload files up to 50 GB.
- `Figshare <https://figshare.com/>`__: Online digital repository where researchers
  can preserve and share their research outputs (figures, datasets, images and videos).
  Users can make all of their research outputs available in a citable,
  shareable and discoverable manner.
- `EUDAT <https://eudat.eu>`__: European platform for researchers and practitioners from any research discipline to preserve, find, access, and process data in a trusted environment.
- `Dryad <https://datadryad.org/>`__: A general-purpose home for a wide diversity of datatypes,
  governed by a nonprofit membership organization.
  A curated resource that makes the data underlying scientific publications discoverable,
  freely reusable, and citable.
- `The Open Science Framework <https://osf.io/>`__: Gives free accounts for collaboration
  around files and other research artifacts. Each account can have up to 5 GB of files
  without any problem, and it remains private until you make it public.

**Sweden:**

- `ICOS for climate data <http://www.icos-sweden.se/>`__
- `Bolin center climate / geodata <https://bolin.su.se/data/>`__
- `NBIS for life science, sequence –omics data <https://nbis.se/infrastructure>`__


.. exercise:: (Optional) Get a DOI by connecting your repository to Zenodo

   Digital object identifiers (DOI) are the backbone of the academic
   reference and metrics system. In this exercise we will see how to
   make a GitHub repository citable by archiving it on the
   [Zenodo](http://about.zenodo.org/) archiving service. Zenodo is a
   general-purpose open access repository created by OpenAIRE and CERN.
   
   1. Sign in to Zenodo using your GitHub account. For this exercise, use the
      sandbox service: https://sandbox.zenodo.org/login/. This is a test version of the real Zenodo platform.
   2. Go to https://sandbox.zenodo.org/account/settings/github/.
   3. Find the repository you wish to publish, and flip the switch to ON.
   4. Go to GitHub and create a **release**  by clicking the `Create a new release` on the 
      right-hand side (a release is based on a Git tag, but is a higher-level GitHub feature).
   5. Creating a new release will trigger Zenodo into archiving your repository,
      and a DOI badge will be displayed next to your repository after a minute
      or two. You can include it in your GitHub README file: click the
      DOI badge and copy the relevant format (Markdown, RST, HTML).


See also
--------

- `Five recommendations for fair software <https://fair-software.eu/>`__
- `The Turing way <https://github.com/alan-turing-institute/the-turing-way/>`__


.. keypoints::

   - 1
   - 2
   - Consider sharing other research outputs than articles.
