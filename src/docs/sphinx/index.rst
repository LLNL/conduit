.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

.. Conduit documentation main file, created by
   sphinx-quickstart on Thu Oct 16 11:23:46 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Conduit
============

**Conduit: Simplified Data Exchange for HPC Simulations**

Introduction
------------

Conduit is an open source project from Lawrence Livermore National Laboratory that provides an intuitive model for describing hierarchical scientific data in C++, C, Fortran, and Python. It is used for data coupling between packages in-core, serialization, and I/O tasks.

Conduit's Core API provides:

- A flexible way to describe hierarchal data:
  
    A JSON-inspired data model for describing hierarchical in-core scientific data.

- A sane API to access hierarchal data:

    A dynamic API for rapid construction and consumption of hierarchical objects.

Conduit is under active development and targets Linux, OSX, and Windows platforms. The C++ API underpins the other language APIs and currently has the most features. We are still filling out the C, Fortran, and Python APIs.

Describing and sharing computational simulation meshes are very important use cases of Conduit.
The ``Mesh Blueprint`` facilitates this. For more details, please see the :ref:`Mesh Blueprint Docs and Examples <mesh_blueprint>`.

For more background on Conduit, please see :doc:`presentations`.


Getting Started
----------------

To get started building and using Conduit, see the :doc:`Quick Start Guide <quick_start>`  and the Conduit Tutorials for  :doc:`C++ <tutorial_cpp>` and :doc:`Python <tutorial_python>`. For more details about building Conduit see the :doc:`Building documentation<building>`.


Unique Features
----------------

Conduit was built around the concept that an intuitive in-core data description capability simplifies many other common tasks in the HPC simulation eco-system. To this aim, Conduit's Core API:

- Provides a runtime focused in-core data description API that does not require repacking or code generation.
- Supports a mix of externally owned and Conduit allocated memory semantics.


Projects Using Conduit
------------------------
Conduit is used in `VisIt <https://wci.llnl.gov/simulation/computer-codes/visit/>`_, `ALPINE Ascent <https://github.com/Alpine-DAV/ascent>`_, `MFEM <http://mfem.org/>`_, 
and `Axom <https://github.com/LLNL/axom>`_.

Conduit Project Resources
--------------------------

**Online Documentation**

http://software.llnl.gov/conduit/

**Github Source Repo**

https://github.com/llnl/conduit

**Issue Tracker**

https://github.com/llnl/conduit/issues


Conduit Libraries 
------------------

The *conduit* library provides Conduit's core data API. The *relay* and *blueprint* libraries provide higher-level services built on top of the core API. 


conduit
~~~~~~~~
 - Provides Conduit's Core API in C++ and subsets of Core API in Python, C, and Fortran.
 - *Optionally depends on Fortran and Python with NumPy*

relay
~~~~~~
 - Provides:
 
   - I/O functionally beyond simple binary, memory mapped, and json-based text file I/O.
   - A light-weight web server for REST and WebSocket clients. 
   - Interfaces for MPI communication using conduit::Node instances as payloads.
 - *Optionally depends on silo, hdf5, szip, adios, and mpi*

blueprint
~~~~~~~~~~
 - Provides interfaces for common higher-level conventions and data exchange protocols (eg. describing a “mesh”) using Conduit.
 - *No optional dependencies* 

See the :doc:`user` for more details on these libraries. 

Contributors 
----------------
- Cyrus Harrison (LLNL)
- Brian Ryujin (LLNL)
- Adam Kunen (LLNL)
- Joe Ciurej (LLNL)
- Kathleen Biagas (LLNL)
- Eric Brugger (LLNL)
- Aaron Black (LLNL)
- George Zagaris (LLNL)
- Kenny Weiss (LLNL)
- Matt Larsen (LLNL)
- Markus Salasoo (LLNL)
- Rebecca Haluska (LLNL)
- Arlie Capps (LLNL)
- Mark Miller (LLNL)
- Todd Gamblin (LLNL)
- Kevin Huynh (LLNL)
- Brad Whitlock (Intelligent Light)
- Chris Laganella (Intelligent Light)
- George Aspesi (Harvey Mudd)
- Justin Bai (Harvey Mudd)
- Rupert Deese (Harvey Mudd)
- Linnea Shin (Harvey Mudd) 

 
In 2014 and 2015 LLNL sponsored a Harvey Mudd Computer Science Clinic project focused on using Conduit in HPC Proxy apps. You can read about more details about the clinic project from this LLNL article: 
http://computation.llnl.gov/newsroom/hpc-partnership-harvey-mudd-college-and-livermore


Conduit Documentation
----------------------

.. toctree::
   :maxdepth: 1
   :hidden:

   quick_start

.. toctree::
   :caption: User Documentation
   :maxdepth: 1
   :hidden:

   conduit
   relay
   blueprint
   building
   glossary

.. toctree::
   :caption: Developer Documentation
   :maxdepth: 2
   :hidden:

   developer_source_layout
   developer_build_system
   developer_git_workflow

.. toctree::
   :caption: Resources
   :maxdepth: 1
   :hidden:

   releases
   presentations
   licenses

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
