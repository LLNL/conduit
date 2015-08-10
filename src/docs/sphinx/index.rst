.. ############################################################################
.. # Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
.. # 
.. # Produced at the Lawrence Livermore National Laboratory
.. # 
.. # LLNL-CODE-666778
.. # 
.. # All rights reserved.
.. # 
.. # This file is part of Conduit. 
.. # 
.. # For details, see: http://cyrush.github.io/conduit.
.. # 
.. # Please also read conduit/LICENSE
.. # 
.. # Redistribution and use in source and binary forms, with or without 
.. # modification, are permitted provided that the following conditions are met:
.. # 
.. # * Redistributions of source code must retain the above copyright notice, 
.. #   this list of conditions and the disclaimer below.
.. # 
.. # * Redistributions in binary form must reproduce the above copyright notice,
.. #   this list of conditions and the disclaimer (as noted below) in the
.. #   documentation and/or other materials provided with the distribution.
.. # 
.. # * Neither the name of the LLNS/LLNL nor the names of its contributors may
.. #   be used to endorse or promote products derived from this software without
.. #   specific prior written permission.
.. # 
.. # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
.. # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
.. # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
.. # ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
.. # LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
.. # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
.. # DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
.. # OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
.. # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
.. # STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
.. # IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
.. # POSSIBILITY OF SUCH DAMAGE.
.. # 
.. ############################################################################

.. Conduit documentation master file, created by
   sphinx-quickstart on Thu Oct 16 11:23:46 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**Conduit: A Scientific Data Exchange Library for HPC Simulations**


Introduction
============

What is Conduit?
----------------


Conduit is an open source project from Lawrence Livermore National Laboratory. It provides an intuitive model for describing hierarchical scientific data in C++, C, Fortran, and Python and is used for data coupling between packages in-core, serialization, and I/O tasks.


Conduit's Core API provides:

- A flexible way to describe hierarchal data:
  
    A JSON-based schema for describing hierarchical in-core data structures.

- A sane API to access hierarchal data:

    A dynamic API for rapid construction and consumption of hierarchical objects.

Conduit is under active development. The C++ API underpins the other language APIs and currently has the most features. We are still filling out the C, Fortran, and Python APIs. 

For more background, please see our :download:`Conduit Introduction <2015_conduit_intro.pdf>` presentation slides.


Unique Features
----------------

Conduit was built around the concept that an intuitive in-core data description capability simplifies many other common tasks in the HPC simulation eco-system. To this aim, Conduit's Core API:

- Provides a runtime focused in-core data description API that does not require repacking or code generation.
- Supports a mix of externally owned and Conduit allocated memory semantics.


Conduit Libraries 
------------------

Core API
~~~~~~~~~~

**conduit**
 - Provides Conduit's Core API in C++ and subsets of Core API in Python, C, and Fortran.
 - *Optionally depends on Python, NumPy, and Fortran*

Additional Libraries 
~~~~~~~~~~~~~~~~~~~~~
These libraries provide higher-level services built on top of Conduit's Core API.

*Caveat Emptor*: These libraries are young and their APIs are in flux.

**conduit_io** 
 - Provides I/O functionally beyond binary file and memory mapped I/O and json-based text I/O.
 - Includes a light-weight web server for REST and WebSocket clients. 
 - *Optionally depends on silo, hdf5, szip*

**conduit_mpi** 
 - Provides interfaces for MPI communication using conduit::Node instances as payloads. 
 - *Depends on mpi*

**blueprint**
 - Provides interfaces for common higher-level conventions and data protocols (eg. describing a “mesh”) using Conduit.


Contributors 
----------------
- Cyrus Harrison (LLNL)
- Brian Ryujin (LLNL)
- Adam Kunen (LLNL)
- Kathleen Biagas (LLNL)
- Eric Brugger (LLNL)
- George Aspesi (Harvey Mudd)
- Justin Bai (Harvey Mudd)
- Rupert Deese (Harvey Mudd)
- Linnea Shin (Harvey Mudd) 

 
In 2014 and 2015 LLNL sponsored a Harvey Mudd Computer Science Clinic project focused on using Conduit in HPC Proxy apps. You can read about more details about the clinic project from this LLNL article: 
http://computation.llnl.gov/newsroom/hpc-partnership-harvey-mudd-college-and-livermore



Conduit Documentation
----------------------

.. toctree::
   :maxdepth: 2

   user
   developer

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
