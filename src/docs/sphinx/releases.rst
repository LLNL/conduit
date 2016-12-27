.. ############################################################################
.. # Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # LLNL-CODE-666778
.. #
.. # All rights reserved.
.. #
.. # This file is part of Conduit.
.. #
.. # For details, see: http://software.llnl.gov/conduit/.
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

================================
Releases
================================

Source distributions for Conduit releases are hosted on github:

https://github.com/LLNL/conduit/releases

v0.2.1
-----------------

* `Source Tarball <https://github.com/LLNL/conduit/archive/v0.2.1.tar.gz>`_
* `Docs <http://software.llnl.gov/conduit/v0.2.1>`_


Highlights
+++++++++++++

* **General**

 * Added fixes to support static builds on BGQ using xlc and gcc
 * Fixed missing install of fortran module files
 * Eliminated separate fortran libs by moving fortran symbols into their associated main libs
 * Change Node set_external to support const Node ref
 * Refactor path and file systems utils functions for clarity.

* **Blueprint**

 * Fixed bug with verify of mesh/coords for rectilinear case
 * Added support to the blueprint python module for the mesh and mcarray protocol methods 
 * Added stand alone blueprint verify executable

* **Relay**
 * Updated the version of civetweb used to avoid dlopen issues with SSL for static builds


v0.2.0
-----------------

* `Source Tarball <https://github.com/LLNL/conduit/archive/v0.2.0.tar.gz>`_
* `Docs <http://software.llnl.gov/conduit/v0.2.0>`_
    
Highlights 
+++++++++++++
* **General**

 * Changes to clarify concepts in the conduit::Node API
 * Added const access to conduit::Node's children and a new NodeConstIterator
 * Added support for building on Windows
 * Added more Python, C, and Fortran API support
 * Resolved several bugs across libraries
 * Resolved compiler warnings and memory leaks
 * Improved unit test coverage
 * Renamed source and header files for clarity and to avoid potential conflicts with other projects

* **Blueprint**
    
 * Added verify support for the mcarray and mesh protocols
 * Added functions that create examples instances of mcarrays and meshes
 * Added memory layout transform helpers for mcarrays
 * Added a helper that creates a mesh blueprint index from a valid mesh

* **Relay**

 * Added extensive HDF5 I/O support for reading and writing between HDF5 files and conduit Node trees
 * Changed I/O protocol string names for clarity
 * Refactored the relay::WebServer and the Conduit Node Viewer application
 * Added entangle, a python script ssh tunneling solution



