.. ############################################################################
.. # Copyright (c) Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
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

.. note:: Conduit uses `BLT <https://github.com/LLNL/blt>`__ as its core CMake build system. We leverage BLT as a git submodule, however github does not include submodule contents in its automatically created source tarballs. To avoid confusion, starting with v0.3.0 we provide our own source tarballs that include BLT. 

v0.5.1
-----------------
* `Source Tarball <https://github.com/LLNL/conduit/releases/download/v0.5.1/conduit-v0.5.1-src-with-blt.tar.gz>`__

Highlights
+++++++++++++

(Extracted from Conduit's :download:`Changelog <../../../CHANGELOG.md>`)

Added
~~~~~~~~~

* **General**

 * Added Node::parse() method, (C++, Python and Fortran) which supports common json and yaml parsing use cases without creating a generator instance.
 * Use FOLDER target property to group targets for Visual Studio
 * Added Node load(), and save() support to the C and Fortran APIs

Changed
~~~~~~~~~

* **General**

 * Node::load() and Node::save() now auto detect which protocol to use when protocol argument is an empty string
 * Changed Node::load() and Node::save() default protocol value to empty (default now is to auto detect)
 * Changed Python linking strategy to defer linking for our compiler modules
 * Conduit Error Exception message strings now print cleaner (avoiding nesting doll string escaping headaches)
 * Build system improvements to support conda-forge builds for Linux, macOS, and Windows

Fixed
~~~~~~~~~

* **General**

 * Fixed install paths for CMake exported target files to follow standard CMake find_package() search conventions. Also perserved duplicate files to support old import path structure for this release.
 * python: Fixed Node.set_external() to accept conduit nodes as well as numpy arrays
 * Fixed dll install locations for Windows


v0.5.0
-----------------
* `Source Tarball <https://github.com/LLNL/conduit/releases/download/v0.5.0/conduit-v0.5.0-src-with-blt.tar.gz>`__

Highlights
+++++++++++++

(Extracted from Conduit's :download:`Changelog <../../../CHANGELOG.md>`)

Added
~~~~~~~~~

* **General**

 *  Added support to parse YAML into Conduit Nodes and to create YAML from Conduit Nodes. Support closely follows the "json" protocol, making similar choices related to promoting YAML string leaves to concrete data types.
 * Added several more Conduit Node methods to the C and Fortran APIs. Additions are enumerated here:  https://github.com/LLNL/conduit/pull/426
 * Added Node set support for Python Tuples and Lists with numeric and string entires
 * Added Node set support for Numpy String Arrays. String Arrays become Conduit lists with child char8_str arrays


* **Blueprint**

 * Added support for a "zfparray" blueprint that holds ZFP compressed array data. 
 * Added the the "specsets" top-level section to the Blueprint schema, which can be used to represent multi-dimensional per-material quantities (most commonly per-material atomic composition fractions).
 * Added explicit topological data generation functions for points, lines, and faces
 * Added derived topology generation functions for element centroids, sides, and corners
 * Added the basic example function to the conduit.mesh.blueprint.examples module

* **Relay**

 * Added optional ZFP support to relay, that enables wrapping and unwraping zfp arrays into conduit Nodes. 
 * Extended relay HDF5 I/O support to read a wider range of HDF5 string representations including H5T_VARIABLE strings.

Changed
~~~~~~~~~

* **General**

 * Conduit's automatic build process (uberenv + spack) now defaults to using Python 3
 * Improved CMake export logic to make it easier to find and use Conduit install in a CMake-based build system. (See using-with-cmake example for new recipe)

* **Relay**

 * Added is_open() method to IOHandle in the C++ and Python interfaces
 * Added file name information to Relay HDF5 error messages


Fixed
~~~~~~~~~

* **General**

 * Fixed bug that caused memory access after free during Node destruction

* **Relay**

 * Fixed crash with mpi broadcast_using_schema() when receiving tasks pass a non empty Node.
 * Fixed a few Windows API export issues for relay io


v0.4.0
-----------------
* `Source Tarball <https://github.com/LLNL/conduit/releases/download/v0.4.0/conduit-v0.4.0-src-with-blt.tar.gz>`__

Highlights
+++++++++++++

(Extracted from Conduit's :download:`Changelog <../../../CHANGELOG.md>`)

Added
~~~~~~~~~~~~~~~

* **General**

 * Added Generic IO Handle class (relay::io::IOHandle) with C++ and Python APIs, tests, and docs.
 * Added ``rename_child`` method to Schema and Node 
 * Added generation and install of conduit_config.mk for using-with-make example
 * Added datatype helpers for long long and long double
 * Added error for empty path fetch
 * Added C functions for setting error, warning, info handlers. 
 * Added limited set of C bindings for DataType
 * Added C bindings for relay IO
 * Added several more functions to conduit node python interfaces

* **Blueprint**

 * Added implicit point topology docs and example
 * Added julia and spiral mesh bp examples
 * Added mesh topology transformations to blueprint
 * Added polygonal mesh support to mesh blueprint
 * Added verify method for mesh blueprint nestset

* **Relay**

 * Added ADIOS Support, enabling ADIOS read and write of Node objects.
 * Added a relay::mpi::io library that mirrors the API of relay::io, except  that all functions take an MPI communicator. The functions are implemented in parallel for the ADIOS protocol. For other protocols, they will behave the same as the serial functions in relay::io. For the ADIOS protocol, the save() and save_merged() functions operate collectively within a communicator to enable multiple MPI ranks to save data to a single file as separate "domains".
 * Added an add_time_step() function to that lets the caller append data collectively to an existing  ADIOS file
 * Added a function to query the number of time steps and the number of domains in a  ADIOS file.
 * Added versions of save and save_merged that take an options node. 
 * Added C API for new save, save_merged functions.
 * Added method to list an HDF5 group's child names
 * Added save and append methods to the HDF5 I/O interface
 * Added docs and examples for relay io


Changed
~~~~~~~~~~~~~

* **General**

 * Changed mapping of c types to bit-width style to be compatible with C++11 std bit-width types when C++11 is enabled
 * Several improvements to uberenv, our automated build process, and building directions
 * Upgraded the type system with more explicit signed support


* **Relay**

 * Improvements to the Silo mesh writer
 * Refactor to support both relay::io and relay::mpi::io namespaces. 
 * Refactor to add support for steps and domains to I/O interfaces
 * Changed to only use ``libver latest`` setting for for hdf5 1.8 to minimize compatibility issues 

Fixed
~~~~~~~~~~~~~

* **General**

 * Fixed bugs with std::vector gap methods
 * Fixed A few C function names in conduit_node.h 
 * Fixed bug in python that was requesting unsigned array for signed cases
 * Fixed issue with Node::diff failing for string data with offsets
 * Fixes for building on BlueOS with the xl compiler

* **Blueprint**

 * Fixed validity status for blueprint functions
 * Fixed improper error reporting for Blueprint references


* **Relay**

 * Relay I/O exceptions are now forwarded to python  
 * Fixed MPI send_with_schema bug when data was compact but not contiguous  
 * Switched to use MPI bit-width style data type enums in ``relay::mpi``

 
v0.3.1
-----------------

* `Source Tarball <https://github.com/LLNL/conduit/releases/download/v0.3.1/conduit-v0.3.1-src-with-blt.tar.gz>`__


Highlights
+++++++++++++

* **General**
 
 * Added new ``Node::diff`` and ``Node::diff_compatible`` methods
 * Updated uberenv to use a newer spack and removed several custom packages
 * C++ ``Node::set`` methods now take const pointers for data
 * Added Python version of basic tutorial
 * Expanded the Node Python Capsule API
 * Added Python API bug fixes
 * Fixed API exports for static libs on Windows

* **Blueprint**
 
 * Mesh Protocol
 
   * Removed unnecessary state member in the braid example
 
 * Added Multi-level Array Protocol (conduit::blueprint::mlarray)

* **Relay**
 
 * Added bug fixes for Relay HDF5 support on Windows
 

v0.3.0
-----------------

* `Source Tarball <https://github.com/LLNL/conduit/releases/download/v0.3.0/conduit-v0.3.0-src-with-blt.tar.gz>`__

Highlights
+++++++++++++

* **General**

 * Moved to use BLT (https://github.com/llnl/blt) as our core CMake-based build system
 * Bug fixes to support building on Visual Studio 2013
 * Bug fixes for ``conduit::Node`` in the List Role
 * Expose more of the Conduit API in Python
 * Use ints instead of bools in the Conduit C-APIs for wider compiler compatibility   
 * Fixed memory leaks in *conduit* and *conduit_relay*
 

* **Blueprint**

 * Mesh Protocol
 
   * Added support for multi-material fields via *matsets* (volume fractions and per-material values)
   * Added initial support for domain boundary info via *adjsets* for distributed-memory unstructured meshes  
  

* **Relay**

 * Major improvements *conduit_relay* I/O HDF5 support 
 
   * Add heuristics with knobs for controlling use of HDF5 compact datasets and compression support
   * Improved error checking and error messages 
   
 * Major improvements to *conduit_relay_mpi* support 
 
   * Add support for reductions and broadcast
   * Add support zero-copy pass to MPI for a wide set of calls
   * Harden notion of `known schema` vs `generic` MPI support
 

v0.2.1
-----------------

* `Source Tarball <https://github.com/LLNL/conduit/archive/v0.2.1.tar.gz>`__


Highlights
+++++++++++++

* **General**

 * Added fixes to support static builds on BGQ using xlc and gcc
 * Fixed missing install of fortran module files
 * Eliminated separate fortran libs by moving fortran symbols into their associated main libs
 * Changed ``Node::set_external`` to support const Node references
 * Refactored path and file systems utils functions for clarity.

* **Blueprint**

 * Fixed bug with verify of mesh/coords for rectilinear case
 * Added support to the blueprint python module for the mesh and mcarray protocol methods 
 * Added stand alone blueprint verify executable

* **Relay**

 * Updated the version of civetweb used to avoid dlopen issues with SSL for static builds


v0.2.0
-----------------

* `Source Tarball <https://github.com/LLNL/conduit/archive/v0.2.0.tar.gz>`__
    
Highlights 
+++++++++++++
* **General**

 * Changes to clarify concepts in the ``conduit::Node`` API
 * Added const access to ``conduit::Node`` children and a new ``NodeConstIterator``
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
 * Refactored the ``relay::WebServer`` and the Conduit Node Viewer application
 * Added entangle, a python script ssh tunneling solution



