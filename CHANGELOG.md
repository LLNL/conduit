# Conduit Changelog
Notable changes to Conduit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added Generic IO Handle class (relay::io::IOHandle) with C++ and Python APIs, tests, and docs.
- Added ``rename_child`` method to Schema and Node 
- Added generation and install of conduit_config.mk for using-with-make example
- Added datatype helpers for long long and long double
- Added error for empty path fetch
- Added C functions for setting error, warning, info handlers. 
- Added limited set of C bindings for DataType
- Added C bindings for relay IO
- Added several more functions to conduit node python interfaces

- Blueprint: Added implicit point topology docs and example
- Blueprint: Added julia and spiral mesh bp examples
- Blueprint: Added mesh topology transformations to blueprint
- Blueprint: Added polygonal mesh support to mesh blueprint
- Blueprint: Added verify method for mesh blueprint nestset

- Relay: Added ADIOS Support
- Relay: Added versions of save and save_merged that take an options node. 
- Relay: Added C API for new save, save_merged functions.
- Relay: Added method to list an HDF5 group's child names
- Relay: Added save and append methods to the HDF5 I/O interface
- Relay: Added docs and examples for relay io

### Changed
- Changed mapping of c types to bit-width style to be compatible with C++11 std bit-width types when C++11 is enabled
- Several improvements to uberenv, our automated build process, and building directions
- Upgraded the type system with more explicit signed support

- Relay: Improvements to the Silo mesh writer
- Relay: Refactor to support both relay::io and relay::mpi::io namespaces. 
- Relay: Refactor to add support for steps and domains to I/O interfaces
- Relay: Changed to only use ``libver latest`` setting for for hdf5 1.8 to minimize compatibility issues 

### Fixed
- Fixed bugs with std::vector gap methods
- Fixed A few C function names in conduit_node.h 
- Fixed bug in python that was requesting unsigned array for signed cases
- Fixed issue with Node::diff failing for string data with offsets
- Fixes for building on BlueOS with the xl compiler

- Blueprint: Fixed validity status for blueprint functions
- Blueprint: Fixed improper error reporting for Blueprint references

- Relay: Relay I/O exceptions are now forwarded to python  
- Relay: Fixed MPI send_with_schema bug when data was compact but not contiguous  
- Relay: Switched to use MPI bit-width style data type enums in ``relay::mpi``

## [0.3.1] - Released 2018-02-26

### Added
- Added new ``Node::diff`` and ``Node::diff_compatible`` methods
- Expanded the Node Python Capsule API
- Added Python version of basic tutorial

- Blueprint: Added Multi-level Array Protocol (conduit::blueprint::mlarray)


### Changed
- Updated uberenv to use a newer spack and removed several custom packages
- C++ ``Node::set`` methods now take const pointers for data

### Fixed 
- Various Python API bug fixes
- Fixed API exports for static libs on Windows

- Relay: Bug fixes for HDF5 support on Windows

### Removed
- Blueprint: Removed unnecessary state member in the braid example

## [0.3.0] - Released 2017-08-21

### Added 
- Exposed more of the Conduit API in Python

- Blueprint: Added support for multi-material fields via *matsets* (volume fractions and per-material values)
- Blueprint: Added initial support for domain boundary info via *adjsets* for distributed-memory unstructured meshes  

- Relay: Added support for MPI reductions and broadcast
- Relay: Added support zero-copy pass to MPI for a wide set of calls
- Relay: Hardened notion of `known schema` vs `generic` MPI support
- Relay: Add heuristics with knobs for controlling use of HDF5 compact datasets and compression support

### Changed
- Moved to use BLT (https://github.com/llnl/blt) as our core CMake-based build system
- Use ints instead of bools in the Conduit C-APIs for wider compiler compatibility

- Relay: Improved error checking and error messages for HDF5 I/O support

### Fixed
- Fixed memory leaks in *conduit* and *conduit_relay* 
- Bug fixes to support building on Visual Studio 2013
- Bug fixes for `conduit::Nodes` in the List Role
   

## [0.2.1] - Released 2017-01-06

### Added
- Blueprint: Added support to the blueprint python module for the mesh and mcarray protocol methods 
- Blueprint: Added stand alone blueprint verify executable

### Changed
- Eliminated separate fortran libs by moving fortran symbols into their associated main libs
- Change Node set_external to support const Node ref
- Refactor path and file systems utils functions for clarity

### Fixed
- Added fixes to support static builds on BGQ using xlc and gcc
- Fixed missing install of fortran module files

- Blueprint: Fixed bug with verify of mesh/coords for rectilinear case

- Relay: Updated the version of civetweb used to avoid dlopen issues with SSL for static builds

## [0.2.0] - Released 2016-11-03

### Added
- Added const access to conduit::Node's children and a new NodeConstIterator
- Added support for building on Windows 
- Added more Python, C, and Fortran API support

- Blueprint: Added verify support for the mcarray and mesh protocols
- Blueprint: Added functions that create examples instances of mcarrays and meshes
- Blueprint: Added memory layout transform helpers for mcarrays
- Blueprint: Added a helper that creates a mesh blueprint index from a valid mesh

- Relay: Added entangle, a python script ssh tunneling solution
- Relay: Added extensive HDF5 I/O support for reading and writing between HDF5 files and conduit Node trees

###  Changed
- Changes to clarify concepts in the conduit::Node API 
- Improved unit test coverage
- Renamed source and header files for clarity and to avoid potential conflicts with other projects 

- Relay: Changed I/O protocol string names for clarity 
- Relay: Refactored the relay::WebServer and the Conduit Node Viewer application 

### Fixed
- Resolved several bugs across libraries 
- Resolved compiler warnings and memory leaks

## 0.1.0 - Released 2016-03-30

### Added
- Initial Open Source Release on GitHub

[Unreleased]: https://github.com/llnl/conduit/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/llnl/conduit/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/llnl/conduit/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/llnl/conduit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/llnl/conduit/compare/v0.1.0...v0.2.0
