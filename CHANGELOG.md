# Conduit Changelog
Notable changes to Conduit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project aspires to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased 

### Added

#### General 
- Added support for children with names that include `/`. Since slashes are part of Conduit's hierarchical path mechanism, you must use explicit methods (add_child(), child(), etc) to create and access children with these types of names. These names are also supported in all basic i/o cases (JSON, YAML, Conduit Binary).
- Added Node::child and Schema::child methods, which provide access to existing children by name.
- Added Node::fetch_existing and Schema::fetch_existing methods, which provide access to existing paths or error when given a bad path.
- Added Node::add_child() and Node::remove_child() to support direct operations and cases where names have `/`s.
- Added a set of conduit::utils::log::remove_* filtering functions, which process conduit log/info nodes and strip out the requested information (useful for focusing the often verbose output in log/info nodes).
- Added to_string() and to_string_default() methods to Node, Schema, DataType, and DataArray. These methods alias either to_yaml() or to_json(). Long term yaml will be preferred over json.
- Added helper script (scripts/regen_docs_outputs.py) that regenerates all example outputs used Conduit's Sphinx docs.
- Added to_yaml() and to_yaml_stream methods() to Schema, DataType, and DataArray.
- Added support for C++-style iterators on node children. You can now do `for (Node &node : node.children()) {}`. You can also do `node.children.begin()` and `node.children.end()` to work with the iterators directly.

#### Relay
- Added an open mode option to Relay IOHandle. See Relay IOHandle docs (https://llnl-conduit.readthedocs.io/en/latest/relay_io.html#relay-i-o-handle-interface) for more details.
- Added the conduit.relay.mpi Python module to support Relay MPI in Python.
- Added support to write and read Conduit lists to HDF5 files. Since HDF5 Groups do not support unnamed indexed children, each list child is written using a string name that represents its index and a special attribute is written to the HDF5 group to mark the list case. On read, the special attribute is used to detect and read this style of group back into a Conduit list.
- Added preliminary support to read Sidre Datastore-style HDF5 using Relay IOHandle,  those grouped with a root file.
- Added `conduit::relay::io::blueprint::read_mesh` functions, were pulled in from Ascent's Blueprint import logic.

#### Blueprint
- Added support for sparse one-to-many relationships with the new `blueprint::o2mrelation` protocol. See the `blueprint::o2mrelation::examples::uniform` example for details.
- Added sparse one-to-many, uni-buffer, and material-dominant specification support to Material sets. See the Material sets documentation
(https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#material-sets) for more details.
- Added support for Adjacency sets for Structured Mesh Topologies. See the `blueprint::mesh::examples::adjset_uniform` example.
- Added `blueprint::mesh::examples::julia_nestsets_simple` and `blueprint::mesh::examples::julia_nestsets_complex` examples represent Julia set fractals using patch-based AMR meshes and the Mesh Blueprint Nesting Set protocol. See the Julia AMR Blueprint docs
(https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#julia-amr-examples) for more details.
- Added `blueprint::mesh::examples::venn` example that demonstrates different ways to encode volume fraction based multi-material fields.  See the Venn Blueprint docs
(https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#venn) for more details.
- Added `blueprint::mesh::number_of_domains` property method for trees that conform to the mesh blueprint.
- Added MPI mesh blueprint methods, `blueprint::mpi::mesh::verify` and  `blueprint::mpi::mesh::number_of_domains` (available in the `conduit_blueprint_mpi` library)
- Added `blueprint::mpi::mesh::examples::braid_uniform_multi_domain` and `blueprint::mpi::mesh::examples::spiral_round_robin` distributed-memory mesh examples to the `conduit_blueprint_mpi` library.
- Added `state/path` to the Mesh Blueprint index, needed for consumers to know the proper path to read extended state info (such as `domain_id`)


### Fixed

#### General 
- Updated to newer BLT to resolve BLT/FindMPI issues with rpath linking commands when using OpenMPI.
- Fixed internal object name string for the Python Iterator object. It used to report `Schema`, which triggered both puzzling and concerned emotions.
- Fixed a bug with `Node.set` in the Python API that undermined setting NumPy arrays with sliced views and complex striding. General slices should now work with `set`. No changes to the `set_external` case, which requires 1-D effective striding and throws an exception when more complex strides are presented.
  

#### Relay
- Use H5F_ACC_RDONLY in relay::io::is_hdf5_file to avoid errors when checking files that already have open HDF5 handles.
- Fixed compatibility check for empty Nodes against HDF5 files with existing paths

### Changed

#### General 
- Node::fetch_child and Schema::fetch_child are deprecated in favor of the more clearly named Node::fetch_existing and Schema::fetch_existing. fetch_child variants still exist, but will be removed in a future release.
- Python str() methods for Node, Schema, and DataType now use their new to_string() methods.
- DataArray<T>::to_json(std::ostring &) is deprecated in favor DataArray<T>::to_json_stream. to_json(std::ostring &) will be removed in a future release.
- Schema::to_json and Schema::save variants with detailed (bool) arg are deprecated. The detailed arg was never used. These methods will be removed in a future release.
- Node::print() now prints yaml instead of json.
- The string return variants of `about` methods now return yaml strings instead of json strings.
- Sphinx Docs code examples and outputs are now included using start-after and end-before style includes.
- Schema to_json() and to_json_stream() methods were expanded to support indent, depth, pad and end-of-element args.

#### Relay
- Provide more context when a Conduit Node cannot be written to a HDF5 file because it is incompatible with the existing HDF5 tree. Error messages now provide the full path and details about the incompatibility.
- `conduit::relay::io_blueprint::save` functions are deprecated in favor of `conduit::relay::io::blueprint::write_mesh`
- `conduit::relay::io::blueprint::write_mesh` functions were pulled in from Ascent's Blueprint export logic.
- `conduit_relay_io_mpi` lib now depends on `conduit_relay_io`. Due to this change, a single build supports either ADIOS serial (no-mpi) or ADIOS with MPI support, but not both. If conduit is configured with MPI support, ADIOS MPI is used.


#### Blueprint
- Refactored the Polygonal and Polyhedral mesh blueprint specification to leverage one-to-many concepts and to allow more zero-copy use cases.
- The `conduit_blueprint_mpi` library now depends on `conduit_relay_mpi`.
- The optional Mesh Blueprint structured topology logical element origin is now specified using `{i,j,k}` instead of `{i0,j0,k0}`.


## [0.5.1] - Released 2020-01-18

### Added

#### General 
- Added Node::parse() method, (C++, Python and Fortran) which supports common json and yaml parsing use cases without creating a generator instance.
- Use FOLDER target property to group targets for Visual Studio
- Added Node load(), and save() support to the C and Fortran APIs

### Changed

#### General 
- Node::load() and Node::save() now auto detect which protocol to use when protocol argument is an empty string
- Changed Node::load() and Node::save() default protocol value to empty (default now is to auto detect)
- Changed Python linking strategy to defer linking for our compiler modules
- Conduit Error Exception message strings now print cleaner (avoiding nesting doll string escaping headaches)
- Build system improvements to support conda-forge builds for Linux, macOS, and Windows

### Fixed

#### General 
- Fixed install paths for CMake exported target files to follow standard CMake find_package() search conventions. Also perserved duplicate files to support old import path structure for this release.
- python: Fixed Node.set_external() to accept conduit nodes as well as numpy arrays
- Fixed dll install locations for Windows


## [0.5.0] - Released 2019-10-25

### Added

#### General 
- Added support to parse YAML into Conduit Nodes and to create YAML from Conduit Nodes. Support closely follows the "json" protocol, making similar choices related to promoting YAML string leaves to concrete data types.
- Added several more Conduit Node methods to the C and Fortran APIs. Additions are enumerated here:  https://github.com/LLNL/conduit/pull/426
- Added Node set support for Python Tuples and Lists with numeric and string entires
- Added Node set support for Numpy String Arrays. String Arrays become Conduit lists with child char8_str arrays


#### Blueprint

- Added support for a "zfparray" blueprint that holds ZFP compressed array data. 
- Added the the "specsets" top-level section to the Blueprint schema, which can be used to represent multi-dimensional per-material quantities (most commonly per-material atomic composition fractions).
- Added explicit topological data generation functions for points, lines, and faces
- Added derived topology generation functions for element centroids, sides, and corners
- Added the basic example function to the conduit.mesh.blueprint.examples module

#### Relay
- Added optional ZFP support to relay, that enables wrapping and unwraping zfp arrays into conduit Nodes. 
- Extended relay HDF5 I/O support to read a wider range of HDF5 string representations including H5T_VARIABLE strings.

### Changed

#### General 
- Conduit's automatic build process (uberenv + spack) now defaults to using Python 3
- Improved CMake export logic to make it easier to find and use Conduit install in a CMake-based build system. (See using-with-cmake example for new recipe)

#### Relay

- Added is_open() method to IOHandle in the C++ and Python interfaces
- Added file name information to Relay HDF5 error messages


### Fixed

#### General 
- Fixed bug that caused memory access after free during Node destruction

#### Relay

- Fixed crash with mpi broadcast_using_schema() when receiving tasks pass a non empty Node.
- Fixed a few Windows API export issues for relay io

## [0.4.0] - Released 2019-03-01

### Added

#### General

- Added Generic IO Handle class (relay::io::IOHandle) with C++ and Python APIs, tests, and docs.
- Added ``rename_child`` method to Schema and Node 
- Added generation and install of conduit_config.mk for using-with-make example
- Added datatype helpers for long long and long double
- Added error for empty path fetch
- Added C functions for setting error, warning, info handlers. 
- Added limited set of C bindings for DataType
- Added C bindings for relay IO
- Added several more functions to conduit node python interfaces

#### Blueprint

- Added implicit point topology docs and example
- Added julia and spiral mesh bp examples
- Added mesh topology transformations to blueprint
- Added polygonal mesh support to mesh blueprint
- Added verify method for mesh blueprint nestset

#### Relay

- Added ADIOS Support, enabling ADIOS read and write of Node objects.
- Added a relay::mpi::io library that mirrors the API of relay::io, except  that all functions take an MPI communicator. The functions are implemented in parallel for the ADIOS protocol. For other protocols, they will behave the same as the serial functions in relay::io. For the ADIOS protocol, the save() and save_merged() functions operate collectively within a communicator to enable multiple MPI ranks to save data to a single file as separate "domains".
- Added an add_time_step() function to that lets the caller append data collectively to an existing  ADIOS file
- Added a function to query the number of time steps and the number of domains in a  ADIOS file.
- Added versions of save and save_merged that take an options node. 
- Added C API for new save, save_merged functions.
- Added method to list an HDF5 group's child names
- Added save and append methods to the HDF5 I/O interface
- Added docs and examples for relay io


### Changed

#### General 

- Changed mapping of c types to bit-width style to be compatible with C++11 std bit-width types when C++11 is enabled
- Several improvements to uberenv, our automated build process, and building directions
- Upgraded the type system with more explicit signed support

#### Relay

- Improvements to the Silo mesh writer
- Refactor to support both relay::io and relay::mpi::io namespaces. 
- Refactor to add support for steps and domains to I/O interfaces
- Changed to only use ``libver latest`` setting for for hdf5 1.8 to minimize compatibility issues 

### Fixed

#### General 

- Fixed bugs with std::vector gap methods
- Fixed A few C function names in conduit_node.h 
- Fixed bug in python that was requesting unsigned array for signed cases
- Fixed issue with Node::diff failing for string data with offsets
- Fixes for building on BlueOS with the xl compiler

#### Blueprint

- Fixed validity status for blueprint functions
- Fixed improper error reporting for Blueprint references

#### Relay

- Relay I/O exceptions are now forwarded to python  
- Fixed MPI send_with_schema bug when data was compact but not contiguous  
- Switched to use MPI bit-width style data type enums in ``relay::mpi``

## [0.3.1] - Released 2018-02-26

### Added

#### General

- Added new ``Node::diff`` and ``Node::diff_compatible`` methods
- Expanded the Node Python Capsule API
- Added Python version of basic tutorial

#### Blueprint

- Added Multi-level Array Protocol (conduit::blueprint::mlarray)

### Changed

#### General

- Updated uberenv to use a newer spack and removed several custom packages
- C++ ``Node::set`` methods now take const pointers for data

### Fixed 

#### General

- Various Python API bug fixes
- Fixed API exports for static libs on Windows

#### Relay

- Bug fixes for HDF5 support on Windows

### Removed

#### Blueprint

- Removed unnecessary state member in the braid example

## [0.3.0] - Released 2017-08-21

### Added 

#### General

- Exposed more of the Conduit API in Python

#### Blueprint

- Added support for multi-material fields via *matsets* (volume fractions and per-material values)
- Added initial support for domain boundary info via *adjsets* for distributed-memory unstructured meshes  

#### Relay

- Added support for MPI reductions and broadcast
- Added support zero-copy pass to MPI for a wide set of calls
- Hardened notion of `known schema` vs `generic` MPI support
- Add heuristics with knobs for controlling use of HDF5 compact datasets and compression support

### Changed

#### General

- Moved to use BLT (https://github.com/llnl/blt) as our core CMake-based build system
- Use ints instead of bools in the Conduit C-APIs for wider compiler compatibility

#### Relay

- Improved error checking and error messages for HDF5 I/O support

### Fixed

#### General

- Fixed memory leaks in *conduit* 
- Bug fixes to support building on Visual Studio 2013
- Bug fixes for `conduit::Nodes` in the List Role

#### Relay

- Fixed memory leaks in *conduit_relay*    

## [0.2.1] - Released 2017-01-06

### Added

#### Blueprint

- Added support to the blueprint python module for the mesh and mcarray protocol methods 
- Added stand alone blueprint verify executable

### Changed

#### General

- Eliminated separate fortran libs by moving fortran symbols into their associated main libs
- Change Node set_external to support const Node ref
- Refactor path and file systems utils functions for clarity

### Fixed

#### General

- Added fixes to support static builds on BGQ using xlc and gcc
- Fixed missing install of fortran module files

#### Blueprint

- Fixed bug with verify of mesh/coords for rectilinear case


#### Relay

- Updated the version of civetweb used to avoid dlopen issues with SSL for static builds

## [0.2.0] - Released 2016-11-03

### Added

#### General

- Added const access to conduit::Node's children and a new NodeConstIterator
- Added support for building on Windows 
- Added more Python, C, and Fortran API support

#### Blueprint

- Added verify support for the mcarray and mesh protocols
- Added functions that create examples instances of mcarrays and meshes
- Added memory layout transform helpers for mcarrays
- Added a helper that creates a mesh blueprint index from a valid mesh

#### Relay

- Added entangle, a python script ssh tunneling solution
- Added extensive HDF5 I/O support for reading and writing between HDF5 files and conduit Node trees

###  Changed

#### General

- Changes to clarify concepts in the conduit::Node API 
- Improved unit test coverage
- Renamed source and header files for clarity and to avoid potential conflicts with other projects 

#### Relay

- Changed I/O protocol string names for clarity 
- Refactored the relay::WebServer and the Conduit Node Viewer application 

### Fixed

#### General

- Resolved several bugs across libraries 
- Resolved compiler warnings and memory leaks

## 0.1.0 - Released 2016-03-30

### Added
- Initial Open Source Release on GitHub

[Unreleased]: https://github.com/llnl/conduit/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/llnl/conduit/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/llnl/conduit/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/llnl/conduit/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/llnl/conduit/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/llnl/conduit/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/llnl/conduit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/llnl/conduit/compare/v0.1.0...v0.2.0
