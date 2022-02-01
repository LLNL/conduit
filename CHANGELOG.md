# Conduit Changelog
Notable changes to Conduit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project aspires to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.2] - Released 2022-02-01

### Fixed

#### Blueprint
- Fixed missing C++ include used by Blueprint Parmetis support.

## [0.8.1] - Released 2022-01-25

### Added
#### General
- Added `CONDUIT_DLL_DIR` env var support on windows, for cases where Conduit DLLs are not installed directly inside the Python Module.

#### Blueprint
- Allow adjsets to be used in `blueprint::mesh::partition` to determine global vertex ids.
- Added partial matset support to `blueprint::mesh::partition` and `blueprint::mesh::combine`.

### Fixed

#### General
- Fixed CMake bug with `ENABLE_RELAY_WEBSERVER` option.
- Fixed build and test issues with Python >= 3.8  on Windows.

#### Blueprint
- Fixed a bug in `blueprint::mesh::partition` where adjsets could be missing in new domains.
- Fixed a bug with `blueprint::mesh::matset::to_silo` and uni-buffer matsets.


## [0.8.0] - Released 2021-12-20


### Added

#### General
- Added `setup.py` for building and installing Conduit and its Python module via pip
- Added DataAccessor class that helps write generic algorithms that consume data arrays using expected types.
- Added support to register custom memory allocators and a custom data movement handler. This allows conduit to move trees of data between heterogenous memory spaces (e.g. CPU and GPU memory). See conduit_utils.hpp for API details.

#### Blueprint
- Added `conduit::blueprint::{mpi}::partition` function that provides a general N-to-M partition capability for Blueprint Meshes. This helps with load balancing and other use cases, including fusing multi-domain data to simplifying post processing. This capability supports several options, see (https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh_partition.html) for more details.
- Added a `Table` blueprint used to represent tables of numeric data. See (https://llnl-conduit.readthedocs.io/en/latest/blueprint_table.html) more details.
- Added `conduit::blueprint::{mpi}::flatten` which transforms Blueprint Meshes into Blueprint Tables. This transforms Mesh Blueprint data into a form that is more easily digestible in machine learning applications.
- Added `conduit::blueprint::mpi::generate_partition_field`, which uses Parmetis to create a field that identifies how to load balance an input mesh elements.  This field can be used as a Field selection input to `conduit::blueprint::mpi::partition` function.
- Added the`blueprint::mesh::examples::polychain` example. It is an example of a polyhedral mesh. See Mesh Blueprint Examples docs (https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#polychain) for more details.
- Added a new function signature for `blueprint::mesh::topology::unstructured::generate_sides`, which performs the same task as the original and also takes fields from the original topology and maps them onto the new topology.
- Added `blueprint::mpi::mesh::to_polygonal`, which provides a MPI aware conversion Blueprint Structured AMR meshes to a Blueprint Polyhedral meshes.
- Added a host of `conduit::blueprint::mpi::mesh::generate_*` methods, which are the MPI parallel equivalents of the `conduit::blueprint::mesh::topology::unstructured::generate_*` functions.
- Added the `conduit::blueprint::mpi::mesh::find_delegate_domain` function, which returns a single delegate domain for the given mesh across MPI ranks (useful when all ranks need mesh information and some ranks can have empty meshes).
- Added check and transform functions for the newly-designated `pairwise` and `maxshare` variants of `adjsets`. For more information, see the `conduit::blueprint::mesh::adjset` namespace.
- Added `mesh::topology::unstructured::to_polytopal` as an alias to `mesh::topology::unstructured::to_polygonal`, to reflect that both polygonal and polyhedral are supported.
- Added `conduit::blueprint::mpi::mesh::to_polytopal` as an alias to `conduit::blueprint::mpi::mesh::to_polygonal` and `conduit::blueprint::mpi::mesh::to_polyhedral`.


#### Relay
- Added `conduit::relay::io::hdf5_identifier_report` methods, which create conduit nodes that describes active hdf5 resource handles.


### Changed

#### General
- Updated CMake logic to provide more robust Python detection and better support for HDF5 installs that were built with CMake.
- Improved Node::diff and Node::diff_compatible to show string values when strings differ.
- `conduit::Node::print()` and in Python Node `repr` and `str` now use `to_summary_string()`. This reduces the output for large Nodes. Full output is still supported via `to_string()`, `to_yaml()`, etc methods.

#### Blueprint
- The `blueprint::mesh::examples::polytess` function now takes a new argument, called `nz`, which allows it to be extended into 3 dimensions. See Mesh Blueprint Examples docs (https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#polytess) for more details.
- Added support for both `const` and non-`const` inputs to the `conduit::blueprint::mesh::domains` function.
- Improved mesh blueprint index generation logic (local and MPI) to support domains with different topos, fields, etc. 
- Deprecated accepting `npts_z !=0` for 2D shape types in `conduit::blueprint::mesh::examples::{braid,basic,grid}`. They issue a `CONDUIT_INFO` message when this detected and future versions will issue a `CONDUIT_ERROR`.
- An empty Conduit Node is now considered a valid multi-domain mesh. This change was made to make serial uses cases better match sparse MPI multi-domain use cases. Existing code that relied `mesh::verify` to exclude empty Nodes will now need an extra check to see if an input mesh has data.
- Added MPI communicator argument to `conduit::blueprint::mpi::mesh::to_polygonal` and `conduit::blueprint::mpi::mesh::to_polyhedral`.



#### Relay
- Added CMake option (`ENABLE_RELAY_WEBSERVER`, default = `ON`) to control if Conduit's Relay Web Server support is built. Down stream codes can check for support via header ifdef `CONDUIT_RELAY_WEBSERVER_ENABLED` or at runtime in `conduit::relay::about`.
- Added support to compile against HDF5 1.12.

### Fixed

#### General
- Avoid compile issue with using `_Pragma()` with Python 3.8 on Windows
- `conduit_node` and `conduit_datatype` in the C API are no longer aliases to `void` so that callers cannot pass just any pointer to the APIs.
- Fixed memory over read issue with Fortran API due to int vs bool binding error. Fortran API still provides logical returns for methods like conduit_node_has_path() however the binding implementation now properly translates C_INT return codes into logical values.
- Fixed a subtle bug with Node fetch and Object role initialization.

#### Blueprint
- Fixed a bug that was causing the `conduit::blueprint::mesh::topology::unstructured::generate_*` functions to produce bad results for polyhedral input topologies with heterogeneous elements (e.g. tets and hexs).
- Fixed a bug with `conduit::relay::io::blueprint::write_mesh` that undermined `truncate=true` option for root-only style output.
- Fixed options parsing bugs and improved error messages for the `conduit_blueprint_verify` exe.

#### Relay
- Changed HDF5 offset support to use 64-bit unsigned integers for offsets, strides, and sizes.
- Fixed a bug with `conduit::relay::mpi::io::blueprint::save_mesh` where `file_style=root_only` could crash or truncate output files.
- Fixed a bug with inconsistent HDF5 handles being used in some cases when converting existing HDF5 Datasets from fixed to extendable.


## [0.7.2] - Released 2021-05-19

### Added

#### General
- Added the `cpp_fort_and_py` standalone example. It demos passing Conduit Nodes between C++, Fortran, and Python. See the related tutorial docs (https://llnl-conduit.readthedocs.io/en/latest/tutorial_cpp_fort_and_py.html) for more details.
- Added  `conduit::utils::info_handler()`, `conduit::utils::warning_handler()`, and `conduit::utils::error_handler()`  methods, which provide access to the currently registered info, warning, and error handlers.
- Added DataType::index_t method. Creates a DataType instance that describes an `index_t`, which is an alias to either `int32`, or `int 64` controlled by the `CONDUIT_INDEX_32` compile time option.
- Added several more methods to Python DataType interface
- Removed duplicate install of CMake exported target files that served as a bridge for clients using old style paths.

### Changed

#### General
- Updated to newer version of uberenv and changed to track spack fork https://github.com/alpine-dav/spack (branch: conduit/develop).
- Updated to newer version of BLT to leverage CMake's FindMPI defined targets when using CMake 3.15 or newer.
- Changed `rapidjson` namespace to `conduit_rapidjson` to avoid symbol collisions with other libraries using RapidJSON.

#### Blueprint
- The semantics of `conduit::blueprint::mesh::verify` changed. An empty conduit Node is now considered a valid multi-domain mesh with zero domains. If you always expect mesh data, you can add an additional check for empty to craft code that works for both the old and new verify semantics.

#### Relay
- Added Relay HDF5 support for reading and writing to an HDF5 dataset with offset.
- Added `conduit::relay::io::hdf5::read_info` which allows you to obtain metadata from an HDF5 file.
- Added configure error when conduit lacks MPI support and HDF5 has MPI support

### Fixed

#### General
- Fixed missing implementation of DataType::is_index_t
- Fixed issue with compiling t_h5z_zfp_smoke.cpp against an MPI-enabled HDF5.

#### Blueprint
- Fixed a bug that caused HDF5 reference paths to appear twice in Relay HDF5 Error messages.

#### Blueprint
- `conduit::relay::io::blueprint.read_mesh` now uses read only I/O handles.


## [0.7.1] - Released 2021-02-11

### Fixed

#### General
- Fixed a bug with Conduit's C interface including C++ headers.

#### Blueprint
- Fixed a bug with `blueprint::mesh::matset::to_silo` and `blueprint::mesh::field::to_silo` that could modify input values.

## [0.7.0] - Released 2021-02-08

### Changed

#### General
- Conduit now requires C++11 support.
- Python Node repr string construction now uses `Node.to_summary_string()`

### Added
- CMake: Added extra check for include dir vs fully resolved hdf5 path.

#### General
- Added a builtin sandboxed header-only version of fmt. The namespace and directory paths were changed to `conduit_fmt` to avoid potential symbol collisions with other codes using fmt. Downstream software can use by including `conduit_fmt/conduit_fmt.h`.
- Added support for using C++11 initializer lists to set Node and DataArray values from numeric arrays. See C++ tutorial docs (https://llnl-conduit.readthedocs.io/en/latest/tutorial_cpp_numeric.html#c-11-initializer-lists) for more details.
- Added a Node::describe() method. This method creates a new node that mirrors the current Node, however each leaf is replaced by summary stats and a truncated display of the values. For use cases with large leaves, printing the describe() output Node is much more helpful for debugging and understanding vs wall of text from other to_string() methods.
- Added conduit::utils::format methods. These methods use fmt to format strings that include fmt style patterns. The formatting arguments are passed as a conduit::Node tree. The `args` case allows named arguments (args passed as object) or ordered args (args passed as list). The `maps` case also supports named or ordered args and works in conjunction with a `map_index`. The `map_index` is used to fetch a value from an array, or list of strings, which is then passed to fmt. The `maps` style of indexed indirection supports generating path strings for non-trivial domain partition mappings in Blueprint. This functionality is also available in Python, via the  `conduit.utils.format` method.
- Added `DataArray::fill` method, which set all elements of a DataArray to a given value.
- Added `Node::to_summary_string` methods, which allow you to create truncated strings that describe a node tree, control the max number of children and max number of elements shown.
- Added python support for `Node.to_summary_string`

#### Relay
- Added Relay IO Handle mode support for `a` (append) and `t` (truncate).  Truncate allows you to overwrite files when the handle is opened. The default is append, which preserves prior IO Handle behavior.
- Added `conduit::relay::io::blueprint::save_mesh` variants, these overwrite existing files (providing relay save semantics) instead of adding mesh data to existing files. We recommend using  `save_mesh` for most uses cases, b/c in many cases `write_mesh` to an existing HDF5 file set can fail due to conflicts with the current HDF5 tree.
- Added `conduit::relay::io::blueprint::load_mesh` variants, these reset the passed node before reading mesh data (providing relay load semantics). We recommend using  `load_mesh` for most uses cases.
- Added `truncate` option to `conduit::relay::io::blueprint::write_mesh`, this is used by `save_mesh`.
- Improve capture and reporting of I/O errors in `conduit::relay::[mpi::]io::blueprint::{save_mesh|write_mesh}`. Now in the MPI case, If any rank fails to open or write to a file all ranks will throw an exception.
- Added yaml detection support to `conduit::relay::io:identify_file_type`.

#### Blueprint
- Added `conduit::blueprint::mesh::matset::to_silo()` which converts a valid blueprint matset to a node that contains arrays that follow Silo's sparse mix slot volume fraction representation.
- Added `conduit::blueprint::mesh::field::to_silo()` which converts a valid blueprint field and matset to a node that contains arrays that follow Silo's sparse mix slot volume fraction representation.
- Added `material_map` to `conduit::blueprint::mesh:matset::index`, to provide an explicit material name to id mapping.
- Added `mat_check` field to `blueprint::mesh::examples::venn`. This field encodes the material info in a scalar field and in the `matset_values` in a way that can be used to easily compare and verify proper construction in other tools.

### Fixed

#### Relay
- Fixed bug in the Relay IOHandle Basic that would create unnecessary "_json" schema files to be written to disk upon open().

### Removed

#### General
- Removed `Node::fetch_child` and `Schema::fetch_child` methods for v0.7.0. (Deprecated in v0.6.0 -- prefer `fetch_existing`)
- Removed `Schema::to_json` method variants with `detailed` for v0.7.0. (Deprecated in v0.6.0 -- prefer standard `to_json`)
- Removed `Schema::save` method variant with `detailed` for v0.7.0. (Deprecated in v0.6.0 -- prefer standard `save`)
- The `master` branch was removed from GitHub (Deprecated in v0.6.0 -- replaced by the `develop` branch)

#### Relay
- Removed `conduit::relay::io_blueprint::save` methods for v0.7.0. (Deprecated in v0.6.0 -- prefer `conduit::relay::io::blueprint::save_mesh`)


## [0.6.0] - Released 2020-11-02

### Added

#### General
- Added support for children with names that include `/`. Since slashes are part of Conduit's hierarchical path mechanism, you must use explicit methods (add_child(), child(), etc) to create and access children with these types of names. These names are also supported in all basic i/o cases (JSON, YAML, Conduit Binary).
- Added Node::child and Schema::child methods, which provide access to existing children by name.
- Added Node::fetch_existing and Schema::fetch_existing methods, which provide access to existing paths or error when given a bad path.
- Added Node::add_child() and Node::remove_child() to support direct operations and cases where names have `/` s.
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
- Added `conduit::relay::mpi::wait` and `conduit::relay::mpi::wait_all` functions. These functions consolidate the logic supporting both `isend` and `irecv` requests. `wait_all` supports cases where both sends and receives were posted, which is a common for non-trivial point-to-point communication use cases.


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
- Fixed a bug with auto detect of protocol for Node.load
- Fixed bugs with auto detect of protocol for Node.load and Node.save in the Python interface


#### Relay
- Use H5F_ACC_RDONLY in relay::io::is_hdf5_file to avoid errors when checking files that already have open HDF5 handles.
- Fixed compatibility check for empty Nodes against HDF5 files with existing paths

### Changed

#### General
- Conduit's main git branch was renamed from `master` to `develop`. To allow time for folks to migrate, the `master` branch is active but frozen and will be removed during the `0.7.0` release.
- We recommend a C++11 (or newer) compiler, support for older C++ standards is deprecated and will be removed in a future release.
- Node::fetch_child and Schema::fetch_child are deprecated in favor of the more clearly named Node::fetch_existing and Schema::fetch_existing. fetch_child variants still exist, but will be removed in a future release.
- Python str() methods for Node, Schema, and DataType now use their new to_string() methods.
- DataArray<T>::to_json(std::ostring &) is deprecated in favor DataArray<T>::to_json_stream. to_json(std::ostring &) will be removed in a future release.
- Schema::to_json and Schema::save variants with detailed (bool) arg are deprecated. The detailed arg was never used. These methods will be removed in a future release.
- Node::print() now prints yaml instead of json.
- The string return variants of `about` methods now return yaml strings instead of json strings.
- Sphinx Docs code examples and outputs are now included using start-after and end-before style includes.
- Schema to_json() and to_json_stream() methods were expanded to support indent, depth, pad and end-of-element args.
- In Python, conduit.Node() repr now returns the YAML string representation of the Node. Perviously verbose `conduit_json` was used, which was overwhelming.
- conduit.about() now reports the git tag if found, and `version` was changed to add git sha and status (dirty) info to avoid confusion between release and development installs.

#### Relay
- Provide more context when a Conduit Node cannot be written to a HDF5 file because it is incompatible with the existing HDF5 tree. Error messages now provide the full path and details about the incompatibility.
- `conduit::relay::io_blueprint::save` functions are deprecated in favor of `conduit::relay::io::blueprint::write_mesh`
- `conduit::relay::io::blueprint::write_mesh` functions were pulled in from Ascent's Blueprint export logic.
- `conduit_relay_io_mpi` lib now depends on `conduit_relay_io`. Due to this change, a single build supports either ADIOS serial (no-mpi) or ADIOS with MPI support, but not both. If conduit is configured with MPI support, ADIOS MPI is used.
- The functions `conduit::relay::mpi::wait_send` and `conduit::relay::mpi::wait_recv` now use `conduit::relay::mpi::wait`. The functions `wait_send` and `wait_recv` exist to preserve the old API, there is no benefit to use them over `wait`.
- The functions `conduit::relay::mpi::wait_all_send` and `conduit::relay::mpi::wait_all_recv` now use `conduit::relay::mpi::wait_all`. The functions `wait_all_send` and `wait_all_recv` exist to preserve the old API, there is no benefit to use them over `wait_all`.


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

[Unreleased]: https://github.com/llnl/conduit/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/llnl/conduit/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/llnl/conduit/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/llnl/conduit/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/llnl/conduit/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/llnl/conduit/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/llnl/conduit/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/llnl/conduit/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/llnl/conduit/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/llnl/conduit/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/llnl/conduit/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/llnl/conduit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/llnl/conduit/compare/v0.1.0...v0.2.0
