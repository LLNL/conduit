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

.. _building:

=================
Building
=================

Getting Started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the Conduit repo:

* From Github

.. code:: bash
    
    git clone https://github.com/llnl/conduit.git


* From LLNL's CZ Stash Instance (LLNL Users)

.. code:: bash
    
    git clone https://{USER_NAME}@lc.llnl.gov/stash/scm/con/conduit.git


Configure a build:

``config-build.sh`` is a simple wrapper for the cmake call to configure conduit. 
This creates a new out-of-source build directory ``build-debug`` and a directory for the install ``install-debug``.
It will optionally include a ``host-config.cmake`` file with detailed configuration options. 


.. code:: bash
    
    cd conduit
    ./config-build.sh


Build, test, and install Conduit:

.. code:: bash
    
    cd build-debug
    make -j 8
    make test
    make install



Build Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core Conduit library has no dependencies outside of the repo, however Conduit provides optional support for I/O and Communication (MPI) features that require externally built third party libraries.  

Conduit's build system supports the following CMake options:

* **BUILD_SHARED_LIBS** - Controls if shared (ON) or static (OFF) libraries are built. *(default = ON)* 
* **ENABLE_TESTS** - Controls if unit tests are built. *(default = ON)* 
* **ENABLE_DOCS** - Controls if the Conduit documentation is built (when sphinx and doxygen are found ). *(default = ON)*
* **ENABLE_COVERAGE** - Controls if code coverage compiler flags are used to build Conduit. *(default = OFF)*
* **ENABLE_PYTHON** - Controls if the Conduit Python module is built. *(default = OFF)*


 The Conduit Python module will build for both Python 2 and Python 3. To select a specific Python, set the CMake variable **PYTHON_EXECUTABLE** to path of the desired python binary. The Conduit Python module requires Numpy. The selected Python instance must provide Numpy, or PYTHONPATH must be set to include a Numpy install compatible with the selected Python install. 

* **ENABLE_MPI** - Controls if the conduit_relay_mpi library is built. *(default = OFF)*

 We are using CMake's standard FindMPI logic. To select a specific MPI set the CMake variables **MPI_C_COMPILER** and **MPI_CXX_COMPILER**, or the other FindMPI options for MPI include paths and MPI libraries.

 To run the mpi unit tests on LLNL's LC platforms, you may also need change the CMake variables **MPIEXEC** and **MPIEXEC_NUMPROC_FLAG**, so you can use srun and select a partition. (for an example see: src/host-configs/chaos_5_x86_64.cmake)

* **HDF5_DIR** - Path to a HDF5 install *(optional)*. 
 Controls if HDF5 I/O support is built into *conduit_relay*.

* **SILO_DIR** - Path to a Silo install *(optional)*. 
 Controls if Silo I/O support is built into *conduit_relay*. When used, the following CMake variables must also be set:
 
 * **HDF5_DIR** - Path to a HDF5 install. (Silo support depends on HDF5) 

Installation Path Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Conduit's build system provides an **install** target that installs the Conduit libraires, headers, python modules, and documentation. These CMake options allow you to control install destination paths:

* **CMAKE_INSTALL_PREFIX** - Standard CMake install path option *(optional)*.

* **PYTHON_MODULE_INSTALL_PREFIX** - Path to install Python modules into *(optional)*.

 When present and **ENABLE_PYTHON** is ON, Conduit's Python modules will be installed to ``${PYTHON_MODULE_INSTALL_PREFIX}`` directory instead of ``${CMAKE_INSTALL_PREFIX}/python-modules``.


Host Config Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To handle build options, third party library paths, etc we rely on CMake's initial-cache file mechanism. 


.. code:: bash
    
    cmake -C config_file.cmake


We call these initial-cache files *host-config* files, since we typically create a file for each platform or specific hosts if necessary. 

The ``config-build.sh`` script will use your machine's hostname, the SYS_TYPE environment variable, and your platform name (via *uname*) to look for an existing host config file in the ``host-configs`` directory at the root of the conduit repo. If found, it will pass the host config file to CMake via the `-C` command line option.

.. code:: bash
    
    cmake {other options} -C host-configs/{config_file}.cmake ../


You can find example files in the ``host-configs`` directory. 

These files use standard CMake commands. CMake *set* commands need to specify the root cache path as follows:

.. code:: cmake

    set(CMAKE_VARIABLE_NAME {VALUE} CACHE PATH "")


Bootstrapping Third Party Dependencies 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use ``bootstrap-env.sh`` (located at the root of the conduit repo) to help setup your development environment on OSX and Linux. This script uses ``scripts/uberenv/uberenv.py``, which leverages **Spack** (http://software.llnl.gov/spack) to build the external third party libraries and tools used by Conduit. Fortran support in is optional, dependencies should build without fortran. After building these libraries and tools, it writes an initial *host-config* file and adds the Spack built CMake binary to your PATH, so can immediately call the ``config-build.sh`` helper script to configure a conduit build.

.. code:: bash
    
    #build third party libs using spack
    source bootstrap-env.sh
    
    #copy the generated host-config file into the standard location
    cp uberenv_libs/`hostname`*.cmake to host-configs/
    
    # run the configure helper script
    ./config-build.sh

    # or you can run the configure helper script and give it the 
    # path to a host-config file 
    ./config-build.sh uberenv_libs/`hostname`*.cmake


Compiler Settings for Third Party Dependencies 
++++++++++++++++++++++++++++++++++++++++++++++++
You can edit ``scripts/uberenv/compilers.yaml`` to change the compiler settings
passed to Spack. See the `Spack Compiler Configuration    <http://software.llnl.gov/spack/basic_usage.html#manual-compiler-configuration>`_   
documentation for details.

For OSX, the defaults in ``compilers.yaml`` are clang from X-Code and gfortran from https://gcc.gnu.org/wiki/GFortranBinaries#MacOS. 

.. note::
    The bootstrapping process ignores ``~/.spack/compilers.yaml`` to avoid conflicts
    and surprises from a user's specific Spack settings on HPC platforms.


Building with Spack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. note::
  Conduit developers use ``scripts/uberenv/uberenv.py`` to setup third party libraries for Conduit 
  development.  Due to this, the process builds more libraries than necessary for most use cases.
  For example, we build independent installs of Python 2 and Python 3 to make it easy 
  to check Python C-API compatibility during development. In the near future, we plan to 
  provide a Spack package to simplify deployment.


