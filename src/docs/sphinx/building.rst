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
.. # For details, see https://lc.llnl.gov/conduit/.
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

Clone conduit from LLNL's CZ Stash

.. code:: bash
    
    git clone https://{USER_NAME}@lc.llnl.gov/stash/scm/con/conduit.git


Configure a build

*config-build.sh* is a simple wrapper for the cmake call to configure conduit. 
This creates a new out-of-source build directory *build-debug* and a directory for the install *install-debug*.
It will optionally include a *host-config.cmake* file with detailed configuration options. 


.. code:: bash
    
    cd conduit
    ./config-build.sh


Build, test, and install conduit:

.. code:: bash
    
    cd build-debug
    make -j 8
    make test
    make install



Build Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conduit's build system supports the following CMake options:

* **ENABLE_TESTS** - Controls if unit tests are built. *(default = ON)* 
* **ENABLE_PYTHON** - Controls if the conduit Python module is built. *(default = OFF)*

To select a specific Python, set the CMake variable **PYTHON_EXECUTABLE** to path of the desired python binary.
Conduit's Python module requires Numpy. The selected Python install must provide Numpy, or PYTHONPATH must be set to include a Numpy install compatible with the selected Python install. 

* **ENABLE_MPI** - Controls if the conduit_mpi library is built. *(default = OFF)*

We are using CMake's standard FindMPI logic. To select a specific MPI set the CMake variables **MPI_C_COMPILER** and **MPI_CXX_COMPILER**, or the other FindMPI option for MPI include paths and MPI libraries.

For running mpi unit tests on LC platforms, you may also need change the CMake variables **MPIEXEC** and **MPIEXEC_NUMPROC_FLAG**, so you can use srun and select a partition. (see: src/host-configs/surface.cmake)

* **ENABLE_SILO** - Controls if the Silo I/O support is built. *(default = OFF)*

If enabled, the following CMake variables must also be set:

 * **SILO_DIR** - Path to a silo install. 
 * **HDF5_DIR** - Path to a hdf5 install. 
 * **SZIP_DIR** - Path to a szip install. 

Host Config Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To handle build options, thirdparty library paths, etc we rely on CMake's initial-cache file mechanism. 


.. code:: bash
    
    cmake -C config_file.cmake


We call these initial-cache files *host-config* files, since we typically create a file for each platform or specific hosts if necessary. 

The *config-build.sh* script will use your machine's hostname, the SYS_TYPE environment variable, and your platform name (via *uname*) to look for an existing host config file in the *host-configs* directory at the root of the conduit repo. If found, it will pass the host config file to cmake via the *-C* command line option.

.. code:: bash
    
    cmake {other options} -C host-configs/{config_file}.cmake ../


You can view several example files under the *host-configs* directory. 

These file use standard cmake commands. CMake *set* commands need to specify the root cache path as follows:

.. code:: cmake

    set(CMAKE_VARIABLE_NAME {VALUE} CACHE PATH "")


Bootstrapping Thirdparty Dependencies 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use *bootstrap-env.sh* (located at the root of the conduit repo) to help setup your development environment on OSX and Linux. This script uses *scripts/uberenv*, which leverages **Spack** (https://scalability-llnl.github.io/spack) to build external thirdparty libraries and tools used by conduit.
It also writes a initial host-config file for you and adds the spack built cmake to your path, so can directly call the *config-build.sh* helper script to configure a conduit build.

.. code:: bash
    
    #build thirdparty libs using spack
    source bootstrap-env.sh
    
    #copy the generated host-config file into the standard location
    cp uberenv_libs/`hostname`.cmake to host-configs/
    
    # run the configure helper script
    ./config-build.sh






