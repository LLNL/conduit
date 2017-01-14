.. ############################################################################
.. # Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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


* From LLNL's CZ Bitbucket Server (Mirror for LLNL Users)

.. code:: bash
    
    git clone https://{USER_NAME}@lc.llnl.gov/bitbucket/scm/con/conduit.git

Configure a build:

``config-build.sh`` is a simple wrapper for the cmake call to configure conduit. 
This creates a new out-of-source build directory ``build-debug`` and a directory for the install ``install-debug``.
It optionally includes a ``host-config.cmake`` file with detailed configuration options. 


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

The ``config-build.sh`` script uses your machine's hostname, the SYS_TYPE environment variable, and your platform name (via *uname*) to look for an existing host config file in the ``host-configs`` directory at the root of the conduit repo. If found, it passes the host config file to CMake via the `-C` command line option.

.. code:: bash
    
    cmake {other options} -C host-configs/{config_file}.cmake ../


You can find example files in the ``host-configs`` directory. 

These files use standard CMake commands. CMake *set* commands need to specify the root cache path as follows:

.. code:: cmake

    set(CMAKE_VARIABLE_NAME {VALUE} CACHE PATH "")


Bootstrapping Third Party Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use **Spack** (http://software.llnl.gov/spack) to automate builds of third party dependencies on OSX and Linux. Conduit builds on Windows as well, but there is no automated process to build dependencies necessary to support Conduit's optional features.

.. note::
  Conduit developers use ``bootstrap-env.sh`` and ``scripts/uberenv/uberenv.py`` to setup third party libraries for Conduit  development.  Due to this, the process builds more libraries than necessary for most use cases. For example, we build independent  installs of Python 2 and Python 3 to make it easy to check Python C-API compatibility during development. For users of conduit, we recommend using the Conduit package included with Spack. For info on how to use this package see :ref:`building_with_spack`.
  

On OSX and Linux, you can use ``bootstrap-env.sh`` (located at the root of the conduit repo) to help setup your development environment. This script uses ``scripts/uberenv/uberenv.py``, which leverages **Spack** to build all of the external third party libraries and tools used by Conduit. Fortran support is optional and all dependencies should build without a fortran compiler. After building these libraries and tools, it writes an initial *host-config* file and adds the Spack built CMake binary to your PATH so can immediately call the ``config-build.sh`` helper script to configure a conduit build.

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


When ``bootstrap-env.sh`` runs ``uberenv.py``, all command line arguments are forwarded:

.. code:: bash

    python scripts/uberenv/uberenv.py $@

So any options to ``bootstrap-env.sh`` are effectively ``uberenv.py`` options.

Uberenv Options for Building Third Party Dependencies
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

``uberenv.py`` has a few options that allow you to control how dependencies are built:

 ================== ==================================== ======================================
  Option             Description                          Default
 ================== ==================================== ======================================
  --prefix           Destination directory                ``uberenv_libs``
  --spec             Spack spec                           linux: **%gcc**
                                                          osx: **%clang**
  --compilers-yaml   Spack compilers settings file        ``scripts/uberenv/compilers.yaml``
  -k                 Ignore SSL Errors                    **False**
 ================== ==================================== ======================================

The ``-k`` option exists for sites where SSL certificate interception undermines fetching
from github and https hosted source tarballs. When enabled, ``uberenv.py`` clones spack using:

.. code:: bash

    git -c http.sslVerify=false clone https://github.com/llnl/spack.git

And passes ``-k`` to any spack commands that may fetch via https.


Default invocation on Linux:

.. code:: bash

    python scripts/uberenv/uberenv.py --prefix uberenv_libs \
                                      --spec %gcc \
                                      --compilers-yaml scripts/uberenv/compilers.yaml

Default invocation on OSX:

.. code:: bash

    python scripts/uberenv/uberenv.py --prefix uberenv_libs \
                                      --spec %clang \
                                      --compilers-yaml scripts/uberenv/compilers.yaml

For details on Spack's spec syntax, see the `Spack Specs & dependencies <http://spack.readthedocs.io/en/latest/basic_usage.html#specs-dependencies>`_ documentation.

 
You can edit ``scripts/uberenv/compilers.yaml`` or use the **--compilers-yaml** option to change the compiler settings
used by Spack. See the `Spack Compiler Configuration <http://spack.readthedocs.io/en/latest/getting_started.html#manual-compiler-configuration>`_
documentation for details.

For OSX, the defaults in ``compilers.yaml`` are X-Code's clang and gfortran from https://gcc.gnu.org/wiki/GFortranBinaries#MacOS. 

.. note::
    The bootstrapping process ignores ``~/.spack/compilers.yaml`` to avoid conflicts
    and surprises from a user's specific Spack settings on HPC platforms.

When run, ``uberenv.py`` checkouts a specific version of Spack from github as ``spack`` in the 
destination directory. It then uses Spack to build and install Conduit's dependencies into 
``spack/opt/spack/``. Finally, it generates a host-config file ``{hostname}.cmake`` in the 
destination directory that specifies the compiler settings and paths to all of the dependencies.


.. _building_with_spack:

Building Conduit and its Dependencies with Spack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
As of 1/4/2017, Spack's develop branch includes a `recipe <https://github.com/LLNL/spack/blob/develop/var/spack/repos/builtin/packages/conduit/package.py>`_ to build and install Conduit.

To install the latest released version of Conduit with all options (and also build all of its dependencies as necessary) run:

.. code:: bash
  
  spack install conduit

To build and install Conduit's github master branch run:
  
.. code:: bash
  
  spack install conduit@master


The Conduit Spack package provides several `variants <http://spack.readthedocs.io/en/latest/basic_usage.html#specs-dependencies>`_ that customize the options and dependencies used to build Conduit:

 ================== ==================================== ======================================
  Variant             Description                          Default
 ================== ==================================== ======================================
  **shared**          Build Conduit as shared libraries    ON (+shared)
  **cmake**           Build CMake with Spack               ON (+cmake)
  **python**          Enable Conduit Python support        ON (+python)
  **mpi**             Enable Conduit MPI support           ON (+mpi)
  **hdf5**            Enable Conduit HDF5 support          ON (+hdf5)
  **silo**            Enable Conduit Silo support          ON (+silo)
  **doc**             Build Conduit's Documentation        OFF (+docs)
 ================== ==================================== ======================================


Variants are enabled using ``+`` and disabled using ``~``. For example, to build Conduit with the minimum set of options (and dependencies) run:

.. code:: bash

  spack install conduit~python~mpi~hdf5~silo~docs



Using Conduit in Another Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under ``src/examples`` there are examples demonstrating how to use Conduit in a CMake-based build system (``using-with-cmake``) and via a Makefile (``using-with-make``).

Building Conduit in a Docker Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under ``src/examples/docker/ubuntu`` there is an example ``Dockerfile`` which can be used to create an ubuntu-based docker image with a build of the Conduit. There is also a script that demonstrates how to build a Docker image from the Dockerfile (``example_build.sh``) and a script that runs this image in a Docker container (``example_run.sh``). The Conduit repo is cloned into the image's file system at ``/conduit``, the build directory is ``/conduit/build-debug``, and the install directory is ``/conduit/install-debug``.






