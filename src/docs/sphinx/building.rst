.. ############################################################################
.. # Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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


This page provides details on several ways to build Conduit.

If you are building features that depend on third party libraries we recommend using :ref:`Spack <building_with_spack>`,
or :ref:`uberenv <building_with_uberenv>`, which leverages Spack. We also provide a 
:ref:`Docker example <building_with_docker>` that leverages Spack.


Getting Started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the Conduit repo:

* From Github

.. code:: bash
    
    git clone --recursive https://github.com/llnl/conduit.git


``--recursive`` is necessary because we are using a git submodule to pull in BLT (https://github.com/llnl/blt). 
If you cloned without ``--recursive``, you can checkout this submodule using:

.. code:: bash
    
    cd conduit
    git submodule init
    git submodule update


Configure a build:

Conduit uses CMake for its build system. These instructions assume ``cmake`` is in your path. 
We recommend CMake 3.9 or newer, for more details see :ref:`Supported CMake Versions <supported_cmake>`.

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


The Conduit Python module can be built for Python 2 or Python 3. To select a specific Python, set the CMake variable **PYTHON_EXECUTABLE** to path of the desired python binary. The Conduit Python module requires Numpy. The selected Python instance must provide Numpy, or PYTHONPATH must be set to include a Numpy install compatible with the selected Python install.
Note: You can not use compiled Python modules built with Python 2 in Python 3 and vice versa. You need to compile against the version you expect to use. 

* **ENABLE_MPI** - Controls if the conduit_relay_mpi library is built. *(default = OFF)*

 We are using CMake's standard FindMPI logic. To select a specific MPI set the CMake variables **MPI_C_COMPILER** and **MPI_CXX_COMPILER**, or the other FindMPI options for MPI include paths and MPI libraries.

 To run the mpi unit tests on LLNL's LC platforms, you may also need change the CMake variables **MPIEXEC** and **MPIEXEC_NUMPROC_FLAG**, so you can use srun and select a partition. (for an example see: src/host-configs/chaos_5_x86_64.cmake)

.. warning::
  Starting in CMake 3.10, the FindMPI **MPIEXEC** variable was changed to **MPIEXEC_EXECUTABLE**. FindMPI will still set **MPIEXEC**, but any attempt to change it before calling FindMPI with your own cached value of **MPIEXEC** will not survive, so you need to set **MPIEXEC_EXECUTABLE** `[reference] <https://cmake.org/cmake/help/v3.10/module/FindMPI.html>`_. 

* **HDF5_DIR** - Path to a HDF5 install *(optional)*. 

 Controls if HDF5 I/O support is built into *conduit_relay*.

* **SILO_DIR** - Path to a Silo install *(optional)*. 

 Controls if Silo I/O support is built into *conduit_relay*. When used, the following CMake variables must also be set:
 
 * **HDF5_DIR** - Path to a HDF5 install. (Silo support depends on HDF5) 


* **BLT_SOURCE_DIR** - Path to BLT.  *(default = "blt")*

 Defaults to "blt", where we expect the blt submodule. The most compelling reason to override is to share a single instance of BLT across multiple projects.
  

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

These files use standard CMake commands. To properly seed the cache, CMake *set* commands need to specify ``CACHE`` as follows:

.. code:: cmake

    set(CMAKE_VARIABLE_NAME {VALUE} CACHE PATH "")



.. _building_with_uberenv:

Building Conduit and Third Party Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use **Spack** (http://software.llnl.gov/spack) to help build Conduit's third party dependencies on OSX and Linux. Conduit builds on Windows as well, but there is no automated process to build dependencies necessary to support Conduit's optional features.

``scripts/uberenv/uberenv.py`` automates fetching spack, building and installinng third party dependencies, and can optionally  install Conduit as well. 


Building Third Party Dependencies for Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
  Conduit developers use ``bootstrap-env.sh`` and ``scripts/uberenv/uberenv.py`` to setup third party libraries for Conduit development.
  This path uses the Conduit Spack package and extra settings, including Spack compiler and external third party package details for some platforms.  For info on how to use the Conduit Spack package see :ref:`building_with_spack`.
  

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

 ==================== ============================================== ================================================
  Option               Description                                     Default
 ==================== ============================================== ================================================
  --prefix             Destination directory                          ``uberenv_libs``
  --spec               Spack spec                                     linux: **%gcc**
                                                                      osx: **%clang**
  --spack-config-dir   Folder with Spack settings files               linux: (empty)
                                                                      osx: ``scripts/uberenv/spack_configs/darwin/``
  -k                   Ignore SSL Errors                              **False**
  --install            Fully install conduit, not just dependencies   **False**
 ==================== ============================================== ================================================

The ``-k`` option exists for sites where SSL certificate interception undermines fetching
from github and https hosted source tarballs. When enabled, ``uberenv.py`` clones spack using:

.. code:: bash

    git -c http.sslVerify=false clone https://github.com/llnl/spack.git

And passes ``-k`` to any spack commands that may fetch via https.


Default invocation on Linux:

.. code:: bash

    python scripts/uberenv/uberenv.py --prefix uberenv_libs \
                                      --spec %gcc 

Default invocation on OSX:

.. code:: bash

    python scripts/uberenv/uberenv.py --prefix uberenv_libs \
                                      --spec %clang \
                                      --spack-config-dir scripts/uberenv/spack_configs/darwin/

For details on Spack's spec syntax, see the `Spack Specs & dependencies <http://spack.readthedocs.io/en/latest/basic_usage.html#specs-dependencies>`_ documentation.

 
You can edit yaml files under ``scripts/uberenv/spack_config/{platform}`` or use the **--spack-config-dir** option to specify a directory with compiler and packages yaml files to use with Spack. See the `Spack Compiler Configuration <http://spack.readthedocs.io/en/latest/getting_started.html#manual-compiler-configuration>`_
and `Spack System Packages
<http://spack.readthedocs.io/en/latest/getting_started.html#system-packages>`_
documentation for details.

For OSX, the defaults in ``spack_configs/darwin/compilers.yaml`` are X-Code's clang and gfortran from https://gcc.gnu.org/wiki/GFortranBinaries#MacOS. 

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


You can specify specific versions of a dependency using ``^``. For Example, to build Conduit with Python 3:


.. code:: bash

  spack install conduit+python ^python@3


.. _supported_cmake:

Supported CMake Versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We recommend CMake 3.9 or newer. We test building Conduit with CMake 3.3.1, 3.8.1 and 3.9.4. Other versions of CMake may work, however CMake 3.4.x to 3.7.x have specific issues with finding and using HDF5 and Python and C++11 support.



Using Conduit in Another Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under ``src/examples`` there are examples demonstrating how to use Conduit in a CMake-based build system (``using-with-cmake``) and via a Makefile (``using-with-make``).


.. _building_with_docker:

Building Conduit in a Docker Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under ``src/examples/docker/ubuntu`` there is an example ``Dockerfile`` which can be used to create an ubuntu-based docker image with a build of the Conduit. There is also a script that demonstrates how to build a Docker image from the Dockerfile (``example_build.sh``) and a script that runs this image in a Docker container (``example_run.sh``). The Conduit repo is cloned into the image's file system at ``/conduit``, the build directory is ``/conduit/build-debug``, and the install directory is ``/conduit/install-debug``.


Notes for Cray systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HDF5 and gtest use runtime features such as ``dlopen``. Because of this, building static on Cray systems commonly yields the following flavor of compiler warning:

.. code:: 

   Using 'zzz' in statically linked applications requires at runtime the shared libraries from the glibc version used for linking

You can avoid related linking warnings by adding the ``-dynamic`` compiler flag, or by setting the CRAYPE_LINK_TYPE environment variable:

.. code:: bash

  export CRAYPE_LINK_TYPE=dynamic

`Shared Memory Maps are read only <https://pubs.cray.com/content/S-0005/CLE%206.0.UP02/xctm-series-dvs-administration-guide-cle-60up02-s-0005/dvs-caveats>`_
on Cray systems, so updates to data using ``Node::mmap`` will not be seen between processes.






