.. #############################################################################
.. # Copyright (c) 2014, Lawrence Livermore National Security, LLC
.. # Produced at the Lawrence Livermore National Laboratory. 
.. # 
.. # All rights reserved.
.. # 
.. # This source code cannot be distributed without further review from 
.. # Lawrence Livermore National Laboratory.
.. #############################################################################

.. role:: bash(code)
   :language: bash

================================
Developer Documentation
================================

Source Code Repo Layout
------------------------
- **src/conduit**: Main library source
- **src/tests**: Unit tests
- **src/docs**: Documentation 
- **src/thirdparty_builtin**:  Third party libraries we build and manage directly


Build System Info
-------------------

Configuring with CMake
~~~~~~~~~~~~~~~~~~~~~~

Conduit uses a CMake based build system. You can run CMake directly to configure an out-of-source build, or use the "config-build.sh" helper script, which does the following:

- Creates a fresh *build-debug* folder.
- Runs :bash:`cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../install-debug` from within the *build-debug* folder.


Important CMake Targets
~~~~~~~~~~~~~~~~~~~~~~~~

- **make**: Builds the conduit library.

- **make test**: Runs conduit unit tests.

- **make docs**: Builds sphinx and doxygen documentation.

- **make install**: Installs the conduit library, headers, and documentation to `CMAKE_INSTALL_PREFIX`

Adding a Unit Test
~~~~~~~~~~~~~~~~~~~
- Create a test source file in *src/tests*
- Add the test to build system by editing *src/tests/CMakeLists.txt*