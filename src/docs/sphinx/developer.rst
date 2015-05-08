.. ############################################################################
.. # Copyright (c) 2014, Lawrence Livermore National Security, LLC.
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

.. role:: bash(code)
   :language: bash

================================
Developer Documentation
================================

Source Code Repo Layout
------------------------
- **src/conduit**: Main library source
- **src/conduit_io**: Conduit I/O library source
- **src/conduit_mpi**: Conduit MPI library source
- **src/conduit_viz**: Conduit Visualization library source
- **src/tests**: Unit tests
- **src/tests/python**: Python Unit tests
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


Git Development Workflow 
------------------------

The master conduit source repository is hosted on LLNL's CZ Stash instance:

https://lc.llnl.gov/stash/projects/CON

We we are using a **Github Flow** model, which is a simpler variant of the confusing similar sounding **Git Flow** model.


Here are the basics: 

- Development is done on topic branches off the master.

- Merge to master is only done via a pull request.

- The master should always compile and pass all tests.

More details on GitHub Flow:

https://guides.github.com/introduction/flow/index.html

Here are some other rules to abide by:

- You *can* merge your own pull requests.

- After completing all intended work on branch, please delete the remote branch after merging to master. (Stash has an option to do this when merging a pull request.)



