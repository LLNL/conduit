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
.. # For details, see: http://llnl.github.io/conduit/.
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
* **src/libs/**

 * **conduit/** - Main Conduit library source
 * **relay/** - Relay libraries source
 * **blueprint/** - Blueprint library source

* **src/tests/**

 * **conduit/** - Unit tests for the main Conduit library
 * **relay/** - Unit tests for Conduit Relay libraries
 * **blueprint/** - Unit tests for Blueprint library
 * **thirdparty/** - Unit tests for third party libraries

* **src/examples/** - Basic examples related to building and using Conduit 

.. (see :ref:`_using_in_another_project` ?)

* **src/docs/** -  Documentation 
* **src/thirdparty_builtin/** - Third party libraries we build and manage directly


Build System Info
-------------------

Configuring with CMake
~~~~~~~~~~~~~~~~~~~~~~

See :ref:`building` in the User Documentation. 

Important CMake Targets
~~~~~~~~~~~~~~~~~~~~~~~~

- **make**: Builds Conduit.

- **make test**: Runs unit tests.

- **make docs**: Builds sphinx and doxygen documentation.

- **make install**: Installs conduit libraries, headers, and documentation to ``CMAKE_INSTALL_PREFIX``

Adding a Unit Test
~~~~~~~~~~~~~~~~~~~
- Create a test source file in ``src/tests/{lib_name}/``
- All test source files should have a ``t_`` prefix on their file name to make them easy to identify.
- Add the test to build system by editing ``src/tests/{lib_name}/CMakeLists.txt``


Git Development Workflow 
------------------------

Conduit's primary source repository and issue tracker are hosted on github:

https://github.com/llnl/conduit


We are using a **Github Flow** model, which is a simpler variant of the confusingly similar sounding **Git Flow** model.

Here are the basics: 

- Development is done on topic branches off the master.

- Merge to master is only done via a pull request.

- The master should always compile and pass all tests.

- Releases are tagged off of master.

More details on GitHub Flow:

https://guides.github.com/introduction/flow/index.html

Here are some other rules to abide by:

- If you have write permissions for the Conduit repo, you *can* merge your own pull requests.

- After completing all intended work on branch, please delete the remote branch after merging to master. (Github has an option to do this after you merge a pull request.)







