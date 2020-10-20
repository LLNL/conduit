.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

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


Running Unit Tests via Valgrind
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use ctest's built-in  valgrind support to check for memory leaks in unit tests. Assuming valgrind is automatically detected when you run CMake to configure conduit, you can check for leaks by running:

.. code:: bash
    
    ctest -D ExperimentalBuild
    ctest -D ExperimentalMemCheck
    
The build system is setup to use **src/cmake/valgrind.supp** to filter memcheck results. We don't yet have all spurious issues suppressed, expect to see leaks reported for python and mpi tests. 


BLT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Conduit's CMake-based build system uses BLT (https://github.com/llnl/blt).

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







