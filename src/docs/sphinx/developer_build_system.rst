.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

.. role:: bash(code)
   :language: bash


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



