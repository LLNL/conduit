.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

=======================================================
Passing Conduit Nodes between C++, Fortran, and Python
=======================================================

The ``cpp_fort_and_py`` example demonstrates how to pass Conduit Nodes between
C++, Fortran, and Python. It is a standalone example that you can build with
CMake against your Conduit install.

You can find this example under ``src/examples/cpp_fort_and_py`` in
Conduit's source tree, or under ``examples/conduit/cpp_fort_and_py`` in
a Conduit install.

It includes source for an embedded python interpreter and also shows how
to create a Fortran module that binds Conduit Nodes via Conduit's C-API.

It creates two executables:

 ============================ ===========================================
  **conduit_cpp_and_py_ex**    Demo of C++ to Python and vice versa
  **conduit_fort_and_py_ex**   Demo of Fortran to Python and vice versa
 ============================ ===========================================

This demos wrapping Conduit Nodes, effectively creating referenced
data across languages. You can also use `set_external` to directly access
and change zero-copied data.

Please see the main CMakeList.txt file for details on building and running:

``cpp_fort_and_py/CMakeLists.txt`` excerpt:

.. literalinclude:: ../../examples/cpp_fort_and_py/CMakeLists.txt
   :lines: 6-34
   :dedent: 2
