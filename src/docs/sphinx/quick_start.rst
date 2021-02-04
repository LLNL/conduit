.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

.. _getting_started:

================================
Quick Start
================================


Installing Conduit and Third Party Dependencies 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quickest path to install conduit and its dependencies is via :ref:`uberenv <building_with_uberenv>`:

.. code:: bash
    
    git clone --recursive https://github.com/llnl/conduit.git
    cd conduit
    python scripts/uberenv/uberenv.py --install --prefix="build"


After this completes, ``build/conduit-install`` will contain a Conduit install.

For more details about building and installing Conduit see :doc:`building`. This page provides detailed info about Conduit's CMake options, :ref:`uberenv <building_with_uberenv>` and :ref:`Spack <building_with_spack>` support. We also provide info about :ref:`building for known HPC clusters using uberenv <building_known_hpc>` and a :ref:`Docker example <building_with_docker>` that leverages Spack.


Using Conduit in Your Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The install includes examples that demonstrate how to use Conduit in a CMake-based build system  and via a Makefile.

CMake-based build system example (see: ``examples/conduit/using-with-cmake``):

.. literalinclude:: ../../examples/using-with-cmake/CMakeLists.txt 
   :lines: 4-

Makefile-based build system example (see: ``examples/conduit/using-with-make``):

.. literalinclude:: ../../examples/using-with-make/Makefile
   :lines: 4-


Learning Conduit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get starting learning the core Conduit API, see the Conduit Tutorials for  :doc:`C++ <tutorial_cpp>` and :doc:`Python <tutorial_python>`.







