.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

===========================
String Formatting Helpers
===========================

fmt
----------------
For C++ users, conduit includes a built-in version of the ``fmt`` library (https://fmt.dev/).
Since other projects also bundle ``fmt``, the conduit version is modified to
place everything in the ``conduit_fmt`` namespace instead of the default ``fmt`` namespace.
This is a safe approach to avoid potential confusion and static linking consequences.

When using conduit in C++, you can use its built-in  ``fmt`` as follows:


.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_utils.cpp
   :start-after: BEGIN_EXAMPLE("using_fmt")
   :end-before:  END_EXAMPLE("using_fmt")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_utils_out.txt
   :start-after: BEGIN_EXAMPLE("using_fmt")
   :end-before:  END_EXAMPLE("using_fmt")


conduit::utils::format
-----------------------

In addition to direct ``fmt`` support, conduit utils provides ``conduit::utils::format`` methods that enable ``fmt`` style string formatting with the arguments are passed as a ``conduit::Node`` tree.
These simplify use cases such as generating path string, allowing the pattern string and 
arguments to be stored as part of a conduit hierarchy (and in HDF5, YAML, etc files).
This feature is also available in Conduit's Python API (``conduit.utils.format``).


conduit::utils::format(string, args)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``args`` case allows named arguments (args passed as object) or ordered args (args passed as list).

**conduit::utils::format(string, args) -- object case**: 

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_utils.cpp
   :start-after: BEGIN_EXAMPLE("using_utils_fmt_args_obj")
   :end-before:  END_EXAMPLE("using_utils_fmt_args_obj")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_utils_out.txt
   :start-after: BEGIN_EXAMPLE("using_utils_fmt_args_obj")
   :end-before:  END_EXAMPLE("using_utils_fmt_args_obj")


**conduit::utils::format(string, args) -- list case**: 

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_utils.cpp
   :start-after: BEGIN_EXAMPLE("using_utils_fmt_args_list")
   :end-before:  END_EXAMPLE("using_utils_fmt_args_list")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_utils_out.txt
   :start-after: BEGIN_EXAMPLE("using_utils_fmt_args_list")
   :end-before:  END_EXAMPLE("using_utils_fmt_args_list")

conduit::utils::format(string, maps, map_index)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``maps`` case also supports named or ordered args and works in conjunction with a ``map_index``. The ``map_index`` is used to fetch a value from an array, or list of strings, which is then passed to fmt. The ``maps`` style of indexed indirection supports generating path strings for non-trivial domain partition mappings in Blueprint.

**conduit::utils::format(string, maps, map_index) -- object case**: 

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_utils.cpp
   :start-after: BEGIN_EXAMPLE("using_utils_fmt_maps_obj")
   :end-before:  END_EXAMPLE("using_utils_fmt_maps_obj")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_utils_out.txt
   :start-after: BEGIN_EXAMPLE("using_utils_fmt_maps_obj")
   :end-before:  END_EXAMPLE("using_utils_fmt_maps_obj")

**conduit::utils::format(string, maps, map_index ) -- list case**: 


.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_utils.cpp
   :start-after: BEGIN_EXAMPLE("using_utils_fmt_maps_list")
   :end-before:  END_EXAMPLE("using_utils_fmt_maps_list")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_utils_out.txt
   :start-after: BEGIN_EXAMPLE("using_utils_fmt_maps_list")
   :end-before:  END_EXAMPLE("using_utils_fmt_maps_list")


