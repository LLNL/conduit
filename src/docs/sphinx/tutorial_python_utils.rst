===========================
String Formatting Helpers
===========================


conduit.utils.format
-----------------------

String formatting in Python land has always been much more pleasant than in C++ land.
In C++, we bundle ``fmt``, but Python's out-of-the box support for string formatting
is fantastic. Since users may encode format string arguments in conduit Nodes
(and in HDF5, YAML, etc files) we still provide access the ``fmt`` based 
``conduit.utils.format`` functionality in Python. 


conduit.utils.format(string, args)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``args`` case allows named arguments (args passed as object) or ordered args (args passed as list).

**conduit.utils.format(string, args) -- object case**: 

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_utils.py
   :start-after: BEGIN_EXAMPLE("py_using_utils_fmt_args_obj")
   :end-before:  END_EXAMPLE("py_using_utils_fmt_args_obj")
   :language: cpp
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_utils_out.txt
   :start-after: BEGIN_EXAMPLE("py_using_utils_fmt_args_obj")
   :end-before:  END_EXAMPLE("py_using_utils_fmt_args_obj")
   :dedent: 4

**conduit.utils.format(string, args) -- list case**: 

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_utils.py
   :start-after: BEGIN_EXAMPLE("py_using_utils_fmt_args_list")
   :end-before:  END_EXAMPLE("py_using_utils_fmt_args_list")
   :language: cpp
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_utils_out.txt
   :start-after: BEGIN_EXAMPLE("py_using_utils_fmt_args_list")
   :end-before:  END_EXAMPLE("py_using_utils_fmt_args_list")
   :dedent: 4

conduit.utils.format(string, maps, map_index)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``maps`` case also supports named or ordered args and works in conjunction with a ``map_index``. The ``map_index`` is used to fetch a value from an array, or list of strings, which is then passed to fmt. The ``maps`` style of indexed indirection supports generating path strings for non-trivial domain partition mappings in Blueprint.

**conduit.utils.format(string, maps, map_index) -- object case**: 

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_utils.py
   :start-after: BEGIN_EXAMPLE("py_using_utils_fmt_maps_obj")
   :end-before:  END_EXAMPLE("py_using_utils_fmt_maps_obj")
   :language: cpp
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_utils_out.txt
   :start-after: BEGIN_EXAMPLE("py_using_utils_fmt_maps_obj")
   :end-before:  END_EXAMPLE("py_using_utils_fmt_maps_obj")
   :dedent: 4

**conduit.utils.format(string, maps, map_index ) -- list case**: 


.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_utils.py
   :start-after: BEGIN_EXAMPLE("py_using_utils_fmt_maps_list")
   :end-before:  END_EXAMPLE("py_using_utils_fmt_maps_list")
   :language: cpp
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_utils_out.txt
   :start-after: BEGIN_EXAMPLE("py_using_utils_fmt_maps_list")
   :end-before:  END_EXAMPLE("py_using_utils_fmt_maps_list")
   :dedent: 4

