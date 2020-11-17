.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

=======================
Accessing Numeric Data
=======================


Accessing Scalars and Arrays
--------------------------------

You can access leaf types (numeric scalars or arrays) using Node's *as_{type}*
methods.

.. # from t_conduit_docs_tutorial_numeric: numeric_as_dtype

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :start-after: BEGIN_EXAMPLE("numeric_as_dtype")
   :end-before:  END_EXAMPLE("numeric_as_dtype")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :start-after: BEGIN_EXAMPLE("numeric_as_dtype")
   :end-before:  END_EXAMPLE("numeric_as_dtype")
   
Or you can use Node::value(), which can infer the correct return type via a
cast.

.. # from t_conduit_docs_tutorial_numeric: numeric_via_value

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :start-after: BEGIN_EXAMPLE("numeric_via_value")
   :end-before:  END_EXAMPLE("numeric_via_value")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :start-after: BEGIN_EXAMPLE("numeric_via_value")
   :end-before:  END_EXAMPLE("numeric_via_value")


Accessing array data via pointers works the same way, using Node's *as_{type}*
methods.

.. # from t_conduit_docs_tutorial_numeric: numeric_ptr_as_dtype

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :start-after: BEGIN_EXAMPLE("numeric_ptr_as_dtype")
   :end-before:  END_EXAMPLE("numeric_ptr_as_dtype")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :start-after: BEGIN_EXAMPLE("numeric_ptr_as_dtype")
   :end-before:  END_EXAMPLE("numeric_ptr_as_dtype")


Or using Node::value():

.. # from t_conduit_docs_tutorial_numeric: numeric_ptr_via_value

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :start-after: BEGIN_EXAMPLE("numeric_ptr_via_value")
   :end-before:  END_EXAMPLE("numeric_ptr_via_value")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :start-after: BEGIN_EXAMPLE("numeric_ptr_via_value")
   :end-before:  END_EXAMPLE("numeric_ptr_via_value")



For non-contiguous arrays, direct pointer access is complex due to the indexing
required. Conduit provides a simple DataArray class that handles per-element 
indexing for all types of arrays.

.. # from t_conduit_docs_tutorial_numeric: numeric_strided_data_array

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :start-after: BEGIN_EXAMPLE("numeric_strided_data_array")
   :end-before:  END_EXAMPLE("numeric_strided_data_array")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :start-after: BEGIN_EXAMPLE("numeric_strided_data_array")
   :end-before:  END_EXAMPLE("numeric_strided_data_array")

C++11 Initializer Lists 
-----------------------------------

When C++11 support is enabled you can set Node values using initializer lists
with numeric literals. 

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :start-after: BEGIN_EXAMPLE("numeric_cxx11_init")
   :end-before:  END_EXAMPLE("numeric_cxx11_init")
   :language: cpp
   :dedent: 4


.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :start-after: BEGIN_EXAMPLE("numeric_cxx11_init")
   :end-before:  END_EXAMPLE("numeric_cxx11_init")


Using Introspection and Conversion
-----------------------------------

In this example, we have an array in a node that we are interested in
processing using an existing function that only handles doubles. We ensure
the node is compatible with the function, or transform it to a contiguous
double array. 


.. # from t_conduit_docs_tutorial_numeric: numeric_double_conversion

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :start-after: _conduit_tutorial_cpp_numeric_introspection_start
   :end-before:  _conduit_tutorial_cpp_numeric_introspection_end
   :language: cpp


.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :start-after: BEGIN_EXAMPLE("numeric_double_conversion")
   :end-before:  END_EXAMPLE("numeric_double_conversion")

