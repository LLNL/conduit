.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

======================
Basic Concepts
======================

*Node* basics
----------------

The *Node* class is the primary object in conduit.

Think of it as a hierarchical variant object.

.. # from t_conduit_docs_tutorial_python_basics: basics_very_basic

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_basics.py
   :start-after: BEGIN_EXAMPLE("py_basics_very_basic")
   :end-before: END_EXAMPLE("py_basics_very_basic")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_basics_out.txt 
   :start-after: BEGIN_EXAMPLE("py_basics_very_basic")
   :end-before: END_EXAMPLE("py_basics_very_basic")
   :dedent: 4

.. # from t_conduit_docs_tutorial_python_basics: basics_hierarchial

The *Node* class supports hierarchical construction.

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_basics.py
   :start-after: BEGIN_EXAMPLE("py_basics_hierarchial")
   :end-before: END_EXAMPLE("py_basics_hierarchial")
   :language: python
   :dedent: 8


.. literalinclude:: t_conduit_docs_tutorial_python_basics_out.txt
   :start-after: BEGIN_EXAMPLE("py_basics_hierarchial")
   :end-before: END_EXAMPLE("py_basics_hierarchial")
   :dedent: 4

Borrowing from JSON (and other similar notations), collections of named nodes are
called *Objects* and collections of unnamed nodes are called *Lists*, all other types
are leaves that represent concrete data. 

.. # from t_conduit_docs_tutorial_python_basics: basics_object_and_list

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_basics.py
   :start-after: BEGIN_EXAMPLE("py_basics_object_and_list")
   :end-before: END_EXAMPLE("py_basics_object_and_list")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_basics_out.txt
   :start-after: BEGIN_EXAMPLE("py_basics_object_and_list")
   :end-before: END_EXAMPLE("py_basics_object_and_list")
   :dedent: 4

You can iterate through a Node's children.

.. # from t_conduit_docs_tutorial_python_basics: basics_object_and_list_itr

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_basics.py
   :start-after: BEGIN_EXAMPLE("py_basics_object_and_list_itr")
   :end-before: END_EXAMPLE("py_basics_object_and_list_itr")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_basics_out.txt 
   :start-after: BEGIN_EXAMPLE("py_basics_object_and_list_itr")
   :end-before: END_EXAMPLE("py_basics_object_and_list_itr")
   :dedent: 4


Behind the scenes, *Node* instances manage a collection of memory spaces.

.. # from t_conduit_docs_tutorial_python_basics: basics_mem_spaces

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_basics.py
   :start-after: BEGIN_EXAMPLE("py_basics_mem_spaces")
   :end-before: END_EXAMPLE("py_basics_mem_spaces")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_basics_out.txt 
   :start-after: BEGIN_EXAMPLE("py_basics_mem_spaces")
   :end-before: END_EXAMPLE("py_basics_mem_spaces")
   :dedent: 4

.. # we could add an example here

There is no absolute path construct, all paths are fetched relative to the current node (a leading 
``/`` is ignored when fetching). Empty paths names are also ignored, fetching ``a///b`` is 
equalvalent to fetching ``a/b``. 

.. # You can fetch a Node's parent using ``..``.

Bitwidth Style Types
--------------------------------

When sharing data in scientific codes, knowing the precision of the underlining types is very important.

Conduit uses well defined bitwidth style types (inspired by NumPy) for leaf values. In Python, leaves
are provided as NumPy ndarrays. 

.. # from t_conduit_docs_tutorial_python_basics: basics_bw_style

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_basics.py
   :start-after: BEGIN_EXAMPLE("py_basics_bw_style")
   :end-before: END_EXAMPLE("py_basics_bw_style")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_basics_out.txt 
   :start-after: BEGIN_EXAMPLE("py_basics_bw_style")
   :end-before: END_EXAMPLE("py_basics_bw_style")
   :dedent: 4
   
Standard Python numeric types will be mapped to bitwidth style types.

.. # from t_conduit_docs_tutorial_python_basics: basics_bw_style_from_native

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_basics.py
   :start-after: BEGIN_EXAMPLE("py_basics_bw_style_from_native")
   :end-before: END_EXAMPLE("py_basics_bw_style_from_native")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_basics_out.txt 
   :start-after: BEGIN_EXAMPLE("py_basics_bw_style_from_native")
   :end-before: END_EXAMPLE("py_basics_bw_style_from_native")
   :dedent: 4

Supported Bitwidth Style Types:
 - signed integers: int8,int16,int32,int64
 - unsigned integers: uint8,uint16,uint32,uint64
 - floating point numbers: float32,float64

Conduit provides these types by constructing a mapping for the current platform the from the following C++ types:
 - char, short, int, long, long long, float, double, long double


Compatible Schemas
--------------------------------

When a **set** method is called on a Node, if the data passed to the **set** is compatible with the Node's Schema the data is simply copied. No allocation or Schema changes occur. If the data is not compatible the Node will be reconfigured to store the passed data.

**Schemas do not need to be identical to be compatible.**

You can check if a Schema is compatible with another Schema using the **Schema::compatible(Schema &test)** method. Here is the criteria for checking if two Schemas are compatible:  

 - **If the calling Schema describes an Object** : The passed test Schema must describe an Object and the test Schema's children must be compatible with the calling Schema's children that have the same name.

 - **If the calling Schema describes a List**: The passed test Schema must describe a List, the calling Schema must have at least as many children as the test Schema, and when compared in list order each of the test Schema's children must be compatible with the calling Schema's children.

 - **If the calling Schema describes a leaf data type**: The calling Schema's and test Schema's **dtype().id()** and **dtype().element_bytes()** must match, and the calling Schema **dtype().number_of_elements()** must be greater than or equal than the test Schema's.






