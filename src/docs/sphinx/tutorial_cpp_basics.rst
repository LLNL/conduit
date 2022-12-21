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

.. # from t_conduit_docs_tutorial_basics: basics_very_basic

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :start-after: BEGIN_EXAMPLE("basics_very_basic")
   :end-before:  END_EXAMPLE("basics_very_basic")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :start-after: BEGIN_EXAMPLE("basics_very_basic")
   :end-before:  END_EXAMPLE("basics_very_basic")

.. # from t_conduit_docs_tutorial_basics: basics_hierarchial

The *Node* class supports hierarchical construction.

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :start-after: BEGIN_EXAMPLE("basics_hierarchial")
   :end-before:  END_EXAMPLE("basics_hierarchial")
   :language: cpp
   :dedent: 4


.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :start-after: BEGIN_EXAMPLE("basics_hierarchial")
   :end-before:  END_EXAMPLE("basics_hierarchial")

Borrowing form JSON (and other similar notations), collections of named nodes are
called *Objects* and collections of unnamed nodes are called *Lists*, all other types
are leaves that represent concrete data. 

.. # from t_conduit_docs_tutorial_basics: basics_object_and_list

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :start-after: BEGIN_EXAMPLE("basics_object_and_list")
   :end-before:  END_EXAMPLE("basics_object_and_list")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :start-after: BEGIN_EXAMPLE("basics_object_and_list")
   :end-before:  END_EXAMPLE("basics_object_and_list")

You can use a *NodeIterator* ( or a *NodeConstIterator*) to iterate through a Node's 
children.

.. # from t_conduit_docs_tutorial_basics: basics_object_and_list_itr

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :start-after: BEGIN_EXAMPLE("basics_object_and_list_itr")
   :end-before:  END_EXAMPLE("basics_object_and_list_itr")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :start-after: BEGIN_EXAMPLE("basics_object_and_list_itr")
   :end-before:  END_EXAMPLE("basics_object_and_list_itr")


Behind the scenes, *Node* instances manage a collection of memory spaces.

.. # from t_conduit_docs_tutorial_basics: basics_mem_spaces

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :start-after: BEGIN_EXAMPLE("basics_mem_spaces")
   :end-before:  END_EXAMPLE("basics_mem_spaces")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :start-after: BEGIN_EXAMPLE("basics_mem_spaces")
   :end-before:  END_EXAMPLE("basics_mem_spaces")

.. # we could add an example here

There is no absolute path construct, all paths are fetched relative to the current node (a leading 
``/`` is ignored when fetching). Empty paths names are also ignored, fetching ``a///b`` is 
equalvalent to fetching ``a/b``. 

.. # You can fetch a Node's parent using ``..``.


Bitwidth Style Types
--------------------------------

When sharing data in scientific codes, knowing the precision of the underlining types is very important.

Conduit uses well defined bitwidth style types (inspired by NumPy) for leaf values.

.. # from t_conduit_docs_tutorial_basics: basics_bw_style

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :start-after: BEGIN_EXAMPLE("basics_bw_style")
   :end-before:  END_EXAMPLE("basics_bw_style")
   :language: cpp
   :dedent: 4


.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :start-after: BEGIN_EXAMPLE("basics_bw_style")
   :end-before:  END_EXAMPLE("basics_bw_style")


Standard C++ numeric types will be mapped by the compiler to bitwidth style types.

.. # from t_conduit_docs_tutorial_basics: basics_bw_style_from_native

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :start-after: BEGIN_EXAMPLE("basics_bw_style_from_native")
   :end-before:  END_EXAMPLE("basics_bw_style_from_native")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :start-after: BEGIN_EXAMPLE("basics_bw_style_from_native")
   :end-before:  END_EXAMPLE("basics_bw_style_from_native")



Supported Bitwidth Style Types:
 - signed integers: int8,int16,int32,int64
 - unsigned integers: uint8,uint16,uint32,uint64
 - floating point numbers: float32,float64

Conduit provides these types by constructing a mapping for the current platform the from the following types:
 - char, short, int, long, long long, float, double, long double

When C++11 support is enabled, Conduit's bitwidth style types will match the C++11 standard bitwidth types defined in ``<cstdint>``.


When a **set** method is called on a leaf Node, if the data passed to the **set** is compatible with the Node's Schema the data is simply copied.

Compatible Schemas
--------------------------------

When passed a compatible Node, Node methods ``update`` and ``update_compatible``
allow you to copy data into Node or extend a Node with new data without
changing existing allocations. 

**Schemas do not need to be identical to be compatible.**

You can check if a Schema is compatible with another Schema using the **Schema::compatible(Schema &test)** method. Here is the criteria for checking if two Schemas are compatible:

 - **If the calling Schema describes an Object** : The passed test Schema must describe an Object and the test Schema's children must be compatible with the calling Schema's children that have the same name.

 - **If the calling Schema describes a List**: The passed test Schema must describe a List, the calling Schema must have at least as many children as the test Schema, and when compared in list order each of the test Schema's children must be compatible with the calling Schema's children.

 - **If the calling Schema describes a leaf data type**: The calling Schema's and test Schema's ``dtype().id()`` and ``dtype().element_bytes()`` must match, and the calling Schema ``dtype().number_of_elements()`` must be greater than or equal than the test Schema's.
 
Here is a C++ pseudocode example that shows the most common use of ``Node::compatible()``:

.. code:: cpp

    conduit::Node a,b;

    // In this example:
    //    the calling schema is `a.schema()`
    //    the test schema is `b.schema()`

    // ask if `a` can already hold data described by `b`
    if(a.compatible(b))
    {
      // data from `b` can be written to `a` without a new allocation
      // ...
    }


Node References
--------------------------------

For most uses cases in C++, Node references are the best solution to build and
manipulate trees. They allow you to avoid expensive copies and pass around
sub-trees without worrying about valid pointers.


.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :start-after: BEGIN_EXAMPLE("basics_node_refs")
   :end-before:  END_EXAMPLE("basics_node_refs")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :start-after: BEGIN_EXAMPLE("basics_node_refs")
   :end-before:  END_EXAMPLE("basics_node_refs")


In C++ the Node assignment operator that takes a Node input is really an alias
to set. That is, if follows set (deep copy) semantics.

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :start-after: BEGIN_EXAMPLE("basics_node_refs_bad")
   :end-before:  END_EXAMPLE("basics_node_refs_bad")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :start-after: BEGIN_EXAMPLE("basics_node_refs_bad")
   :end-before:  END_EXAMPLE("basics_node_refs_bad")


const Nodes
--------------------------------

If you aren't careful, the ability to easily create dynamic trees can
also undermine your process to consume them. 
For example, asking for an expected but non-existent path will return
a reference to an empty Node. Surprise!

Methods like `fetch_existing` allow you to be more explicit
when asking for expected data. In C++, ``const Node`` references are also common
way to process trees in an read-only fashion. ``const`` methods will not modify
the tree structure, so if you ask for a non-existent path, you will receive
an error instead of reference to an empty Node.


.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :start-after: BEGIN_BLOCK("basics_const_example")
   :end-before:  END_BLOCK("basics_const_example")
   :language: cpp

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :start-after: BEGIN_EXAMPLE("basics_const_vs_non_const")
   :end-before:  END_EXAMPLE("basics_const_vs_non_const")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :start-after: BEGIN_EXAMPLE("basics_const_vs_non_const")
   :end-before:  END_EXAMPLE("basics_const_vs_non_const")