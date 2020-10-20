.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

==================================
Reading YAML and JSON Strings
==================================

Parsing text with *Node::parse()*
------------------------------------------------------

*Node.parse()* parses YAML and JSON strings into a *Node* tree.

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
   :start-after: BEGIN_EXAMPLE("py_conduit_docs_tutorial_yaml")
   :end-before: END_EXAMPLE("py_conduit_docs_tutorial_yaml")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_parse_out.txt
   :start-after: BEGIN_EXAMPLE("py_conduit_docs_tutorial_yaml")
   :end-before: END_EXAMPLE("py_conduit_docs_tutorial_yaml")
   :dedent: 4

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
   :start-after: BEGIN_EXAMPLE("py_conduit_docs_tutorial_json")
   :end-before: END_EXAMPLE("py_conduit_docs_tutorial_json")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_parse_out.txt
   :start-after: BEGIN_EXAMPLE("py_conduit_docs_tutorial_json")
   :end-before: END_EXAMPLE("py_conduit_docs_tutorial_json")
   :dedent: 4

The first argument is the string to parse and the second argument
selects the protocol to use when parsing.

Valid Protocols:    ``json``, ``conduit_json``, ``conduit_base64_json``, ``yaml``.

* ``json`` and ``yaml`` protocols parse pure JSON or YAML strings. For leaf 
  nodes wide types such as *int64*, *uint64*, and *float64* are inferred.


Homogeneous numeric lists are parsed as Conduit arrays.

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
   :start-after: BEGIN_EXAMPLE("py_conduit_docs_tutorial_yaml_inline_array")
   :end-before: END_EXAMPLE("py_conduit_docs_tutorial_yaml_inline_array")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_parse_out.txt
   :start-after: BEGIN_EXAMPLE("py_conduit_docs_tutorial_yaml_inline_array")
   :end-before: END_EXAMPLE("py_conduit_docs_tutorial_yaml_inline_array")
   :dedent: 4

* ``conduit_json`` parses JSON with conduit data type information, allowing you
  to specify bitwidth style types, strides, etc. 

* ``conduit_base64_json`` combines the *conduit_json* protocol with an embedded 
  base64-encoded data block

======================
Generators 
======================

Using *Generator* instances
---------------------------------------------------
Node.parse() is sufficient for most use cases, but you can also use a *Generator* 
instance to parse JSON and YAML. Additionally, Generators can parse a 
conduit JSON schema and bind it to in-core data.

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
   :start-after: BEGIN_EXAMPLE("py_json_generator_std")
   :end-before: END_EXAMPLE("py_json_generator_std")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_parse_out.txt
   :start-after: BEGIN_EXAMPLE("py_json_generator_std")
   :end-before: END_EXAMPLE("py_json_generator_std")
   :dedent: 4

Like *Node::parse()*, *Generators* can also parse pure JSON or YAML.
For leaf nodes: wide types such as *int64*, *uint64*, and *float64* are inferred.


.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
   :start-after: BEGIN_EXAMPLE("py_json_generator_pure_json")
   :end-before: END_EXAMPLE("py_json_generator_pure_json")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_parse_out.txt
   :start-after: BEGIN_EXAMPLE("py_json_generator_pure_json")
   :end-before: END_EXAMPLE("py_json_generator_pure_json")
   :dedent: 4

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
   :start-after: BEGIN_EXAMPLE("py_json_generator_pure_yaml")
   :end-before: END_EXAMPLE("py_json_generator_pure_yaml")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_python_parse_out.txt
   :start-after: BEGIN_EXAMPLE("py_json_generator_pure_yaml")
   :end-before: END_EXAMPLE("py_json_generator_pure_yaml")
   :dedent: 4

.. ======================
.. Generators
.. ======================
..
..
.. Using *Generator* instances to parse JSON schemas
.. ---------------------------------------------------
..
.. The *Generator* class is used to parse conduit JSON schemas into a *Node*.
..
..
.. .. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
..    :start-after: BEGIN_EXAMPLE("py_json_generator_std")
..    :end-before: END_EXAMPLE("py_json_generator_std")
..    :language: python
..    :dedent: 8
..
.. .. literalinclude:: t_conduit_docs_python_tutorial_parse_out.txt
..    :start-after: BEGIN_EXAMPLE("py_json_generator_std")
..    :end-before: END_EXAMPLE("py_json_generator_std")
..    :dedent: 4
..
..
.. The *Generator* can also parse pure json. For leaf nodes: wide types such as *int64*, *uint64*, and *float64* are inferred.
..
.. .. # from t_conduit_docs_tutorial_python_json: json_generator_pure_json
..
.. .. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
..    :start-after: BEGIN_EXAMPLE("py_json_generator_pure_json")
..    :end-before: END_EXAMPLE("py_json_generator_pure_json")
..    :language: python
..    :dedent: 8
..
.. .. literalinclude:: t_conduit_docs_python_tutorial_parse_out.txt
..    :start-after: BEGIN_EXAMPLE("py_json_generator_pure_json")
..    :end-before: END_EXAMPLE("py_json_generator_pure_json")
..    :dedent: 4


.. # TODO, analog to C++ Compacting Nodes
