.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

==================================
Reading YAML and JSON Strings
==================================

Parsing text with *Node::parse()*
------------------------------------------------------

*Node::parse()* parses YAML and JSON strings into a *Node* tree.

.. # from t_conduit_docs_tutorial_parse: t_conduit_docs_tutorial_yaml

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_parse.cpp
   :start-after: BEGIN_EXAMPLE("t_conduit_docs_tutorial_yaml")
   :end-before: END_EXAMPLE("t_conduit_docs_tutorial_yaml")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_conduit_docs_tutorial_yaml")
   :end-before: END_EXAMPLE("t_conduit_docs_tutorial_yaml")

.. # from t_conduit_docs_tutorial_parse: t_conduit_docs_tutorial_json

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_parse.cpp
   :start-after: BEGIN_EXAMPLE("t_conduit_docs_tutorial_json")
   :end-before: END_EXAMPLE("t_conduit_docs_tutorial_json")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_conduit_docs_tutorial_json")
   :end-before: END_EXAMPLE("t_conduit_docs_tutorial_json")

The first argument is the string to parse and the second argument
selects the protocol to use when parsing.

Valid Protocols:    ``json``, ``conduit_json``, ``conduit_base64_json``, ``yaml``.

* ``json`` and ``yaml`` protocols parse pure JSON or YAML strings. For leaf 
  nodes wide types such as *int64*, *uint64*, and *float64* are inferred.


Homogeneous numeric lists are parsed as Conduit arrays.

.. # from t_conduit_docs_tutorial_parse: t_conduit_docs_tutorial_yaml_inline_array

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_parse.cpp
   :start-after: BEGIN_EXAMPLE("t_conduit_docs_tutorial_yaml_inline_array")
   :end-before: END_EXAMPLE("t_conduit_docs_tutorial_yaml_inline_array")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_conduit_docs_tutorial_yaml_inline_array")
   :end-before: END_EXAMPLE("t_conduit_docs_tutorial_yaml_inline_array")

* ``conduit_json`` parses JSON with conduit data type information, allowing you
  to specify bitwidth style types, strides, etc. 

* ``conduit_base64_json`` combines the *conduit_json* protocol with an embedded 
  base64-encoded data block

======================
Generators 
======================

Using *Generator* instances
---------------------------------------------------
Node::parse() is sufficient for most use cases, but you can also use a *Generator* 
instance to parse JSON and YAML. Additionally, Generators can parse a 
conduit JSON schema and bind it to in-core data.

.. # from t_conduit_docs_tutorial_json: json_generator_std

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_parse.cpp
   :start-after: BEGIN_EXAMPLE("t_json_generator_std")
   :end-before: END_EXAMPLE("t_json_generator_std")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_json_generator_std")
   :end-before: END_EXAMPLE("t_json_generator_std")


Like *Node::parse()*, *Generators* can also parse pure JSON or YAML.
For leaf nodes: wide types such as *int64*, *uint64*, and *float64* are inferred.

.. # from t_conduit_docs_tutorial_json: json_generator_pure_json

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_parse.cpp
   :start-after: BEGIN_EXAMPLE("t_json_generator_pure_json")
   :end-before: END_EXAMPLE("t_json_generator_pure_json")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_json_generator_pure_json")
   :end-before: END_EXAMPLE("t_json_generator_pure_json")

.. # from t_conduit_docs_tutorial_json: json_generator_pure_yaml

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_parse.cpp
   :start-after: BEGIN_EXAMPLE("t_json_generator_pure_yaml")
   :end-before: END_EXAMPLE("t_json_generator_pure_yaml")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_json_generator_pure_yaml")
   :end-before: END_EXAMPLE("t_json_generator_pure_yaml")



Schemas can be bound to in-core data.

.. # from t_conduit_docs_tutorial_json: json_generator_bind_to_incore

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_parse.cpp
   :start-after: BEGIN_EXAMPLE("t_json_generator_bind_to_incore")
   :end-before: END_EXAMPLE("t_json_generator_bind_to_incore")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_json_generator_bind_to_incore")
   :end-before: END_EXAMPLE("t_json_generator_bind_to_incore")



Compacting Nodes
--------------------------------

*Nodes* can be compacted to transform sparse data.

.. # from t_conduit_docs_tutorial_json: json_generator_compact

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_parse.cpp
   :start-after: BEGIN_EXAMPLE("t_json_generator_compact")
   :end-before: END_EXAMPLE("t_json_generator_compact")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_json_generator_compact")
   :end-before: END_EXAMPLE("t_json_generator_compact")


