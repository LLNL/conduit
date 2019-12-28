.. ############################################################################
.. # Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
.. # 
.. # Produced at the Lawrence Livermore National Laboratory
.. # 
.. # LLNL-CODE-666778
.. # 
.. # All rights reserved.
.. # 
.. # This file is part of Conduit. 
.. # 
.. # For details, see: http://software.llnl.gov/conduit/.
.. # 
.. # Please also read conduit/LICENSE
.. # 
.. # Redistribution and use in source and binary forms, with or without 
.. # modification, are permitted provided that the following conditions are met:
.. # 
.. # * Redistributions of source code must retain the above copyright notice, 
.. #   this list of conditions and the disclaimer below.
.. # 
.. # * Redistributions in binary form must reproduce the above copyright notice,
.. #   this list of conditions and the disclaimer (as noted below) in the
.. #   documentation and/or other materials provided with the distribution.
.. # 
.. # * Neither the name of the LLNS/LLNL nor the names of its contributors may
.. #   be used to endorse or promote products derived from this software without
.. #   specific prior written permission.
.. # 
.. # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
.. # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
.. # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
.. # ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
.. # LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
.. # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
.. # DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
.. # OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
.. # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
.. # STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
.. # IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
.. # POSSIBILITY OF SUCH DAMAGE.
.. # 
.. ############################################################################

==================================
Reading YAML and JSON Strings
==================================

Parsing text with *Node::parse()*
------------------------------------------------------

*Node.parse()* parses YAML and JSON strings into a *Node* tree.

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
   :start-after: BEGIN_EXAMPLE("t_py_conduit_docs_tutorial_yaml")
   :end-before: END_EXAMPLE("t_py_conduit_docs_tutorial_yaml")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_python_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_py_conduit_docs_tutorial_yaml")
   :end-before: END_EXAMPLE("t_py_conduit_docs_tutorial_yaml")
   :dedent: 4

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
   :start-after: BEGIN_EXAMPLE("t_py_conduit_docs_tutorial_json")
   :end-before: END_EXAMPLE("t_py_conduit_docs_tutorial_json")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_python_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_py_conduit_docs_tutorial_json")
   :end-before: END_EXAMPLE("t_py_conduit_docs_tutorial_json")
   :dedent: 4

The first argument is the string to parse and the second argument
selects the protocol to use when parsing.

Valid Protocols:    ``json``, ``conduit_json``, ``conduit_base64_json``, ``yaml``.

* ``json`` and ``yaml`` protocols parse pure JSON or YAML strings. For leaf 
  nodes wide types such as *int64*, *uint64*, and *float64* are inferred.


Homogeneous numeric lists are parsed as Conduit arrays.

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
   :start-after: BEGIN_EXAMPLE("t_py_conduit_docs_tutorial_yaml_inline_array")
   :end-before: END_EXAMPLE("t_py_conduit_docs_tutorial_yaml_inline_array")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_python_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_py_conduit_docs_tutorial_yaml_inline_array")
   :end-before: END_EXAMPLE("t_py_conduit_docs_tutorial_yaml_inline_array")
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
   :start-after: BEGIN_EXAMPLE("t_py_json_generator_std")
   :end-before: END_EXAMPLE("t_py_json_generator_std")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_python_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_py_json_generator_std")
   :end-before: END_EXAMPLE("t_py_json_generator_std")
   :dedent: 4

Like *Node::parse()*, *Generators* can also parse pure JSON or YAML.
For leaf nodes: wide types such as *int64*, *uint64*, and *float64* are inferred.


.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
   :start-after: BEGIN_EXAMPLE("t_py_json_generator_pure_json")
   :end-before: END_EXAMPLE("t_py_json_generator_pure_json")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_py_json_generator_pure_json")
   :end-before: END_EXAMPLE("t_py_json_generator_pure_json")
   :dedent: 4

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
   :start-after: BEGIN_EXAMPLE("t_py_json_generator_pure_yaml")
   :end-before: END_EXAMPLE("t_py_json_generator_pure_yaml")
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_python_tutorial_parse_out.txt
   :start-after: BEGIN_EXAMPLE("t_py_json_generator_pure_yaml")
   :end-before: END_EXAMPLE("t_py_json_generator_pure_yaml")
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
..    :start-after: BEGIN_EXAMPLE("t_py_json_generator_std")
..    :end-before: END_EXAMPLE("t_py_json_generator_std")
..    :language: python
..    :dedent: 8
..
.. .. literalinclude:: t_conduit_docs_python_tutorial_parse_out.txt
..    :start-after: BEGIN_EXAMPLE("t_py_json_generator_std")
..    :end-before: END_EXAMPLE("t_py_json_generator_std")
..    :dedent: 4
..
..
.. The *Generator* can also parse pure json. For leaf nodes: wide types such as *int64*, *uint64*, and *float64* are inferred.
..
.. .. # from t_conduit_docs_tutorial_python_json: json_generator_pure_json
..
.. .. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_parse.py
..    :start-after: BEGIN_EXAMPLE("t_py_json_generator_pure_json")
..    :end-before: END_EXAMPLE("t_py_json_generator_pure_json")
..    :language: python
..    :dedent: 8
..
.. .. literalinclude:: t_conduit_docs_python_tutorial_parse_out.txt
..    :start-after: BEGIN_EXAMPLE("t_py_json_generator_pure_json")
..    :end-before: END_EXAMPLE("t_py_json_generator_pure_json")
..    :dedent: 4


.. # TODO, analog to C++ Compacting Nodes
