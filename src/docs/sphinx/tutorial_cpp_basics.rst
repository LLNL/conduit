.. ############################################################################
.. # Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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

======================
Basic Concepts
======================

*Node* basics
----------------

The *Node* class is the primary object in conduit.

Think of it as a hierarchical variant object.

.. # from t_conduit_docs_tutorial_basics: basics_very_basic

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :lines: 65-68
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :lines: 9-11

.. # from t_conduit_docs_tutorial_basics: basics_hierarchial

The *Node* class supports hierarchical construction.

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :lines: 76-84
   :language: cpp
   :dedent: 4


.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :lines: 19-30

Borrowing form JSON (and other similar notations), collections of named nodes are
called *Objects* and collections of unnamed nodes are called *Lists*, all other types
are leaves that represent concrete data. 

.. # from t_conduit_docs_tutorial_basics: basics_object_and_list

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :lines: 92-105
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :lines: 38-53

You can use a *NodeIterator* ( or a *NodeConstIterator*) to iterate through a Node's 
children.

.. # from t_conduit_docs_tutorial_basics: basics_object_and_list_itr

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :lines: 113-142
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :lines: 61-84


Behind the scenes, *Node* instances manage a collection of memory spaces.

.. # from t_conduit_docs_tutorial_basics: basics_mem_spaces

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :lines: 151-160
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :lines: 92-118

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
   :lines: 168-174
   :language: cpp
   :dedent: 4


.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :lines: 126-132

Standard C++ numeric types will be mapped by the compiler to bitwidth style types.

.. # from t_conduit_docs_tutorial_basics: basics_bw_style_from_native

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_basics.cpp
   :lines: 182-187
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_basics_out.txt
   :lines: 140-142


Supported Bitwidth Style Types:
 - signed integers: int8,int16,int32,int64
 - unsigned integers: uint8,uint16,uint32,uint64
 - floating point numbers: float32,float64

Conduit provides these types by constructing a mapping for the current platform the from the following types:
 - char, short, int, long, long long, float, double, long double


Compatible Schemas
--------------------------------

When a **set** method is called on a Node, if the data passed to the **set** is compatible with the Node's Schema the data is simply copied. No allocation or Schema changes occur. If the data is not compatible the Node will be reconfigured to store the passed data.

**Schemas do not need to be identical to be compatible.**

You can check if a Schema is compatible with another Schema using the **Schema::compatible(Schema &test)** method. Here is the criteria for checking if two Schemas are compatible:  

 - **If the calling Schema describes an Object** : The passed test Schema must describe an Object and the test Schema's children must be compatible with the calling Schema's children that have the same name.

 - **If the calling Schema describes a List**: The passed test Schema must describe a List, the calling Schema must have at least as many children as the test Schema, and when compared in list order each of the test Schema's children must be compatible with the calling Schema's children.

 - **If the calling Schema describes a leaf data type**: The calling Schema's and test Schema's **dtype().id()** and **dtype().element_bytes()** must match, and the calling Schema **dtype().number_of_elements()** must be greater than or equal than the test Schema's.






