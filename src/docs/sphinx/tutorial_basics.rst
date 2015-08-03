.. ############################################################################
.. # Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
.. # 
.. # Produced at the Lawrence Livermore National Laboratory
.. # 
.. # LLNL-CODE-666778
.. # 
.. # All rights reserved.
.. # 
.. # This file is part of Conduit. 
.. # 
.. # For details, see https://lc.llnl.gov/conduit/.
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

.. # from conduit_tutorial_examples: basics_very_basic

.. literalinclude:: ../../tests/docs/conduit_tutorial_examples.cpp
   :lines: 63-66
   :language: cpp
   :dedent: 4

.. literalinclude:: tutorial_examples_out.txt
   :lines: 9-11

.. # from conduit_tutorial_examples: basics_hierarchial

The *Node* class supports hierarchical construction.

.. literalinclude:: ../../tests/docs/conduit_tutorial_examples.cpp
   :lines: 74-82
   :language: cpp
   :dedent: 4


.. literalinclude:: tutorial_examples_out.txt
   :lines: 19-30

Borrowing form JSON (and other similar notations), collections of named nodes are
called *Objects* and collections of unnamed nodes are called *Lists*, all other types
are leafs that represent concrete data. 

.. # from conduit_tutorial_examples: basics_object_and_list

.. literalinclude:: ../../tests/docs/conduit_tutorial_examples.cpp
   :lines: 90-103
   :language: cpp
   :dedent: 4

.. literalinclude:: tutorial_examples_out.txt
   :lines: 38-53

Behind the scenes, *Node* instances manage a collection of memory spaces.

.. # from conduit_tutorial_examples: basics_mem_spaces

.. literalinclude:: ../../tests/docs/conduit_tutorial_examples.cpp
   :lines: 112-120
   :language: cpp
   :dedent: 4

.. literalinclude:: tutorial_examples_out.txt
   :lines: 61-87

Bitwidth Style Types
--------------------------------

When sharing data in scientific codes, knowing the precision of the underlining types is very important.

Conduit uses well defined bitwidth style types (inspired by NumPy) for leaf values.

.. # from conduit_tutorial_examples: basics_bw_style

.. literalinclude:: ../../tests/docs/conduit_tutorial_examples.cpp
   :lines: 128-134
   :language: cpp
   :dedent: 4


.. literalinclude:: tutorial_examples_out.txt
   :lines: 95-97

Standard C++ numeric types will be mapped by the compiler to bitwidth style types.

.. # from conduit_tutorial_examples: basics_bw_style_from_native

.. literalinclude:: ../../tests/docs/conduit_tutorial_examples.cpp
   :lines: 143-147
   :language: cpp
   :dedent: 4

.. literalinclude:: tutorial_examples_out.txt
   :lines: 109-111


Supported Bitwidth Style Types:
 - signed integers: int8,int16,int32,int64
 - unsigned integers: uint8,uint16,uint32,uint64
 - floating point numbers: float32,float64

Conduit provides these types by constructing a mapping for the current platform the from the following types:
 - char, short, int, long, long long, float, double, long double


