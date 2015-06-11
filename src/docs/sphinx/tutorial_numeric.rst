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

=======================
Accessing Numeric Data
=======================


Accessing Scalars and Arrays
--------------------------------

You can access leaf types (numeric scalars or arrays) using Node's *as_{type}* methods.

.. # from conduit_tutorial_examples: numeric_as_dtype

.. literalinclude:: ../../tests/conduit/conduit_tutorial_examples.cpp
   :lines: 159-162
   :language: cpp
   :dedent: 4

.. literalinclude:: tutorial_examples_out.txt
   :lines: 118

Or you can use Node::value(), which can infer the correct return type via a cast.

.. # from conduit_tutorial_examples: numeric_via_value

.. literalinclude:: ../../tests/conduit/conduit_tutorial_examples.cpp
   :lines: 171-177
   :language: cpp
   :dedent: 4

.. literalinclude:: tutorial_examples_out.txt
   :lines: 125


Accessing array data via pointers works the same way, using Node's *as_{type}* methods.

.. # from conduit_tutorial_examples: numeric_ptr_as_dtype

.. literalinclude:: ../../tests/conduit/conduit_tutorial_examples.cpp
   :lines: 188-199
   :language: cpp
   :dedent: 4

.. literalinclude:: tutorial_examples_out.txt
   :lines: 132-135


Or using Node::value():

.. # from conduit_tutorial_examples: numeric_ptr_via_value

.. literalinclude:: ../../tests/conduit/conduit_tutorial_examples.cpp
   :lines: 209-220
   :language: cpp
   :dedent: 4

.. literalinclude:: tutorial_examples_out.txt
   :lines: 142-145



For non-contiguous arrays, direct pointer access is complex due to the indexing required. Conduit provides a simple DataArray class that handles per-element indexing for all types of arrays.

.. # from conduit_tutorial_examples: numeric_strided_data_array

.. literalinclude:: ../../tests/conduit/conduit_tutorial_examples.cpp
   :lines: 229-245
   :language: cpp
   :dedent: 4

.. literalinclude:: tutorial_examples_out.txt
   :lines: 152-154



Using Introspection and Conversion
-----------------------------------

In this example, we have an array in a node that we are interested in processing using an and existing function that only handles doubles. We ensure the node is compatible with the function, or transform it 
to a contiguous double array. 


.. # from conduit_tutorial_examples: numeric_double_conversion

.. literalinclude:: ../../tests/conduit/conduit_tutorial_examples.cpp
   :lines: 255-313
   :language: cpp

.. literalinclude:: tutorial_examples_out.txt
   :lines: 163-176

