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

=======================
Accessing Numeric Data
=======================


Accessing Scalars and Arrays
--------------------------------

You can access leaf types (numeric scalars or arrays) using Node's *as_{type}*
methods.

.. # from t_conduit_docs_tutorial_numeric: numeric_as_dtype

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :lines: 64-69
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :lines: 8

Or you can use Node::value(), which can infer the correct return type via a
cast.

.. # from t_conduit_docs_tutorial_numeric: numeric_via_value

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :lines: 77-83
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :lines: 15


Accessing array data via pointers works the same way, using Node's *as_{type}*
methods.

.. # from t_conduit_docs_tutorial_numeric: numeric_ptr_as_dtype

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :lines: 92-103
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :lines: 22-25


Or using Node::value():

.. # from t_conduit_docs_tutorial_numeric: numeric_ptr_via_value

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :lines: 112-124
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :lines: 32-35



For non-contiguous arrays, direct pointer access is complex due to the indexing
required. Conduit provides a simple DataArray class that handles per-element 
indexing for all types of arrays.

.. # from t_conduit_docs_tutorial_numeric: numeric_strided_data_array

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :lines: 132-148
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :lines: 42-44



Using Introspection and Conversion
-----------------------------------

In this example, we have an array in a node that we are interested in
processing using an and existing function that only handles doubles. We ensure
the node is compatible with the function, or transform it to a contiguous
double array. 


.. # from t_conduit_docs_tutorial_numeric: numeric_double_conversion

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_numeric.cpp
   :lines: 157-218
   :language: cpp

.. literalinclude:: t_conduit_docs_tutorial_numeric_out.txt
   :lines: 53-66

