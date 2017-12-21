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
Generators 
======================


Using *Generator* instances to parse JSON schemas
---------------------------------------------------

The *Generator* class is used to parse conduit JSON schemas into a *Node*.


.. # from t_conduit_docs_tutorial_python_json: json_generator_std

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_json.py
   :lines: 62-67
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_python_tutorial_json_out.txt
   :lines: 29-32
   :dedent: 4


The *Generator* can also parse pure json. For leaf nodes: wide types such as *int64*, *uint64*, and *float64* are inferred.

.. # from t_conduit_docs_tutorial_python_json: json_generator_pure_json

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_json.py
   :lines: 74-79
   :language: python
   :dedent: 8

.. literalinclude:: t_conduit_docs_python_tutorial_json_out.txt
   :lines: 37-42
   :dedent: 4


.. # TODO, analog to C++ Compacting Nodes
