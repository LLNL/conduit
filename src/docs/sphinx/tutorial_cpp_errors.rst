.. ############################################################################
.. # Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
Error Handling 
======================

Conduit's APIs emit three types of messages for logging and error handling:

================ ===================================================
Message Type      Description
================ ===================================================
**Info**          General Information
**Warning**       Recoverable Error
**Error**         Fatal Error
================ ===================================================


Default Error Handlers
---------------------------------------------------

Conduit provides a default handler for each message type: 

================ ===================================================
Message Type      Default Action
================ ===================================================
**Info**          Prints the message to standard out
**Warning**       Throws a C++ Exception (conduit::Error instance)
**Error**         Throws a C++ Exception (conduit::Error instance)
================ ===================================================

Using Custom Error Handlers
--------------------------------

The conduit::utils namespace provides functions to override each of the three default handlers with a method
that provides the following signature: 

.. code-block:: cpp

  void my_handler(const std::string &msg,
                  const std::string &file,
                  int line)
  {
    // your handling code here ...
  }

  conduit::utils::set_error_handler(my_handler);

Here is an example that re-wires all three error handlers to print to standard out:

.. # from t_conduit_docs_tutorial_errors: custom range

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_errors.cpp
   :lines:  65-90
   :language: cpp

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_errors.cpp
   :lines:  103-130
   :language: cpp

.. literalinclude:: t_conduit_docs_tutorial_errors_out.txt
   :lines: 16-18



Using Restoring Default Handlers
--------------------------------

The default handlers are part of the conduit::utils interface, so you can restore them using:


.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_errors.cpp
   :lines: 132-135
   :language: cpp
   :dedent: 4

