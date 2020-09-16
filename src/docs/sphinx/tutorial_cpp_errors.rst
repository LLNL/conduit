.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

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
   :start-after: _conduit_error_handlers_funcs_start
   :end-before:  _conduit_error_handlers_funcs_end
   :language: cpp

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_errors.cpp
   :start-after: BEGIN_EXAMPLE("error_handlers_rewire")
   :end-before:  END_EXAMPLE("error_handlers_rewire")
   :language: cpp
   :dedent: 4


.. literalinclude:: t_conduit_docs_tutorial_errors_out.txt
   :start-after: BEGIN_EXAMPLE("error_handlers_rewire")
   :end-before:  END_EXAMPLE("error_handlers_rewire")


Using Restoring Default Handlers
--------------------------------

The default handlers are part of the conduit::utils interface, so you can restore them using:


.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_errors.cpp
   :start-after: BEGIN_EXAMPLE("error_handlers_reset")
   :end-before:  END_EXAMPLE("error_handlers_reset")
   :language: cpp
   :dedent: 4

