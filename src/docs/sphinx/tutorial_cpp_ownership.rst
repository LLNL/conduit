.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

============================================
Data Ownership
============================================

The *Node* class provides two ways to hold data, the data is either **owned** or **externally described**:

- If a *Node* **owns** data, the *Node* allocated the memory holding the data and is responsible or deallocating it. 
-  If a *Node* **externally describes** data, the *Node* holds a pointer to the memory where the data resides and is not responsible for deallocating it. 

*set* vs *set_external* 
--------------------------------

The **Node::set** methods support creating **owned** data and copying data values in both the **owned** and **externally described** cases. The **Node::set_external** methods allow you to create **externally described** data:

- **set(...)**: Makes a copy of the data passed into the *Node*. This will trigger an allocation if the current data type of the *Node* is incompatible with what was passed. The *Node* assignment operators use their respective **set** variants, so they follow the same copy semantics. 

- **set_external(...)**: Sets up the *Node* to describe data passed and access the data externally. Does not copy the data.

.. # from t_conduit_docs_tutorial_ownership: mem_ownership_external

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_ownership.cpp
   :start-after: BEGIN_EXAMPLE("mem_ownership_external")
   :end-before: END_EXAMPLE("mem_ownership_external")
   :language: cpp
   :dedent: 4

.. literalinclude:: t_conduit_docs_tutorial_ownership_out.txt
   :start-after: BEGIN_EXAMPLE("mem_ownership_external")
   :end-before: END_EXAMPLE("mem_ownership_external")



