.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

============================================
Node Move and Swap
============================================

The *Node* class provides *move* and *swap* methods that allow you to efficiently combine or exchange data between Node trees. These provide similar semantics to C++ *std::move* and *std::swap*.


- **Node::move(Node &n)**: Allows you to move the entire contents and description of a passed Node into the calling Node. There are no copies. The calling Node assumes management of all pointers and relevant metadata.

- **Node::swap(Node &n)**: Allows you to swap the entire contents and description of two Nodes. There are no copies. The Node's swap management of all pointers and relevant metadata. The schema parent pointers are updated to reflect the new hierarchy.

.. # from t_conduit_docs_tutorial_move_and_swap: move

**Move Example Source:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_move_and_swap.cpp
   :start-after: BEGIN_EXAMPLE("move")
   :end-before: END_EXAMPLE("move")
   :language: cpp
   :dedent: 4

**Move Example Output:**

.. literalinclude:: t_conduit_docs_tutorial_move_and_swap_out.txt
   :start-after: BEGIN_EXAMPLE("move")
   :end-before: END_EXAMPLE("move")


**Swap Example Source:**

.. # from t_conduit_docs_tutorial_move_and_swap: swap

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_move_and_swap.cpp
   :start-after: BEGIN_EXAMPLE("swap")
   :end-before: END_EXAMPLE("swap")
   :language: cpp
   :dedent: 4


**Swap Example Output:**

.. literalinclude:: t_conduit_docs_tutorial_move_and_swap_out.txt
   :start-after: BEGIN_EXAMPLE("swap")
   :end-before: END_EXAMPLE("swap")