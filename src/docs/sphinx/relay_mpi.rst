.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

===================
Relay MPI
===================

The Conduit Relay MPI library enables MPI communication using conduit::Node instances as payloads. It provides two categories of functionality: :ref:`mpi_known_schema_methods` and :ref:`mpi_generic_methods`. These categories balance flexibility and performance tradeoffs. In all cases the implementation tries to avoid unnecessary reallocation, subject to the constraints of MPI's API input requirements.



.. _mpi_known_schema_methods:

Known Schema Methods
----------------------

Methods that transfer a Node's data, assuming the schema is known. They assume that Nodes used for output are implicitly **compatible** with their sources.

Supported MPI Primitives:
 * send/recv
 * isend/irecv
 * reduce/all_reduce
 * broadcast
 * gather/all_gather


For both point to point and collectives, here is the basic logic for how input Nodes are treated by these methods:

* For Nodes holding data to be sent:

 * If the Node is compact and contiguously allocated, the Node's pointers are passed directly to MPI.

 * If the Node is not compact or not contiguously allocated, the data is compacted to temporary contiguous buffers that are passed to MPI.

* For Nodes used to hold output data:

 * If the output Node is compact and contiguously allocated, the Node's pointers are passed directly to MPI.

 * If the output Node is not compact or not contiguously allocated, a Node with a temporary contiguous buffer is created and that buffer is passed to MPI. An **update** call is used to copy out the data from the temporary buffer to the output Node. This avoids re-allocation and modifying the schema of the output Node.

.. _mpi_generic_methods:

Generic Methods
---------------

Methods that transfer both a Node's data and schema. These are useful for generic messaging, since the schema does not need to be known by receiving tasks. The semantics of MPI place constraints on what can be supported in this category.

Supported MPI Primitives:
  * send/recv
  * gather/all_gather
  * broadcast


Unsupported MPI Primitives:
  * isend/irecv
  * reduce/all_reduce


For both point to point and collectives, here is the basic logic for how input Nodes are treated by these methods:

* For Nodes holding data to be sent:

 * If the Node is compact and contiguously allocated:
 
  * The Node's schema is sent as JSON
  * The Node's pointers are passed directly to MPI

 * If the Node is not compact or not contiguously allocated:
 
  * The Node is compacted to temporary Node
  * The temporary Node's schema is sent as JSON
  * The temporary Nodes's pointers are passed to MPI
 

* For Nodes used to hold output data:

 * If the output Node is not compatible with the received schema, it is reset using the received schema.

 * If the output Node is compact and contiguously allocated, its pointers are passed directly to MPI.

 * If the output Node is not compact or not contiguously allocated, a Node with a temporary contiguous buffer is created and that buffer is passed to MPI. An **update** call is used to copy out the data from the temporary buffer to the output Node. This avoids re-allocation and modifying the schema of the output Node.



Python Relay MPI Module
------------------------

Relay MPI is supported in Python via the conduit.relay.mpi module.
Methods take Fortran-style MPI communicator handles which are effectively integers.
(We hope to also support direct use of `mpi4py` communicator objects in the future.)

Use the following to get a handle from the `mpi4py` world communicator:
 
.. code-block:: python

    from mpi4py import MPI
    comm_id   = MPI.COMM_WORLD.py2f()


Python Relay MPI Module Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Send and Receive Using Schema
++++++++++++++++++++++++++++++

* **Python Source:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_relay_mpi_examples.py
   :start-after: BEGIN_EXAMPLE("py_mpi_send_and_recv_using_schema")
   :end-before: END_EXAMPLE("py_mpi_send_and_recv_using_schema")
   :language: python
   :dedent: 8

* **Output:**

.. literalinclude:: t_conduit_docs_tutorial_python_relay_mpi_examples_out.txt
   :start-after: BEGIN_EXAMPLE("py_mpi_send_and_recv_using_schema")
   :end-before: END_EXAMPLE("py_mpi_send_and_recv_using_schema")
   :dedent: 4
   

Send and Receive
++++++++++++++++++

* **Python Source:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_relay_mpi_examples.py
   :start-after: BEGIN_EXAMPLE("py_mpi_send_and_recv")
   :end-before: END_EXAMPLE("py_mpi_send_and_recv")
   :language: python
   :dedent: 8

* **Output:**

.. literalinclude:: t_conduit_docs_tutorial_python_relay_mpi_examples_out.txt
   :start-after: BEGIN_EXAMPLE("py_mpi_send_and_recv")
   :end-before: END_EXAMPLE("py_mpi_send_and_recv")
   :dedent: 4

Send and Receive
++++++++++++++++++

* **Python Source:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_relay_mpi_examples.py
   :start-after: BEGIN_EXAMPLE("py_mpi_send_and_recv")
   :end-before: END_EXAMPLE("py_mpi_send_and_recv")
   :language: python
   :dedent: 8

* **Output:**

.. literalinclude:: t_conduit_docs_tutorial_python_relay_mpi_examples_out.txt
   :start-after: BEGIN_EXAMPLE("py_mpi_send_and_recv")
   :end-before: END_EXAMPLE("py_mpi_send_and_recv")
   :dedent: 4

Sum All Reduce
++++++++++++++++++

* **Python Source:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_relay_mpi_examples.py
   :start-after: BEGIN_EXAMPLE("py_mpi_sum_all_reduce")
   :end-before: END_EXAMPLE("py_mpi_sum_all_reduce")
   :language: python
   :dedent: 8

* **Output:**

.. literalinclude:: t_conduit_docs_tutorial_python_relay_mpi_examples_out.txt
   :start-after: BEGIN_EXAMPLE("py_mpi_sum_all_reduce")
   :end-before: END_EXAMPLE("py_mpi_sum_all_reduce")
   :dedent: 4

Broadcast Using Schema
+++++++++++++++++++++++

* **Python Source:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_relay_mpi_examples.py
   :start-after: BEGIN_EXAMPLE("py_mpi_bcast_using_schema")
   :end-before: END_EXAMPLE("py_mpi_bcast_using_schema")
   :language: python
   :dedent: 8

* **Output:**

.. literalinclude:: t_conduit_docs_tutorial_python_relay_mpi_examples_out.txt
   :start-after: BEGIN_EXAMPLE("py_mpi_bcast_using_schema")
   :end-before: END_EXAMPLE("py_mpi_bcast_using_schema")
   :dedent: 4


Broadcast 
+++++++++++

* **Python Source:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_relay_mpi_examples.py
   :start-after: BEGIN_EXAMPLE("py_mpi_bcast")
   :end-before: END_EXAMPLE("py_mpi_bcast")
   :language: python
   :dedent: 8

* **Output:**

.. literalinclude:: t_conduit_docs_tutorial_python_relay_mpi_examples_out.txt
   :start-after: BEGIN_EXAMPLE("py_mpi_bcast")
   :end-before: END_EXAMPLE("py_mpi_bcast")
   :dedent: 4


All Gather Using Schema 
++++++++++++++++++++++++

* **Python Source:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_relay_mpi_examples.py
   :start-after: BEGIN_EXAMPLE("py_mpi_all_gather_using_schema")
   :end-before: END_EXAMPLE("py_mpi_all_gather_using_schema")
   :language: python
   :dedent: 8

* **Output:**

.. literalinclude:: t_conduit_docs_tutorial_python_relay_mpi_examples_out.txt
   :start-after: BEGIN_EXAMPLE("py_mpi_all_gather_using_schema")
   :end-before: END_EXAMPLE("py_mpi_all_gather_using_schema")
   :dedent: 4

..  
..
..  ================== ====================================
..   Method             Supported Modes
..  ================== ====================================
..   send/recv          known schema, generic
..   isend/irecv        known schema
..   reduce/all_reduce  known schema (w/ only leaf types)
..   gather/all_gather  known schema, generic
..   scatter            ? (known schema, generic)
..   all_to_all         ? (known schema, generic)
..  ================== ====================================






