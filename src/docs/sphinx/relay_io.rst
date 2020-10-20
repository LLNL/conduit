.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

===================
Relay I/O
===================

Conduit Relay I/O provides optional Silo, HDF5, and ADIOS I/O interfaces. 

These interfaces can be accessed through a generic path-based API, generic handle class, or through APIs specific to each underlying I/O interface.  The specific APIs provide lower level control and allow reuse of handles, which is more efficient for most non-trivial use cases. The generic handle class strikes a balance between usability and efficiency. 


.. _relay_io_generic_interface:

Relay I/O Path-based Interface
--------------------------------

The path-based Relay I/O interface allows you to read and write conduit::Nodes using any enabled I/O interface through a simple path-based (string) API. The underlying I/O interface is selected using the extension of the destination path or an explicit protocol argument.


The ``conduit_relay`` library provides the following methods in the ``conduit::relay::io`` namespace:

 * ``relay::io::save`` 
 
   * Saves the contents of the passed Node to a file. Works like a ``Node::set`` to the file: if the file exists, it is overwritten to reflect contents of the passed Node. 
 
 * ``relay::io::save_merged`` 
 
   * Merges the contents of the passed Node to a file. Works like a ``Node::update`` to the file: if the file exists, new data paths are appended, common paths are overwritten, and other existing paths are not changed. 


 * ``relay::io::load``  
    
   * Loads the contents of a file into the passed Node. Works like a ``Node::set`` from the contents of the file: if the Node has existing data, it is overwritten to reflect contents of the file.
 
 * ``relay::io::load_merged`` 
   
   * Merges the contents of a file into the passed Node. Works like a ``Node::update`` rom the contents of the file: if the Node has existing data, new data paths are appended, common paths are overwritten, and other existing paths are not changed. 

                             
The ``conduit_relay_mpi_io`` library provides the ``conduit::relay::mpi::io`` namespace which includes variants of these methods which take a MPI Communicator. These variants pass the communicator to the underlying I/O interface to enable collective I/O. Relay currently only supports collective I/O for ADIOS.


Relay I/O Path-based Interface Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save and Load
+++++++++++++++

* **C++ Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_relay_io_generic_examples.cpp
   :start-after: BEGIN_EXAMPLE("relay_io_example_1_json")
   :end-before:  END_EXAMPLE("relay_io_example_1_json")
   :language: cpp
   :dedent: 4

* **Output:**

.. literalinclude:: t_conduit_docs_relay_io_generic_examples_out.txt
   :start-after: BEGIN_EXAMPLE("relay_io_example_1_json")
   :end-before:  END_EXAMPLE("relay_io_example_1_json")


Save Merged 
+++++++++++++

* **C++ Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_relay_io_generic_examples.cpp
   :start-after: BEGIN_EXAMPLE("relay_io_example_2_hdf5")
   :end-before:  END_EXAMPLE("relay_io_example_2_hdf5")
   :language: cpp
   :dedent: 4

* **Output:**

.. literalinclude:: t_conduit_docs_relay_io_generic_examples_out.txt
   :start-after: BEGIN_EXAMPLE("relay_io_example_2_hdf5")
   :end-before:  END_EXAMPLE("relay_io_example_2_hdf5")


Load Merged 
+++++++++++++

* **C++ Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_relay_io_generic_examples.cpp
   :start-after: BEGIN_EXAMPLE("relay_io_example_3_hdf5")
   :end-before:  END_EXAMPLE("relay_io_example_3_hdf5")
   :language: cpp
   :dedent: 4

* **Output:**

.. literalinclude:: t_conduit_docs_relay_io_generic_examples_out.txt
   :start-after: BEGIN_EXAMPLE("relay_io_example_3_hdf5")
   :end-before:  END_EXAMPLE("relay_io_example_3_hdf5")

Load from Subpath
+++++++++++++++++++

* **C++ Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_relay_io_generic_examples.cpp
   :start-after: BEGIN_EXAMPLE("relay_io_example_4_hdf5")
   :end-before:  END_EXAMPLE("relay_io_example_4_hdf5")
   :language: cpp
   :dedent: 4

* **Output:**

.. literalinclude:: t_conduit_docs_relay_io_generic_examples_out.txt
   :start-after: BEGIN_EXAMPLE("relay_io_example_4_hdf5")
   :end-before:  END_EXAMPLE("relay_io_example_4_hdf5")


Save to Subpath
+++++++++++++++++

* **C++ Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_relay_io_generic_examples.cpp
   :start-after: BEGIN_EXAMPLE("relay_io_example_5_hdf5")
   :end-before:  END_EXAMPLE("relay_io_example_5_hdf5")
   :language: cpp
   :dedent: 4

* **Output:**

.. literalinclude:: t_conduit_docs_relay_io_generic_examples_out.txt
   :start-after: BEGIN_EXAMPLE("relay_io_example_5_hdf5")
   :end-before:  END_EXAMPLE("relay_io_example_5_hdf5")


.. things not yet covered: options

Relay I/O Handle Interface
----------------------------

The ``relay::io::IOHandle`` class provides a high level interface to query, read, and modify files.

It provides a generic interface that is more efficient than the path-based interface for protocols like HDF5 which support partial I/O and querying without reading the entire contents of a file.
It also supports simpler built-in protocols (conduit_bin, json, etc) that do not support partial I/O for convenience. Its basic contract is that changes to backing (file on disk, etc) are not guaranteed to be reflected until the handle is closed. Relay I/O Handle supports reading AXOM Sidre DataStore Style files. Relay I/O Handle does not yet support Silo or ADIOS. 

IOHandle has the following instance methods: 

 * ``open``
   
   * Opens a handle. The underlying I/O interface is selected using the extension of the destination path or an explicit protocol argument. Supports reading and writing by default. Select a different mode by passing an options node that contains a ``mode`` child with one of the following strings:


   .. list-table:: 
      :widths: 10 20

      * - ``rw`` read + write (default mode)
        - Supports both read and write operations. Creates file if it does not exist.

      * - ``r`` read only 
        - Only supports read operations. Throws an Error if you open a non-existing file or on any attempt to write. 

      * - ``w`` write only 
        - Only supports write operations. Throws an Error on any attempt to read.

  .. DANGER::
    Note: While you can read from and write to subpaths using a handle, IOHandle *does not* support opening a file with a subpath (e.g. ``myhandle.open("file.hdf5:path/data")``).


 * ``read``
   
   * Merges the contents from the handle or contents from a subpath of the handle into the passed Node. Works like a ``Node::update`` from the handle: if the Node has existing data, new data paths are appended, common paths are overwritten, and other existing paths are not changed. 


 * ``write``
 
   * Writes the contents of the passed Node to the handle or to a subpath of the handle. Works like a ``Node::update`` to the handle: if the handle has existing data, new data paths are appended, common paths are overwritten, and other existing paths are not changed. 

 * ``has_path``
 
   * Checks if the handle contains a given path.

 * ``list_child_names``
 
   * Returns a list of the child names at a given path, or an empty list if the path does not exist.

 * ``remove``
 
   * Removes any data at and below a given path. With HDF5 the space may not be fully reclaimed.

 * ``close``
   
   * Closes a handle. This is when changes are realized to the backing (file on disc, etc).


Relay I/O Handle Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~
* **C++ Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_relay_io_handle_examples.cpp
   :start-after: BEGIN_EXAMPLE("relay_io_handle_example_1")
   :end-before:  END_EXAMPLE("relay_io_handle_example_1")
   :language: cpp
   :dedent: 4

* **Output:**

.. literalinclude:: t_conduit_docs_relay_io_handle_examples_out.txt
   :start-after: BEGIN_EXAMPLE("relay_io_handle_example_1")
   :end-before:  END_EXAMPLE("relay_io_handle_example_1")

* **Python Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_relay_io_handle_examples.py
   :start-after: BEGIN_EXAMPLE("py_relay_io_handle")
   :end-before:  END_EXAMPLE("py_relay_io_handle")
   :language: python
   :dedent: 8

* **Output:**

.. literalinclude:: t_conduit_docs_tutorial_python_relay_io_handle_examples_out.txt
   :start-after: BEGIN_EXAMPLE("py_relay_io_handle")
   :end-before:  END_EXAMPLE("py_relay_io_handle")
   :dedent: 4


* **C++ Sidre Basic Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_relay_io_handle_examples.cpp
   :start-after: BEGIN_EXAMPLE("relay_io_handle_example_sidre_1")
   :end-before:  END_EXAMPLE("relay_io_handle_example_sidre_1")
   :language: cpp
   :dedent: 4

* **Output:**

.. literalinclude:: t_conduit_docs_relay_io_handle_examples_out.txt
   :start-after: BEGIN_EXAMPLE("relay_io_handle_example_sidre_1")
   :end-before:  END_EXAMPLE("relay_io_handle_example_sidre_1")

* **Python Sidre Basic Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_relay_io_handle_examples.py
   :start-after: BEGIN_EXAMPLE("py_relay_io_handle_sidre")
   :end-before:  END_EXAMPLE("py_relay_io_handle_sidre")
   :language: python
   :dedent: 8

* **Output:**

.. literalinclude:: t_conduit_docs_tutorial_python_relay_io_handle_examples_out.txt
   :start-after: BEGIN_EXAMPLE("py_relay_io_handle_sidre")
   :end-before:  END_EXAMPLE("py_relay_io_handle_sidre")
   :dedent: 4

* **C++ Sidre with Root File Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_relay_io_handle_examples.cpp
   :start-after: BEGIN_EXAMPLE("relay_io_handle_example_sidre_2")
   :end-before:  END_EXAMPLE("relay_io_handle_example_sidre_2")
   :language: cpp
   :dedent: 4

* **Output:**

.. literalinclude:: t_conduit_docs_relay_io_handle_examples_out.txt
   :start-after: BEGIN_EXAMPLE("relay_io_handle_example_sidre_2")
   :end-before:  END_EXAMPLE("relay_io_handle_example_sidre_2")

* **Python Sidre with Root File Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_tutorial_python_relay_io_handle_examples.py
   :start-after: BEGIN_EXAMPLE("py_relay_io_handle_sidre_root")
   :end-before:  END_EXAMPLE("py_relay_io_handle_sidre_root")
   :language: python
   :dedent: 8

* **Output:**

.. literalinclude:: t_conduit_docs_tutorial_python_relay_io_handle_examples_out.txt
   :start-after: BEGIN_EXAMPLE("py_relay_io_handle_sidre_root")
   :end-before:  END_EXAMPLE("py_relay_io_handle_sidre_root")
   :dedent: 4




Relay I/O HDF5 Interface
---------------------------

The Relay I/O HDF5 interface provides methods to read and write Nodes using HDF5 handles.
It is also the interface used to implement the path-based and handle I/O interfaces for 
HDF5. This interface provides more control and allows more efficient reuse of I/O handles.
It is only available in C++.

Relay I/O HDF5 Interface Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a example exercising the basic parts of Relay I/O's HDF5 interface, for
more detailed documentation see the ``conduit_relay_io_hdf5_api.hpp`` header file. 

HDF5 I/O Interface Basics 
++++++++++++++++++++++++++

* **C++ Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_relay_io_hdf5_examples.cpp
   :start-after: BEGIN_EXAMPLE("relay_io_example_hdf5_interface_1")
   :end-before:  END_EXAMPLE("relay_io_example_hdf5_interface_1")
   :language: cpp
   :dedent: 4



* **Output:**

.. literalinclude:: t_conduit_docs_relay_io_hdf5_examples_out.txt
   :start-after: BEGIN_EXAMPLE("relay_io_example_hdf5_interface_1")
   :end-before:  END_EXAMPLE("relay_io_example_hdf5_interface_1")



HDF5 I/O Options 
++++++++++++++++++

* **C++ Example:**

.. literalinclude:: ../../tests/docs/t_conduit_docs_relay_io_hdf5_examples.cpp
   :start-after: BEGIN_EXAMPLE("relay_io_example_hdf5_interface_opts")
   :end-before:  END_EXAMPLE("relay_io_example_hdf5_interface_opts")
   :language: cpp
   :dedent: 4

* **Output:**

.. literalinclude:: t_conduit_docs_relay_io_hdf5_examples_out.txt
   :start-after: BEGIN_EXAMPLE("relay_io_example_hdf5_interface_opts")
   :end-before:  END_EXAMPLE("relay_io_example_hdf5_interface_opts")

You can verify using ``h5stat`` that the data set was written to the hdf5 file using chunking and
compression.


