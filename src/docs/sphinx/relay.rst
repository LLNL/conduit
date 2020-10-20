.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

===================
Relay
===================

.. note::
    The **relay** APIs and docs are work in progress.


Conduit Relay is an umbrella project for I/O and communication functionality built on top of Conduit's Core API. It includes four components:

* **io** - I/O functionally beyond binary, memory mapped, and json-based text file I/O. Includes optional Silo, HDF5, and ADIOS I/O support. 
* **web** - An embedded web server (built using `CivetWeb <https://github.com/civetweb/civetweb>`_) that can host files and supports developing custom REST and WebSocket backends that use conduit::Node instances as payloads.
* **mpi**  - Interfaces for MPI communication using conduit::Node instances as payloads.
* **mpi::io** - I/O functionality as with io library but with some notion of collective writing to a shared file that can include multiple time steps and domains.

The **io** and **web** features are built into the *conduit_relay* library. The MPI functionality exists in a separate library *conduit_relay_mpi* to avoid include and linking issues for serial codes that want to use relay. Likewise, the parallel versions of the I/O functions are built into the *conduit_relay_mpi_io* library so it can be linked to parallel codes.


.. toctree::
     relay_io
     relay_mpi
..    relay_web
..     relay_mpi


