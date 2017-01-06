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

===================
Relay
===================

.. note::
    The **relay** APIs and docs are work in progress.


Conduit Relay is an umbrella project for I/O and communication functionality built on top of Conduit's Core API. It includes three components:

* **io** - I/O functionally beyond binary, memory mapped, and json-based text file I/O. Includes optional Silo and HDF5 I/O support. 
* **web** - An embedded web server (built using `CivetWeb <https://github.com/civetweb/civetweb>`_) that can host files and supports developing custom REST and WebSocket backends that use conduit::Node instances as payloads.
* **mpi**  - Interfaces for MPI communication using conduit::Node instances as payloads.

The **io** and **web** features are built into the *conduit_relay* library. The MPI functionality exists in a separate library *conduit_relay_mpi* to avoid include and linking issues for serial codes that want to use relay.


.. .. toctree::
..     relay_io
..    relay_web
..     relay_mpi


