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

===================
indexset
===================

Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To conform to the indexset blueprint protocol, a Node must have

 * An integer child named ``n`` specifying the size of the domain,
 * An integer list named ``idx`` specifying the members of the domain selected by the indexset.

===================
carray
===================

Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To conform to the carray blueprint protocol, a Node must have:

 * A child ``nz``, the nonzero data in the compressed array
 * A child ``idx`` specifying locations of the nonzero data.  ``idx`` is one or more strings or indexsets.

If ``idx`` is a string, it refers by name to a previously-defined indexset.  If ``idx`` is a list, the indexsets should be composed in sequence to support multiple levels of indirection.  The first indexset must have the same number of elements as ``nz``.

Properties and Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * **conduit::blueprint::carray::nnz(const Node &carray)**

     Returns the number of nonzero elements in ``carray``.

 * **conduit::blueprint::carray::length(const Node &carray)**

     Returns the uncompressed length of ``carray``, that is, ``nnz(carray)`` + number of zeros in ``carray``.

 * **conduit::blueprint::carray::compress(const Node &array, Node &out)**

     Copies the data from a numeric array into a new carray with a single level of indirection.

 * **conduit::blueprint::carray::compress(const Node &mcarray, Node &out)**

     Copies the data from an mcarray into an object:
     - The nodes in the object are carrays representing the ordinals in the mcarray tuples
     - Each carray has a single level of indirection, and the (uncompressed) length of each carray equals the number of tuples in mcarray

 * **conduit::blueprint::carray::inflate(const Node &carray, Node &out)**

     Copies the data from a carray into a new numeric Node with the same length as the carray.  (Restores the zeros.)

 * **conduit::blueprint::carray::inflate(const Node &object, Node &out)**

     Given an object with ``k`` children that are all carrays of equal length ``l``, copies the data from each of the children to a new mcarray.  The new mcarray has ``l`` tuples, of ``k`` members.

Examples
~~~~~~~~~~~~~~~~~~~~~

