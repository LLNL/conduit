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
sarray
===================

Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To conform to the sarray blueprint protocol, a Node must have:

 * An integer child ``length``, the total number of elements (zero and nonzero) in the array.
 * A child ``nz``, the nonzero data in the array.
 * A child ``idx``, the locations of the nonzero data.

The common case is that ``idx`` will be a single integer array with a length equal to ``nz``.  To support codes that need multiple levels of indirection, ``idx`` can also be a list of integer arrays.


Properties and Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * **conduit::blueprint::sarray::nnz(const Node &sarray)**

     Returns the number of nonzero elements in ``sarray``.

 * **conduit::blueprint::sarray::length(const Node &sarray)**

     Returns the full length of ``sarray``, that is, ``nnz(sarray)`` + number of zeros in ``sarray``.

 * **conduit::blueprint::sarray::make_sparse(const Node &array, Node &out)**

     Copies the data from a numeric array into a new sarray.

 * **conduit::blueprint::sarray::make_sparse(const Node &mcarray, Node &out)**

     Copies the data from an mcarray into an object ``out``:
     - The child objects of ``out`` are sparse arrays representing the components in the mcarray tuples
     - The (full) length of each sarray equals the number of tuples in mcarray

 * **conduit::blueprint::sarray::make_dense(const Node &sarray, Node &out)**

     Copies the data from a sparse array into a new numeric Node with the same length as the sarray.  (Restores the zeros.)

 * **conduit::blueprint::sarray::make_dense(const Node &object, Node &out)**

     Given an object with ``k`` children that are all carrays of equal length ``l``, copies the data from each of the children to a new mcarray.  The new mcarray has ``l`` tuples, of ``k`` members.

Examples
~~~~~~~~~~~~~~~~~~~~~

The sarray blueprint namespace includes a function *sparse()*, that generates examples of a range of sarray use cases.  The namespace also includes *full()*, which generates the full equivalents (zeros included) of the same use cases.

.. code:: cpp

    conduit::blueprint::sarray::examples::sparse(const std::string &array_type,
                                                 index_t x,
                                                 index_t y,
                                                 Node &out);

    conduit::blueprint::sarray::examples::full(const std::string &array_type,
                                               index_t x,
                                               index_t y,
                                               Node &out);

Here is a list of valid strings for the *array_type* argument:

+-------------------+----------------------------------------------------------+
| **Array Type**    | **Description**                                          |
+-------------------+----------------------------------------------------------+
| eye               | A 2D x-by-x identity matrix represented by a row-major   |
|                   | 1D array.                                                |
+-------------------+----------------------------------------------------------+
| volfrac_matmajor  | An mcarray with three components.  Each component is a   |
|                   | separate allocation.  This represents an (x, y)          |
|                   | Cartesian grid with three different materials.           |
+-------------------+----------------------------------------------------------+
