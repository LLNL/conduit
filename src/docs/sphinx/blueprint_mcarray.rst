.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

===================
MCArray Blueprint
===================

Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To conform to the mcarray blueprint protocol, a Node must have at least one child and:

 * All children must be numeric leaves
 * All children must have the same number of elements


Properties and Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * **conduit::Node::is_contiguous()** 
     conduit::Node contains a general is_contiguous() instance method that is useful in the context of an mcarray.
     It can be used to detect if an mcarray has a contiguous memory layout for tuple components (eg: struct of arrays style)
     
     * Example: {x0, x1, ... , xN, y0, y1, ... , yN , z0, z1, ... , xN}

 * **conduit::blueprint::mcarray::is_interleaved(const Node &mcarray)** 
   
     Checks if an mcarray has an interleaved memory layout for tuple components (eg: struct of arrays style) 

     * Example: {x0, y0, z0, x1, y1, z1, ... , xN, yN, zN}

    
 * **conduit::blueprint::mcarray::to_contiguous(const Node &mcarray, Node &out)** 
   
     Copies the data from an mcarray into a new mcarray with a contiguous memory layout for tuple components 

     * Example: {x0, x1, ... , xN, y0, y1, ... , yN , z0, z1, ... , xN}

 * **conduit::blueprint::mcarray::to_interleaved(const Node &mcarray, Node &out)**  
   
     Copies the data from an mcarray into a new mcarray with interleaved tuple values 

     * Example: {x0, y0, z0, x1, y1, z1, ... , xN, yN, zN}

MCArray Examples
~~~~~~~~~~~~~~~~~~~

The mcarray blueprint namespace includes a function *xyz()*, that generates examples 
that cover a range of mcarray memory layout use cases.

.. code:: cpp

    conduit::blueprint::mcarray::examples::xyz(const std::string &mcarray_type,
                                                index_t npts,
                                                Node &out);

Here is a list of valid strings for the *mcarray_type* argument:

+-------------------+----------------------------------------------------------+
| **MCArray Type**  | **Description**                                          |
+-------------------+----------------------------------------------------------+
| interleaved       | One allocation, using interleaved memory layout          |
|                   | with float64 components (array of structs style)         |
+-------------------+----------------------------------------------------------+
| separate          | Three allocations, separe float64 components arrays for  |
|                   | {x,y,z}                                                  |
+-------------------+----------------------------------------------------------+
| contiguous        | One allocation, using a contiguous memory layout with    |
|                   | float64 components (struct of arrays style)              |
+-------------------+----------------------------------------------------------+
| interleaved_mixed | One allocation, using interleaved memory layout with:    |
|                   |  * float32 x components                                  |
|                   |  * float64 y components                                  |
|                   |  * uint8 z components                                    |
+-------------------+----------------------------------------------------------+

The number of components per tuple is always three (x,y,z).

*npts* specifies the number tuples created.

The resulting data is placed the Node *out*, which is passed in via a reference.

For more details, see the unit tests that exercise these examples in ``src/tests/blueprint/t_blueprint_mcarray_examples.cpp``.

