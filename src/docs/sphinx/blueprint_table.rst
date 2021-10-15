.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

===================
Table Blueprint
===================
The *table* blueprint protocol provides a convention for expressing tabular data in Conduit.
Each data entry in a *table* represents a column and each column contains the same
number of rows.
Nodes that conform to the *table* blueprint protocol are easily translated to and from CSV files.

Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *table* blueprint protocol comes in two forms - *single table* and *many table*.

To conform to the *single table* blueprint protocol, a Node must have a "values" child which is a *list* OR an *object* and:

 * All of "values" children are data arrays OR *mcarrays*
 * All of "values" children must have the same number of elements

To conform to the *many table* blueprint protocol, a Node must be a
*list* OR an *object* and:

 * All of its children are valid *single tables* as defined above.

Table Examples
~~~~~~~~~~~~~~~~~~~

An example of a *table* blueprint in yaml format:

.. code:: yaml

    values:
        scalar_column: [0, 1, 2, 3]
        vector_column:
            x: [0, 1, 2, 3]
            y: [0, 1, 2, 3]
            z: [0, 1, 2, 3]

An example of a *table* blueprint containing multiple tables in yaml format:

.. code:: yaml

    point_data:
        values:
            points:
                x: [0, 1, 2, 3]
                y: [0, 1, 2, 3]
                z: [0, 1, 2, 3]
            scalar_data: [0, 1, 2, 3]
    element_data:
        values:
            scalar_data: [0, 1]
            vector_data:
                a: [0, 1]
                b: [0, 1]
                c: [0, 1]


The table blueprint namespace includes a function *basic()*, that generates a simple
example of tabular data.

.. code:: cpp

    conduit::blueprint::table::examples::basic(conduit::index_t nx,
                                               conduit::index_t ny,
                                               conduit::index_t nz,
                                               Node &res);

This function will generate points (points/x, points/y, points/z) in a uniform manner based off the arguments
*nx*, *ny*, and *nz*.
Also included in the output table is a point_data column that starts at 0 and increases by 1 for each point.

The resulting data is placed the Node *res*, which is passed in via a reference.

For more details, see the unit tests that exercise these examples in ``src/tests/blueprint/t_blueprint_table_verify.cpp``
and ``src/tests/blueprint/t_blueprint_table_examples.cpp``.

