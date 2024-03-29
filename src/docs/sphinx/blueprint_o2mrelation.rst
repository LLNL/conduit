.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

=====================
O2MRelation Blueprint
=====================

Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To conform to the **o2mrelation** protocol, a *Node* must have the following characteristics:

 * Be of an *Object* type (*List* types are not allowed)
 * Contain at least one child that is a numeric leaf

The numeric leaf/leaves of the *Node* must not be under any of the following "meta component" paths, which all have special meanings and particular requirements when specified as part of an **o2mrelation**:

 * ``sizes``: An integer leaf that specifies the number "many" items associated with each "one" in the relationship.
 * ``offsets``: An integer leaf that denotes the start index of the "many" sequence for each "one" in the relationship.
 * ``indices``: An integer leaf that indicates the index values of items in the values array(s).

All of the above paths are optional and will resolve to simple defaults if left unspecified. These defaults are outlined below:

 * ``sizes``: An array of ones (i.e. ``[1, 1, 1, ...]``) to indicate that the values have one-to-one correspondance.
 * ``offsets``: An array of monotonically increasing index values (i.e. ``[0, 1, 2, ...]``) to indicate that the values are compacted.
 * ``indices``: An array of monotonically increasing index values (i.e. ``[0, 1, 2, ...]``) to indicate that the values are ordered sequentially.

Taken in sum, the consituents of the **o2mrelation** schema describe how data (contained in numeric leaves and indexed through ``indices``) maps in grouped clusters (defined by ``sizes`` and ``offsets``) from a source space (the "one" space) to a destination space (the "many" space).

.. note::
   While the ``sizes``, ``offsets``, and ``indices`` meta components of the **o2mrelation** definition are
   independently defined, they interplay in ways that aren't immediately obvious. The most commonly missed
   of these "gotcha" behaviors are defined below:

   * Every **o2mrelation** must define both or neither of ``sizes`` and ``offsets``.
   * If none of the meta component paths are specified, their defaults set the **o2mrelation** to be a compacted, one-to-one relationship.
   * The ``sizes`` and ``offsets`` values always refer to entries in ``indices``. If ``indices`` isn't present, it defaults to a "pass through" index, so in this case ``sizes`` and ``offsets`` can be thought of as indexing directly into the numeric leaves.

Properties, Queries, and Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * **conduit::blueprint::o2mrelation::data_paths(const Node &o2mrelation)**

     Returns a ``std::vector<std::string>`` object containing all of the data paths in the given **o2mrelation** node.

     * Example Input:

     .. code:: json

         {
           "values":   // [int64],
           "sizes":    // [int64],
           "offsets":  // [int32],
           "other":    // [ char8]
         }

     * Example Output:

     .. code:: json

         ["values"]

 * **conduit::blueprint::o2mrelation::compact_to(const Node &o2mrelation, Node &res)**

     Generates a data-compacted version of the given **o2mrelation** (first parameter) and stores it in the given output node (second parameter).

     * Example Input:

     .. code:: json

         {
           "values":  [-1, 2, 3, -1, 0, 1, -1],
           "sizes":   [2, 2],
           "offsets": [4, 1]
         }

     * Example Output:

     .. code:: json

         {
           "values":  [0, 1, 2, 3],
           "sizes":   [2, 2],
           "offsets": [0, 2]
         }


 * **conduit::blueprint::o2mrelation::generate_offsets(Node &n, Node &info)**

     Updates the contents of the given node's ``offsets`` child so that it refers to a compacted sequence of one-to-many relationships.

     * Example:

     .. code:: json

         // Input //
         {
           "values": [0, 1, 2, 3],
           "sizes": [2, 2]
         }

         // Output //
         {
           "values": [0, 1, 2, 3],
           "sizes": [2, 2],
           "offsets": [0, 2]
         }

O2MRelation Examples
~~~~~~~~~~~~~~~~~~~~~~~

The **o2mrelation** blueprint namespace includes a function *uniform()*, which generates example
hierarchies that cover a range of **o2mrelation** use cases.

.. code:: cpp

    conduit::blueprint::o2mrelation::examples::uniform(conduit::Node &res,
                                                       conduit::index_t nones,
                                                       conduit::index_t nmany = 0,
                                                       conduit::index_t noffset = 0,
                                                       const std::string &index_type = "unspecified");

This function's arguments have the following precise meanings:

 * ``nones``: The number of "one"s in the one-to-many relationship.
 * ``nmany``: The number of "many"s associated with each of the "one"s.
 * ``noffset``: The stride between each "many" sequence (must be at least ``nmany``).
 * ``index_type``: The style of element indirection, which must be one of the following:

   * ``"unspecified"``: Index indirection will be omitted from the output.
   * ``"default"``: The default value for index indirection will be supplied in the output.
   * ``"reversed"``: The index indirection will be specified such that the data is reversed relative to its default order.

The ``nmany`` and ``noffset`` parameters can both be set to zero to omit the ``sizes`` and ``offsets`` meta components from the output.
Similarly, the ``index_type`` parameter can be omitted or set to ``"unspecified"`` in order to remove the ``indices`` section from the output.

For more details, see the unit tests that exercise these examples in ``src/tests/blueprint/t_blueprint_o2mrelation_examples.cpp``.
