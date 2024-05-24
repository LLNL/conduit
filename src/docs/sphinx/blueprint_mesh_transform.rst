.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

.. _mesh_blueprint_transforms:

===========================
Mesh Blueprint Transforms
===========================

These are methods in conduit::blueprint::mesh.  Generally they take a Node mesh and return information about it or transform it.

Multi-domain methods
--------------------
- is_multi_domain()
- number_of_domains()
- domains() returns the domains in the mesh into a std::vector of Node
- to_multi_domain()
- generate_index()
- generate_index_for_single_domain()


Strip mesh
----------
Blueprint represents a 1D mesh as a polyline.  Carter represents a 1D mesh as a "strip" of 2D quads.
- can_generate_strip()
- generate_strip() (two overloads)

Visualize adjacency sets
------------------------
Create new fields to help debug adjacency sets.
- paint_adjset()

Partition a mesh
----------------
- partition()
- partition_map_back()
- generate_boundary_partition_field()

Flatten a mesh into a table
---------------------------
- flatten()
- generate_domain_ids()  a helper method?

Generate a derived topology
---------------------------
The blueprint library has several helper functions that each take a polyhedral or polygonal topology and produce a new topology based on it.  Each of these functions uses the TopologyMetadata helper class to map the "dimensional cascade" from one dimension to an adjacent dimensions.  For example, the TopologyMetadata for a 3D mesh tells which sides are associated with an element, which edges are associated with a side, and which vertices area associated with an edge.  For illustration purposes, here are two simple meshes, one polygonal (2D) and one polyhedral (3D).

  [include pentagon and hexahedron]

- The generate_points(), generate_lines(), and generate_faces() functions each produce new topology containing those elements of the original topology and are useful for visualization and debugging.

  [include a triple figure showing points, lines, and faces]

- generate_sides() produces a "side" topology.  The new topology adds a vertex at each face centroid and, for 3D, each element centroid, and adds edges from each original, adjacent vertex to these new vertices.  Thus, each 2D element or 3D face in the new topology is divided into triangles.  VisIt can use this new topology to display polyhedral or polygonal meshes.

  [include a figure showing sides]

- generate_corners() produces a "corner" topology.  The new topology adds a vertex at each face centroid, each edge midpoint, and, for 3D, each element centroid, and adds edges connecting the edge and face midpoints and the face and element centroids.  Thus, each 2D element or 3D face in the new topology is divided into quads.

  [include a figure showing corners]

- generate_centroids() ... not sure?? beyond what the name of the function implies.  This function is ultimately called from the ___ which is used by the Tiler class to generate a tiled mesh pattern.
  
Transform a coordset
--------------------
- Transform between explicit and rectilinear: to_explicit, to_rectilinear

Transform a topology
--------------------
- Transform uniform -> rectilinear, structured, unstructured
- Transform rectilinear -> structured, unstructured
- Transform structured -> unstructured
- Transform unstructured -> polytopal (polygonal)

Transform a matset or a field
-----------------------------
- To multi-buffer-full
- To sparse by element
- To multi-buffer-by-material
- To Silo (specset too)

Transform adjset
----------------
Transform between pairwise and maxshare

