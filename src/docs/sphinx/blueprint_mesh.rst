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

.. _mesh_blueprint:

===================
Mesh Blueprint
===================

This section provides details about the Mesh Blueprint. Lots of them.
We don't have a Mesh Blueprint tutorial yet, if you are looking to wrap your mind 
around the basic mechanics of describing a mesh, you may want to start by reviewing
the :ref:`detailed_uniform_example`
and exploring the other :ref:`examples` included in the blueprint library. 



Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


The Blueprint protocol defines a single-domain computational mesh using one or more Coordinate Sets (via child ``coordsets``), one or more Topologies (via child  ``topologies``), zero or more Materials Sets (via child ``matsets``), zero or more Fields (via child ``fields``), optional Adjacency Set information (via child ``adjsets``), and optional State information (via child ``state``).
The protocol defines multi-domain meshes as *Objects* that contain one or more single-domain mesh entries.
For simplicity, the descriptions below are structured relative to a single-domain mesh *Object* that contains one Coordinate Set named ``coords``, one Topology named ``topo``, and one Material Set named ``matset``.


Coordinate Sets
++++++++++++++++++++

To define a computational mesh, the first required entry is a set of spatial coordinate tuples that can underpin a mesh topology.

The mesh blueprint protocol supports sets of spatial coordinates from three coordinate systems:

* Cartesian: {x,y,z}
* Cylindrical: {r,z}
* Spherical: {r,theta,phi}

The mesh blueprint protocol supports three types of Coordinate Sets: ``uniform``, ``rectilinear``, and ``explicit``.  To conform to the protocol, each entry under ``coordsets`` must be an *Object* with entries from one of the cases outlined below: 

* **uniform**

   An implicit coordinate set defined as the cartesian product of i,j,k dimensions starting at an ``origin`` (ex: {x,y,z}) using a given ``spacing`` (ex: {dx,dy,dz}).

  * Cartesian
  
  
    * coordsets/coords/type: “uniform”
    * coordsets/coords/dims/{i,j,k}
    * coordsets/coords/origin/{x,y,z} (optional, default = {0.0, 0.0, 0.0})
    * coordsets/coords/spacing/{dx,dy,dz} (optional, default = {1.0, 1.0, 1.0})


  * Cylindrical
  
  
    * coordsets/coords/type: “uniform”
    * coordsets/coords/dims/{i,j}
    * coordsets/coords/origin/{r,z} (optional, default = {0.0, 0.0})
    * coordsets/coords/spacing/{dr,dz} (optional, default = {1.0, 1.0})


  * Spherical
  
  
    * coordsets/coords/type: “uniform”
    * coordsets/coords/dims/{i,j}
    * coordsets/coords/origin/{r,theta,phi} (optional, default = {0.0, 0.0, 0.0})
    * coordsets/coords/spacing/{dr,dtheta, dphi} (optional, default = {1.0, 1.0, 1.0})


* **rectilinear** 

  An implicit coordinate set defined as the cartesian product of passed coordinate arrays.
  
  * Cartesian
  
  
    * coordsets/coords/type: “rectilinear”
    * coordsets/coords/values/{x,y,z}

  * Cylindrical:
  
    * coordsets/coords/type: “rectilinear”
    * coordsets/coords/values/{r,z}

  * Spherical


    * coordsets/coords/type: “rectilinear”
    * coordsets/coords/values/{r,theta,phi}


* **explicit**

  An explicit set of coordinates, which includes ``values`` that conforms to the  **mcarray** blueprint protocol.

  * Cartesian
  
  
    * coordsets/coords/type: “explicit”
    * coordsets/coords/values/{x,y,z}

  * Cylindrical
  
  
    * coordsets/coords/type: “explicit”
    * coordsets/coords/values/{r,z}

  * Spherical
  
  
    * coordsets/coords/type: “explicit”
    * coordsets/coords/values/{r,theta,phi}

.. note::
   In all of the coordinate space definitions outlined above, spherical coordinates adhere to the definitions of
   ``theta``/``phi`` used in the physics and engineering domains. Specifically, this means that ``theta`` refers to
   the polar angle of the coordinate (i.e. the angle from the +Z cartesian axis) and ``phi`` refers to the azimuthal
   angle of the coordinate (i.e. the angle from the +X cartesian axis). The figure below most succinctly describes
   these conventions:

   .. figure:: spherical_coordinates_render.png
       :width: 400px
       :align: center

       Figure of ``spherical`` coordinate conventions (courtesy of `Wikipedia <https://en.wikipedia.org/wiki/Spherical_coordinate_system>`_)

Toplogies
++++++++++++++++++++
The next entry required to describe a computational mesh is its topology. To conform to the protocol, each entry under *topologies* must be an *Object* that contains one of the topology descriptions outlined below.


Topology Nomenclature 
====================================

The mesh blueprint protocol describes meshes in terms of ``vertices``, ``edges``, ``faces``, and ``elements``.

The following element shape names are supported:

========== ================  ===================================================
Name        Geometric Type    Specified By
========== ================  ===================================================
point       point             an index to a single coordinate tuple
line        line              indices to 2 coordinate tuples
tri         triangle          indices to 3 coordinate tuples
quad        quadrilateral     indices to 4 coordinate tuples
tet         tetrahedron       indices to 4 coordinate tuples
hex         hexahedron        indices to 8 coordinate tuples
polygonal   polygon           indices to N end-to-end coordinate tuples
polyhedral  polyhedron        indices to M polygonal faces
========== ================  ===================================================

.. note
   The expected index ordering with in an element (also referred to as a winding order) is not specified by the blueprint. 
   In the future, we plan to provide transforms to help convert between orderings, are not likely to specify specific orderings.
..

Association with a Coordinate Set
====================================

Each topology entry must have a child ``coordset`` with a string that references a valid coordinate set by name.

    * topologies/topo/coordset: "coords"

Optional association with a Grid Function
==========================================

Topologies can optionally include a child ``grid_function`` with a string that references a valid field by name.

    * topologies/topo/grid_function: "gf"


Implicit Topology
===============================

The mesh blueprint protocol accepts four implicit ways to define a topology on a coordinate set. The first simply uses all the points in a given coordinate set and the rest define grids of elements on top of a coordinate set. For the grid cases with a coordinate set with 1D coordinate tuples, *line* elements are used, for sets with 2D coordinate tuples *quad* elements are used, and for 3D coordinate tuples *hex* elements are used.

* **points**: An implicit topology using all of the points in a coordinate set. 
   
   * topologies/topo/coordset: "coords"
   * topologies/topo/type: "points"

* **uniform**: An implicit topology that defines a grid of elements on top of a *uniform* coordinate set. 
   
   * topologies/topo/coordset: "coords"
   * topologies/topo/type: “uniform”
   * topologies/topo/elements/origin/{i0,j0,k0} (optional, default = {0,0,0})
   
* **rectilinear**: An implicit topology that defines a grid of elements on top of a *rectilinear* coordinate set. 
   
   * topologies/topo/coordset: "coords"
   * topologies/topo/type: “rectilinear”
   * topologies/topo/elements/origin/{i0,j0,k0} (optional, default = {0,0,0})
  

.. .. attention::
..    (can we collapse uniform + rectilinear?)
.. * topologies/topo/type: “structured”
.. * topologies/topo/elements/dims: "implicit"
.. * topologies/topo/elements/origin/{i0,j0,k0} (optional, default = {0,0,0})
.. * topologies/coordset: "coords"


* **structured**: An implicit topology that defines a grid of elements on top of an *explicit* coordinate set.
  
  * topologies/topo/coordset: "coords"
  * topologies/topo/type = “structured”
  * topologies/topo/elements/dims/{i,j,k}
  * topologies/topo/elements/origin/{i0,j0,k0} (optional, default = {0,0,0})



Explicit (Unstructured) Topology
=================================


Single Shape Topologies
************************

For topologies using a homogenous collection of element shapes (eg: all hexs), the topology can be specified by 
a connectivity array and a shape name.

  * topologies/topo/coordset: "coords"
  * topologies/topo/type: “unstructured”
  * topologies/topo/elements/shape: (shape name)
  * topologies/topo/elements/connectivity: (index array)



Mixed Shape Toplogies 
************************

For topologies using a non-homogenous collections of element shapes (eg: hexs and tets), the topology can 
specified using a single shape topology for each element shape.

* **list** - A Node in the *List* role, that contains a children that conform to the *Single Shape Topology* case. 

* **object** - A Node in the *Object* role, that contains a children that conform to the *Single Shape Topology* case. 

.. note::
   Future version of the mesh blueprint will expand support to include mixed elements types in a single array with related
   index arrays.


Element Windings
^^^^^^^^^^^^^^^^^^^^^^

The mesh blueprint does yet not have a prescribed winding convention (a way to order the association of vertices to elements) or more generally to 
outline a topology's `dimensional cascade`  (how elements are related to faces, faces are related to edges, and edges are related to vertices. )

This is a gap we are working to solve in future versions of the mesh blueprint, with a goal of providing transforms to
help convert between different winding or cascade schemes.

That said VTK (and VTK-m) winding conventions are assumed by MFEM, VisIt, or Ascent when using Blueprint data.


.. * **stream** - (strem description)
..   (specifying stream ids and stream connectivity)
..
..
..   * topologies/topo/elements/element_types: ()
..   * topologies/topo/elements/stream: ()
..
.. Indexed Streams
.. ^^^^^^^^^^^^^^^^^^^
..
.. * Stream of Indexed Elements
..
..
..     * topology/elements/element_types: ()
..     * topology/elements/element_index/stream_ids: ()
..     * topology/elements/element_index/offsets: ()
..     * topology/elements/stream: ()
..
.. * Stream of Contiguous Segments of Element Types
..
..
..     * topology/elements/element_types: ()
..     * topology/elements/segment_index/stream_ids: ()
..     * topology/elements/segment_index/element_counts: ()
..     * topology/elements/stream: ()


Polygonal/Polyhedral Topologies
*********************************

The **polygonal** and **polyhedral** topology shape types are structually
identical to the other explicit topology shape types (see the *Single Shape Topologies*
section above), but the contents of their ``elements`` sections look slightly different.
In particular, these sections are structured as **o2mrelation** objects that map elements
(the *ones*) to their subelement constituents (the *many*). For **polyhedral** topologies,
these constituents reside in an additional ``subelements`` section that specifies
the polyhedral faces in a format identical to ``elements`` in a **polygonal** schema.


Polygonal Topologies
^^^^^^^^^^^^^^^^^^^^^^^

The schema for a **polygonal** shape topology is as follows:

  * topologies/topo/coordset: "coords"
  * topologies/topo/type: “unstructured”
  * topologies/topo/elements: (o2mrelation object)
  * topologies/topo/elements/shape: "polygonal"
  * topologies/topo/elements/connectivity: (index array)

It's important to note that the ``elements/connectivity`` path defines the vertex
index sequences (relative to ``coordset``) for each element in the topology. These
vertex sequences must be arranged end-to-end (i.e. such that ``(v[i], v[i+1])``
defines an edge) relative to their container polygonal elements.

The following diagram illustrates a simple **polygonal** topology:

  .. code:: yaml

      #
      #    4--------5
      #    |`--     |
      # e1 |   `.   | e0
      #    |     --.|
      #    7--------6
      #

      topologies:
        topology:
          coordset: coords
          type: unstructured
          elements:
            shape: polygonal
            connectivity: [4, 6, 5, 7, 6, 4]
            sizes: [3, 3]
            offsets: [0, 3]


Polyhedral Topologies
^^^^^^^^^^^^^^^^^^^^^^^

The schema for a **polyhedral** shape topology is as follows:

  * topologies/topo/coordset: "coords"
  * topologies/topo/type: “unstructured”
  * topologies/topo/elements: (o2mrelation object)
  * topologies/topo/elements/shape: "polyhedral"
  * topologies/topo/elements/connectivity: (index array)
  * topologies/topo/subelements: (o2mrelation object)
  * topologies/topo/subelements/shape: (shape name)
  * topologies/topo/subelements/connectivity: (index array)

An important nuance to the structure of a **polyhedral** shape topology is that
the ``elements/connectivity`` path indexes into the ``subelements`` object to list
the *many* faces associated with each *one* polyhedron. Similarly, the
``subelements/connectivity`` path indexes into the ``coordset`` path to list the
*many* vertices associated with each *one* polyhedral face. There is no assumed
ordering for constituent polyhedral faces relative to their source polyhedra.

The following diagram illustrates a simple **polyhedral** topology:

  .. code:: yaml

      #
      #         0
      #        /|\
      #       / | \ <- e0
      #      /  |  \
      #     /_.-3-._\
      #    1.,  |  ,.4
      #     \ `'2'` /
      #      \  |  /
      # e1 -> \ | /
      #        \|/
      #         5
      #|

      topologies:
        topology:
          coordset: coords
          type: unstructured
          elements:
            shape: polyhedral
            connectivity: [0, 1, 2, 3, 4, 0, 5, 6, 7, 8]
            sizes: [5, 5]
            offsets: [0, 5]
          subelements:
            shape: polygonal
            connectivity: [1, 2, 4, 3, 1, 2, 0, 2, 4, 0, 4, 3, 0, 3, 1, 0, 1, 2, 5, 2, 4, 5, 4, 3, 5, 3, 1, 5]
            sizes: [4, 3, 3, 3, 3, 3, 3, 3, 3]
            offsets: [0, 4, 7, 10, 13, 16, 19, 22, 25]


Material Sets
++++++++++++++++++++

Materials Sets contain material name and volume fraction information defined over a specified mesh topology.

A material set is a type of **o2mrelation** that houses per-material, per-element volume fractions that are defined over a referenced source topology.
Each material set conforms to a schema variant based on:

 * The layout of its per-material buffers.
 * The indexing scheme used to associate volume fractions with topological elements.

The options for each of these variants are detailed in the following sections.


Material Set Buffer Variants
=================================

Each material set follows one of two variants based on the presented structure of its volume fractions.
These variants cover volume fractions presented in a single, unified buffer (called **uni-buffer** presentation) and in multiple, per-material buffers (called **multi-buffer** presentation).
Both of these variants and their corresponding schemas are outlined in the subsections below.


Uni-Buffer Material Sets
*********************************

A **uni-buffer** material set is one that presents all of its volume fraction data in a single data buffer.
In this case, the material set schema must include this volume fraction data buffer, a parallel buffer associating each volume with a material identifier, and an *Object* mapping of human-readable material names to each unique material identifier.
Additionally, the top-level of this schema is an **o2mrelation** that sources from the volume fraction/material identifier buffers and targets the material topology.
To conform to protocol, each ``matsets`` child of this type must be an *Object* that contains the following information:

   * matsets/matset/topology: "topo"
   * matsets/matset/material_map: (integer object)
   * matsets/matset/material_ids: (integer array)
   * matsets/matset/volume_fractions: (floating-point array)

The following diagram illustrates a simple **uni-buffer** material set example:

  .. code:: yaml

      #     z0       z1       z2
      # +--------+--------+--------+
      # | a0     | a1 ___/|        |
      # |___-----|----    |   b2   |
      # |     b0 |     b1 |        |
      # +--------+--------+--------+
      #

      matsets:
        matset:
          topology: topology
          volume_fractions:
            values: [0, a0, b2, b1, b0, 0, a1, 0]
            material_ids: [0, 1, 2, 2, 2, 0, 1, 0]
            material_map:
              a: 1
              b: 2
              c: 0
            sizes: [2, 2, 1]
            offsets: [0, 2, 4]
            indices: [1, 4, 6, 3, 2]


Multi-Buffer Material Sets
*********************************

A **multi-buffer** material set is a material set variant wherein the volume fraction data is split such that one buffer exists per material.
The schema for this variant dictates that each material be presented as an *Object* entry of the ``volume_fractions`` field with the material name as the entry key and the material volume fractions as the entry value.
Optionally, the value for each such entry can be specified as an **o2mrelation** instead of a flat array to enable greater specification flexibility.
To conform to protocol, each ``matsets`` child of this type must be an *Object* that contains the following information:

   * matsets/matset/topology: "topo"
   * matsets/matset/volume_fractions: (object)

The following diagram illustrates a simple **multi-buffer** material set example:

  .. code:: yaml

      #     z0       z1       z2
      # +--------+--------+--------+
      # | a0     | a1 ___/|        |
      # |___-----|----    |   b2   |
      # |     b0 |     b1 |        |
      # +--------+--------+--------+
      #

      matsets:
        matset:
          topology: topology
          volume_fractions:
            a:
              values: [0, 0, 0, a1, 0, a0]
              indices: [5, 3]
            b:
              values: [0, b0, b2, b1, 0]
              indices: [1, 3, 2]


Material Set Indexing Variants
=================================

Material sets can also vary in how volume fractions are associated with topological elements.
This associative variance leads to two additional schema variants: **element-dominant** (elements/volumes have the same ordering) and **material-dominant** (elements/volumes have independent orderings).
Both of these variants and their corresponding schemas are outlined in the subsections below.


Element-Dominant Material Sets
*********************************

In an **element-dominant** material set, the volume fraction data order matches the topological element order.
In other words, the volume fraction group at ``i`` (e.g. ``matset/volume_fractions/mat[i]``) contains the volume fraction data for topological element ``i``.
This variant is assumed in all material sets that don't have an ``element_ids`` child.

The following diagram illustrates a simple **element-dominant** material set example:

  .. code:: yaml

      #     z0       z1       z2
      # +--------+--------+--------+
      # | a0     | a1 ___/|\___ c2 |
      # |___-----|----    |    ----|
      # |     b0 |     b1 | b2     |
      # +--------+--------+--------+
      #

      matsets:
        matset:
          topology: topology
          volume_fractions:
            a: [a0, a1, 0]
            b: [b0, b1, b2]
            c: [0, 0, c2]


Material-Dominant Material Sets
*********************************

In a **material-dominant** material set, the orders for the volume fractions and topological elements are mismatched and need to be bridged via indirection arrays.
For these schemas, the ``element_ids`` field hosts these indirection arrays per material (with just one indirection array for uni-buffer material sets).
In explicit terms, the **material-dominant** volume fraction group at ``i`` (e.g. ``matset/volume_fractions/mat[i]``) contains the volume fraction data for the indirected topological element ``i`` (e.g. ``matset/element_ids/mat[i]``).
Complementary to the **element-dominant** variant, the **material-dominant** variant applies to all material sets that have an ``element_ids`` child.

The following diagram illustrates a simple **material-dominant** material set example:

  .. code:: yaml

      #     z0       z1       z2
      # +--------+--------+--------+
      # | a0     | a1 ___/|\___ c2 |
      # |___-----|----    |    ----|
      # |     b0 |     b1 | b2     |
      # +--------+--------+--------+
      #

      matsets:
        matset:
          topology: topology
          volume_fractions:
            a: [a0, a1]
            b: [b0, b1, b2]
            c: [c2]
          element_ids:
            a: [0, 1]
            b: [0, 1, 2]
            c: [2]


Fields
++++++++++++++++++++

Fields are used to hold simulation state arrays associated with a mesh topology and (optionally) a mesh material set.

Each field entry can define an **mcarray** of material-independent values and/or an **mcarray** of per-material values.
These data arrays must be specified alongside a source space, which specifies the space over which the field values are defined (i.e. a topology for material-independent values and a material set for material-dependent values).
Minimally, each field entry must specify one of these data sets, the source space for the data set, an association type (e.g. per-vertex, per-element, or per-grid-function-entity), and a volume scaling type (e.g. volume-dependent, volume-independent).
Thus, to conform to protocol, each entry under the ``fields`` section must be an *Object* that adheres to one of the following descriptions:

 * Material-Independent Fields:

   * fields/field/association: "vertex" | "element" 
   * fields/field/grid_function: (mfem-style finite element collection name) (replaces "association")
   * fields/field/volume_dependent: "true" | "false"
   * fields/field/topology: "topo"
   * fields/field/values: (mcarray)

 * Material-Dependent Fields:

   * fields/field/association: "vertex" | "element"
   * fields/field/grid_function: (mfem-style finite element collection name) (replaces "association")
   * fields/field/volume_dependent: "true" | "false"
   * fields/field/matset: "matset"
   * fields/field/matset_values: (mcarray)

 * Mixed Fields:

   * fields/field/association: "vertex" | "element"
   * fields/field/grid_function: (mfem-style finite element collection name) (replaces "association")
   * fields/field/volume_dependent: "true" | "false"
   * fields/field/topology: "topo"
   * fields/field/values: (mcarray)
   * fields/field/matset: "matset"
   * fields/field/matset_values: (mcarray)




Topology Association for Field Values
======================================

For implicit topologies, the field values are associated with the topology by fast varying logical dimensions starting with ``i``, then ``j``, then ``k``.

For explicit topologies, the field values are associated with the topology by assuming the order of the field values matches the order the elements are defined in the topology.


Species Sets
++++++++++++++++++++

Species Sets are a means of representing multi-dimensional per-material quantities, most commonly per-material substance fractions.

Individual Species Sets are entries in the ``specsets`` section of the Blueprint hierarchy, and these entries are formatted in much the same way as ``fields`` entries that describe per-material, multi-dimensional fields.
Just as with this class of ``fields`` entries, each ``specsets`` entry must specify the material set over which it is defined and enumerate its values within an **mcarray** that's organized in material-major and component-minor order.
Additionally, like ``field`` entries, each ``specsets`` item must indicate a volumetric scaling type (e.g. volume-dependent, volume-independent).
To put it in short, each entry in the ``specsets`` section of the Blueprint hierarchy must be an *Object* that follows this template:

 * specsets/specset/volume_dependent: "true" | "false"
 * specsets/specset/matset: "matset"
 * specsets/specset/matset_values: (mcarray)



Nesting Sets
++++++++++++++++++++

Nesting Sets are used to represent the nesting relationships between different domains in multi-domain mesh environments. Most commonly, this subset of the Blueprint specification is used for AMR (adaptive mesh refinement) meshes.

Each entry in the Nesting Sets section contains an independent set of nesting relationships between domains in the described mesh.
On an individual basis, a nesting set contains a source topology, an element association, and a list of nesting windows.
The windows for a particular nesting set describe the topological nesting pattern for a paired set of domains, which includes the ID of the partnered domain, the type of the partnered domain (parent or child), and the self-relative origin and dimensions of the nesting relationship.
The Blueprint schema for each entry in the ``nestsets`` section matches the following template:

   * nestsets/nestset/association: "vertex" | "element"
   * nestsets/nestset/topology: "topo"
   * nestsets/nestset/windows/window/domain_id: (integer)
   * nestsets/nestset/windows/window/domain_type: "parent" | "child"
   * nestsets/nestset/windows/window/ratio/{i, j, k}
   * nestsets/nestset/windows/window/origin/{i, j, k}
   * nestsets/nestset/windows/window/dims/{i, j, k}

Each domain that contains a Nesting Sets section must also update its State section to include the domain's global nesting level.
This additional requirement adds the follow constraint to the ``state`` section:

   * state/level_id: (integer)

.. note::
   The Nesting Sets section currently only supports nesting specifications for
   structured topologies. There are plans to extend this feature to support
   unstructured topologies in future versions of Conduit.


Adjacency Sets
++++++++++++++++++++

Adjacency Sets are used to outline the shared geometry between subsets of domains in multi-domain meshes.

Each entry in the Adjacency Sets section is meant to encapsulate a set of adjacency information shared between domains.
Each individual adjacency set contains a source topology, an element association, and a list of adjacency groups.
An adjacency set's contained groups describe adjacency information shared between subsets of domains, which is represented by a subset of adjacent neighbor domains IDs and a list of shared element IDs.
The fully-defined Blueprint schema for the ``adjsets`` entries looks like the following:

   * adjsets/adjset/association: "vertex" | "element"
   * adjsets/adjset/topology: "topo"
   * adjsets/adjset/groups/group/neighbors: (integer array)
   * adjsets/adjset/groups/group/values: (integer array)



State
++++++++++++++++++++

Optional state information is used to provide metadata about the mesh. While the mesh blueprint is focused on describing a single domain of a domain decomposed mesh, the state info can be used to identify a specific mesh domain in the context of a domain decomposed mesh.

To conform, the ``state`` entry must be an *Object* and can have the following optional entries:

   * state/time: (number)
   * state/cycle: (number)
   * state/domain_id: (integer)

.. _examples:


Mesh Blueprint Examples
~~~~~~~~~~~~~~~~~~~~~~~~~


The C++ ``conduit::blueprint::mesh::examples`` namespace and the Python ``conduit.blueprint.mesh.examples`` module provide
functions that generate example Mesh Blueprint data. For details on how to write these data sets to files, see the unit
tests that exercise these examples in ``src/tests/blueprint/t_blueprint_mesh_examples.cpp`` and the
`mesh output <Outputting Meshes for Visualization_>`_ example below. This section outlines the examples that demonstrate
the most commonly used mesh schemas.

basic
+++++++++

The simplest of the mesh examples, ``basic()``, generates an homogenous example mesh with a configurable element
representation/type (see the ``mesh_type`` table below) spanned by a single scalar field that contains a unique
identifier for each mesh element. The function that needs to be called to generate an example of this type has the
following signature:

.. code:: cpp

    conduit::blueprint::mesh::examples::basic(const std::string &mesh_type, // element type/dimensionality
                                              index_t nx,                   // number of grid points along x
                                              index_t ny,                   // number of grid points along y
                                              index_t nz,                   // number of grid points along z (3d only)
                                              Node &res);                   // result container



The element representation, type, and dimensionality are all configured through the ``mesh_type`` argument. The
supported values for this parameter and their corresponding effects are outlined in the table below:

+--------------------------------+--------------------+-------------------+-------------------+------------------+
| **Mesh Type**                  | **Dimensionality** | **Coordset Type** | **Topology Type** | **Element Type** |
+--------------------------------+--------------------+-------------------+-------------------+------------------+
| `uniform <Uniform_>`_          | 2d/3d              | implicit          | implicit          | quad/hex         |
+--------------------------------+--------------------+-------------------+-------------------+------------------+
| `rectilinear <Rectilinear_>`_  | 2d/3d              | implicit          | implicit          | quad/hex         |
+--------------------------------+--------------------+-------------------+-------------------+------------------+
| `structured <Structured_>`_    | 2d/3d              | explicit          | implicit          | quad/hex         |
+--------------------------------+--------------------+-------------------+-------------------+------------------+
| `tris <Tris_>`_                | 2d                 | explicit          | explicit          | tri              |
+--------------------------------+--------------------+-------------------+-------------------+------------------+
| `quads <Quads_>`_              | 2d                 | explicit          | explicit          | quad             |
+--------------------------------+--------------------+-------------------+-------------------+------------------+
| `polygons <Polygons_>`_        | 2d                 | explicit          | explicit          | polygon          |
+--------------------------------+--------------------+-------------------+-------------------+------------------+
| `tets <Tets_>`_                | 3d                 | explicit          | explicit          | tet              |
+--------------------------------+--------------------+-------------------+-------------------+------------------+
| `hexs <Hexs_>`_                | 3d                 | explicit          | explicit          | hex              |
+--------------------------------+--------------------+-------------------+-------------------+------------------+
| `polyhedra <Polyhedra_>`_      | 3d                 | explicit          | explicit          | polyhedron       |
+--------------------------------+--------------------+-------------------+-------------------+------------------+

The remainder of this section demonstrates each of the different ``basic()`` mesh types, outlining
each type with a simple example that (1) presents the generating call, (2) shows the results of the
call in Blueprint schema form, and (3) displays the corresponding graphical rendering of this schema.

Uniform
====================================

* **Usage Example**

.. literalinclude:: ../../tests/docs/t_conduit_docs_blueprint_demos.cpp
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_uniform")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_uniform")
   :language: cpp
   :dedent: 4

* **Result**

.. literalinclude:: t_conduit_docs_blueprint_demos_out.txt
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_uniform")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_uniform")
   :language: yaml

* **Visual**

.. figure:: basic_hex_2d_render.png
    :width: 400px
    :align: center

    Pseudocolor plot of ``basic`` (mesh type 'uniform')

Rectilinear
====================================

* **Usage Example**

.. literalinclude:: ../../tests/docs/t_conduit_docs_blueprint_demos.cpp
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_rectilinear")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_rectilinear")
   :language: cpp
   :dedent: 4

* **Result**

.. literalinclude:: t_conduit_docs_blueprint_demos_out.txt
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_rectilinear")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_rectilinear")
   :language: yaml

* **Visual**

.. figure:: basic_hex_2d_render.png
    :width: 400px
    :align: center

    Pseudocolor plot of ``basic`` (mesh type 'rectilinear')

Structured
====================================

* **Usage Example**

.. literalinclude:: ../../tests/docs/t_conduit_docs_blueprint_demos.cpp
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_structured")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_structured")
   :language: cpp
   :dedent: 4

* **Result**

.. literalinclude:: t_conduit_docs_blueprint_demos_out.txt
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_structured")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_structured")
   :language: yaml

* **Visual**

.. figure:: basic_hex_2d_render.png
    :width: 400px
    :align: center

    Pseudocolor plot of ``basic`` (mesh type 'structured')

Tris
====================================

* **Usage Example**

.. literalinclude:: ../../tests/docs/t_conduit_docs_blueprint_demos.cpp
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_tris")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_tris")
   :language: cpp
   :dedent: 4

* **Result**

.. literalinclude:: t_conduit_docs_blueprint_demos_out.txt
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_tris")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_tris")
   :language: yaml

* **Visual**

.. figure:: basic_tet_2d_render.png
    :width: 400px
    :align: center

    Pseudocolor plot of ``basic`` (mesh type 'tris')

Quads
====================================

* **Usage Example**

.. literalinclude:: ../../tests/docs/t_conduit_docs_blueprint_demos.cpp
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_quads")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_quads")
   :language: cpp
   :dedent: 4

* **Result**

.. literalinclude:: t_conduit_docs_blueprint_demos_out.txt
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_quads")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_quads")
   :language: yaml

* **Visual**

.. figure:: basic_hex_2d_render.png
    :width: 400px
    :align: center

    Pseudocolor plot of ``basic`` (mesh type 'quads')

Polygons
====================================

* **Usage Example**

.. literalinclude:: ../../tests/docs/t_conduit_docs_blueprint_demos.cpp
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_polygons")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_polygons")
   :language: cpp
   :dedent: 4

* **Result**

.. literalinclude:: t_conduit_docs_blueprint_demos_out.txt
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_polygons")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_polygons")
   :language: yaml

* **Visual**

.. figure:: basic_hex_2d_render.png
    :width: 400px
    :align: center

    Pseudocolor plot of ``basic`` (mesh type 'polygons')

Tets
====================================

* **Usage Example**

.. literalinclude:: ../../tests/docs/t_conduit_docs_blueprint_demos.cpp
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_tets")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_tets")
   :language: cpp
   :dedent: 4

* **Result**

.. literalinclude:: t_conduit_docs_blueprint_demos_out.txt
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_tets")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_tets")
   :language: yaml

* **Visual**

.. figure:: basic_tet_3d_render.png
    :width: 400px
    :align: center

    Pseudocolor plot of ``basic`` (mesh type 'tets')

Hexs
====================================

* **Usage Example**

.. literalinclude:: ../../tests/docs/t_conduit_docs_blueprint_demos.cpp
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_hexs")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_hexs")
   :language: cpp
   :dedent: 4

* **Result**

.. literalinclude:: t_conduit_docs_blueprint_demos_out.txt
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_hexs")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_hexs")
   :language: yaml

* **Visual**

.. figure:: basic_hex_3d_render.png
    :width: 400px
    :align: center

    Pseudocolor plot of ``basic`` (mesh type 'hexs')

Polyhedra
====================================

* **Usage Example**

.. literalinclude:: ../../tests/docs/t_conduit_docs_blueprint_demos.cpp
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_polyhedra")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_polyhedra")
   :language: cpp
   :dedent: 4

* **Result**

.. literalinclude:: t_conduit_docs_blueprint_demos_out.txt
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_polyhedra")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_polyhedra")
   :language: yaml

* **Visual**

.. figure:: basic_hex_3d_render.png
    :width: 400px
    :align: center

    Pseudocolor plot of ``basic`` (mesh type 'polyhedra')


braid
++++++

.. figure:: braid_render.png
    :width: 400px
    :align: center

    Pseudocolor plot of a 3D braid example ``braid`` field
    
The ``braid()`` generates example meshes that cover the range of coordinate sets and topologies supported by the Mesh Blueprint.

The example datasets include a vertex-centered scalar field ``braid``, an element-centered scalar field ``radial`` and
a vertex-centered vector field ``vel``.

.. code:: cpp

    conduit::blueprint::mesh::examples::braid(const std::string &mesh_type,
                                              index_t nx,
                                              index_t ny,
                                              index_t nz,
                                              Node &res);

Here is a list of valid strings for the ``mesh_type`` argument:

+---------------+-----------------------------------------------+
| **Mesh Type** | **Description**                               |
+---------------+-----------------------------------------------+
| uniform       | 2d or 3d uniform grid                         |
|               | (implicit coords, implicit topology)          |
+---------------+-----------------------------------------------+
| rectilinear   | 2d or 3d rectilinear grid                     |
|               | (implicit coords, implicit topology)          |
+---------------+-----------------------------------------------+
| structured    | 2d or 3d structured grid                      |
|               | (explicit coords, implicit topology)          |
+---------------+-----------------------------------------------+
| point         | 2d or 3d unstructured mesh of point elements  |
|               | (explicit coords, explicit topology)          |
+---------------+-----------------------------------------------+
| lines         | 2d or 3d unstructured mesh of line elements   |
|               | (explicit coords, explicit topology)          |
+---------------+-----------------------------------------------+
| tris          | 2d unstructured mesh of triangle elements     |
|               | (explicit coords, explicit topology)          |
+---------------+-----------------------------------------------+
| quads         | 2d unstructured mesh of quadrilateral elements|
|               | (explicit coords, explicit topology)          |
+---------------+-----------------------------------------------+
| tets          | 3d unstructured mesh of tetrahedral elements  |
|               | (explicit coords, explicit topology)          |
+---------------+-----------------------------------------------+
| hexs          | 3d unstructured mesh of hexahedral elements   |
|               | (explicit coords, explicit topology)          |
+---------------+-----------------------------------------------+

``nx``, ``ny``, ``nz`` specify the number of elements in the x, y, and z directions.

``nz`` is ignored for 2d-only examples.

The resulting data is placed the Node ``res``, which is passed in via reference.

spiral
+++++++

.. figure:: spiral_render.png
    :width: 400px
    :align: center

    Pseudocolor and Contour plots of the spiral example ``dist`` field.

The ``sprial()`` function generates a multi-domain mesh composed of 2D square 
domains with the area of successive fibonacci numbers. The result estimates the 
`Golden spiral <https://en.wikipedia.org/wiki/Golden_spiral>`_.

The example dataset provides a vertex-centered scalar field ``dist`` that estimates the distance from 
each vertex to the Golden spiral.

.. code:: cpp

    conduit::blueprint::mesh::examples::spiral(conduit::index_t ndomains,
                                               Node &res);


``ndomains`` specifies the number of domains to generate, which is also the number of entries from fibonacci sequence used.

The resulting data is placed the Node ``res``, which is passed in via reference.

julia
+++++++


.. figure:: julia_render.png
    :width: 350px
    :align: center

    Pseudocolor plot of the julia example ``iter`` field
    

The ``julia()`` function creates a uniform grid that visualizes
`Julia set fractals <https://en.wikipedia.org/wiki/Julia_set>`_.


The example dataset provides an element-centered scalar field ``iter`` that represents the number of iterations
for each point tested or zero if not found in the set.
 

.. code:: cpp

    conduit::blueprint::mesh::examples::julia(index_t nx,
                                              index_t ny,
                                              float64 x_min,
                                              float64 x_max,
                                              float64 y_min,
                                              float64 y_max,
                                              float64 c_re,
                                              float64 c_im,
                                              Node &res);

``nx``, ``ny`` specify the number of elements in the x and y directions.

``x_min``, ``x_max``, ``y_min``, ``y_max`` specify the x and y extents.

``c_re``, ``c_im`` specify real and complex parts of the constant used.

The resulting data is placed the Node ``res``, which is passed in via reference.

julia amr examples
++++++++++++++++++++

We also provide examples that represent the julia set using AMR meshes. These functions provide concrete examples of the Mesh Blueprint `nestset` protocol for patch-based AMR meshes.


.. figure:: julia_nestsets_simple.png
    :width: 350px
    :align: center

    Pseudocolor, Mesh, and Domain Boundary plots of the julia_nestsets_simple example.


.. code:: cpp

    conduit::blueprint::mesh::examples::julia_nestsets_simple(float64 x_min,
                                                              float64 x_max,
                                                              float64 y_min,
                                                              float64 y_max,
                                                              float64 c_re,
                                                              float64 c_im,
                                                              Node &res);

`julia_nestsets_simple` provides a basic AMR example with two levels and one
parent/child nesting relationship.

``x_min``, ``x_max``, ``y_min``, ``y_max`` specify the x and y extents.

``c_re``, ``c_im`` specify real and complex parts of the constant used.

The resulting data is placed the Node ``res``, which is passed in via reference.


.. figure:: julia_nestsets_complex.png
    :width: 350px
    :align: center

    Pseudocolor, Mesh, and Domain Boundary plots of the julia_nestsets_complex example.


.. code:: cpp

    conduit::blueprint::mesh::examples::julia_nestsets_complex(index_t nx,
                                                               index_t ny,
                                                               float64 x_min,
                                                               float64 x_max,
                                                               float64 y_min,
                                                               float64 y_max,
                                                               float64 c_re,
                                                               float64 c_im,
                                                               index_t levels,
                                                               Node &res);

`julia_nestsets_complex` provides an AMR example that refines the mesh
using more resolution in complex areas.


``nx``, ``ny`` specify the number of elements in the x and y directions.

``x_min``, ``x_max``, ``y_min``, ``y_max`` specify the x and y extents.

``c_re``, ``c_im`` specify real and complex parts of the constant used.

``levels`` specifies the number of refinement levels to use.

The resulting data is placed the Node ``res``, which is passed in via reference.



venn
+++++++


.. figure:: venn_example.png
    :width: 350px
    :align: center

    Pseudocolor plot of the venn example ``overlap`` field


The ``venn()`` function creates meshes that use three overlapping circle regions, demonstrating different ways to encode volume fraction based multi-material fields. The volume fractions are provided as both standard fields and using the Material sets (``matsets``) Blueprint. It also creates other fields related to overlap pattern.

.. code:: cpp

    conduit::blueprint::mesh::examples::venn(const std::string &matset_type,
                                             index_t nx,
                                             index_t ny,
                                             float64 radius,
                                             Node &res);


``matset_type`` specifies the style of matset generated by the example.

Here is a list of valid strings for the ``matset_type`` argument:

.. list-table:: 
   :widths: 10 15
   :header-rows: 1

   * - **Matset Type**
     - **Description**

   * - full
     - non-sparse volume fractions and matset values

   * - sparse_by_material
     - sparse (material dominant) volume fractions and matset values

   * - sparse_by_element
     - sparse (element dominant) volume fractions and matset values

``nx``, ``ny`` specify the number of elements in the x and y directions.

``radius`` specifies the radius of the three circles.

The resulting data is placed the Node ``res``, which is passed in via reference.


polytess
++++++++++

.. figure:: polytess_render.png
    :width: 400px
    :align: center

    Pseudocolor plot of the polytess example ``level`` field.

The ``polytess()`` function generates a polygonal tesselation in the 2D
plane comprised of octogons and squares (known formally as a `two-color
truncated square tiling <https://en.wikipedia.org/wiki/Truncated_square_tiling>`_).

The scalar element-centered field ``level`` defined in the result mesh associates each element with its
topological distance from the center of the tesselation.

.. code:: cpp

    conduit::blueprint::mesh::examples::polytess(index_t nlevels,
                                                 Node &res);


``nlevels`` specifies the number of tesselation levels/layers to generate. If this value is specified
as 1 or less, only the central tesselation level (i.e. the octogon in the center of the geometry) will
be generated in the result.

The resulting data is placed the Node ``res``, which is passed in via reference.

miscellaneous
++++++++++++++

This section doesn't overview any specific example in the ``conduit::blueprint::mesh::examples`` namespace,
but rather provides a few additional code samples to help with various common tasks. Each subsection covers
a specific task and presents how it can be accomplished using a function or set of functions in Conduit
and/or the Mesh Blueprint library.

Outputting Meshes for Visualization
====================================

Suppose that you have an arbitrary Blueprint mesh that you want to output from a running code and
subsequently visualize using a visualization tool (e.g. `VisIt <https://wci.llnl.gov/simulation/computer-codes/visit>`_).
Provided that your mesh is sufficiently simple (see the note at the end of this section for details),
you can output your mesh using one of the following ``conduit::relay`` library functions:

.. code:: cpp

    // saves the given mesh to disk at the given path (using the extension
    // suffix in the path to inform the output data protocol)
    conduit::relay::io_blueprint::save(const conduit::Node &mesh,
                                       const std::string &path);

    // saves the given mesh to disk at the given path with the given explicit
    // output data protocol (e.g. "json", "hdf5")
    conduit::relay::io_blueprint::save(const conduit::Node &mesh,
                                       const std::string &path,
                                       const std::string &protocol);

It's important to note that both of these functions expect the given path to have
a valid extension to properly output results. The valid extensions for these
functions are as follows:

- ``.blueprint_root`` (JSON Extension)
- ``.blueprint_root_hdf5`` (HDF5 Extension)

Files output from these functions can be opened and subsequently visualized
directly using `VisIt <https://wci.llnl.gov/simulation/computer-codes/visit>`_.

.. note::
   This automatic index generation and save functionality is under development. 
   It handles most basic cases, but only supports ``json`` and ``hdf5`` output
   protocols and has limited multi-domain support. We are working on API changes
   and a more robust capability for future versions of Conduit.

.. _detailed_uniform_example:

Detailed Uniform Example
====================================

This snippet provides a complete C++ example that demonstrates:

  * Describing a uniform mesh in a Conduit tree
  * Verifying the tree conforms to the Mesh Blueprint
  * Saving the result to a JSON file that VisIt can open

.. literalinclude:: ../../tests/docs/t_conduit_docs_blueprint_demos.cpp
   :start-after: BEGIN_EXAMPLE("blueprint_demo_basic_uniform_detailed")
   :end-before:  END_EXAMPLE("blueprint_demo_basic_uniform_detailed")
   :language: cpp
   :dedent: 4

Expressions (Derived Fields)
============================

An *expression* is a mathemtical formula which defines a new field in terms of other fields and/or
other expressions. Expressions are specified in the ``expressions`` section of the Blueprint
protocol. The ``expressions`` section is optional. When it exists, it is a peer to the ``fields`` section.
It is a list of *Objects* of the form:

* expressions/expression/number_of_components
* expressions/expression/topology
* expressions/expression/definition

The ``number_of_components`` and ``topology`` entries are identical to their meaning as
entries in the ``fields`` section.

The ``definition`` entry is string valued and holds the expression (e.g. *mathemtical formula*) defining
how the new field is computed. Blueprint does not interpret this string. It simply passes it along for
downstream consumers that have the ability to interpret the string and perform the associated operations
to compute the expression.

If the expected consumer is `VisIt <https://visit.llnl.gov>`_, data producers may wish to consult the
`Expressions chapter of the VisIt user's manual <https://visit-sphinx-github-user-manual.readthedocs.io/en/develop/gui_manual/Quantitative/Expressions.html#built-in-expressions>`_.
In addition, data producers should *escape* all names of fields or expressions by bracketing
them in ``<`` and ``>`` characters. An example expressions entry in the index is

  .. code:: json

      "fields":
      {
        "braid":
        {
          ...
        },
        "radial":
        {
          ...
        },
      "expressions":
      {
        "scalar_expr":
        {
          "number_of_components": 1,
          "topology": "mesh",
          "definition": "<vector_expr>[1]"
        },
        "vector_expr":
        {
          "number_of_components": 2,
          "topology": "mesh",
          "definition": "{<braid>,recenter(<radial>,\"nodal\")}"
        }
      }

.. Properties and Transforms
.. ---------------------------




