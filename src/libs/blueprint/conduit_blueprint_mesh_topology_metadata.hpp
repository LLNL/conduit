// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_topology_metadata.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_TOPOLOGY_METADATA_HPP
#define CONDUIT_BLUEPRINT_MESH_TOPOLOGY_METADATA_HPP

//-----------------------------------------------------------------------------
// std includes
//-----------------------------------------------------------------------------
#include <map>
#include <set>
#include <string>
#include <vector>

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_exports.h"
#include "conduit_range_vector.hpp"
#include "conduit_vector_view.hpp"
#include "conduit_blueprint_mesh_utils.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils --
//-----------------------------------------------------------------------------
namespace utils
{

// Base class for TopologyMetadata classes - it provides IndexType.
class CONDUIT_BLUEPRINT_API TopologyMetadataBase
{
public:
    enum IndexType { GLOBAL = 0, LOCAL = 1 };
    // The 'IndexType' indicates the index space to be used when referring to
    // entities within this topological cascade.
};

//---------------------------------------------------------------------------//
/**
 The TopologicalMetadata takes an input topology and decomposes each element
 according to a topological cascade to produce connectivity from the input
 topology's level down to the points. In 3D, this gives rise to:

   3D cells, 2D faces, 1D edges, 0D points
 
 The set of 2D faces is a unique set of faces shared among the input 3D elements
 and mappings are produced that allow us to access for each element which faces
 it "owns" as children. When 2 elements share a face, they will each count it
 as a child in their connectivity. The 2D faces consist of a set of edges
 and these are made unique as well and are available in the mapping.

 An example is helpful to understand the concept of the associations.

 4 3D Hex topology (points are labelled):

             |--------------------------- element 2 (3,4,7,6,12,13,16,15)
             |
      +-------------+-------------+
     /|6           /|7           /|8
    / |           / |           / |
   /  |          /  |          /  |
  /   |         /   |         /   |  <--- element 3 (4,5,8,7,13,14,17,16)
 +-------------+-------------+    |
 |15  |        |16  |        |17  |
 |    +- - - - | - -+ - - - -|- - +
 |   /|3       |   /|4       |   /|5
 |  / |        |  / |        |  / |
 | /  |        | /  |        | /  |
 |/   |        |/   |        |/   |
 +-------------+-------------+    |
 |12  |        |13  |        |14  |
 |    +- - - - | - -+ - - - -|- - +
 |   / 0       |   / 1       |   / 2
 |  /          |  /          |  /
 | /           | /           | /  <------ element 1 (1,2,5,4,10,11,14,13)
 |/            |/            |/
 +-------------+-------------+
  9             10            11 
        ^              
        |-------------------------------- element 0  (0,1,4,3,9,10,13,12)


 When the topological cascade is done, each element is turned into a set of faces
 and these are made unique and elements that abut will share the common face. So,
 elements 0,1 will share the quad face (10,1,4,13). The unique set of faces in all
 elements will form the metadata's dimension 2 connectivity:

 face 0: 0, 3, 4, 1
 face 1: 0, 1, 10, 9
 face 2: 1, 4, 13, 10  <----- face 2 is made of these points. face 2 is shared
 face 3: 4, 3, 12, 13         by elements 0, 1.
 face 4: 3, 0, 9, 12
 face 5: 9, 10, 13, 12
 face 6: 1, 4, 5, 2
 face 7: 1, 2, 11
 face 8: 10, 2, 5, 14
 face 9: 11, 5, 4, 13
 face 10: 14, 10, 11, 14
 ...

 In the same way, the entire collection of faces can be thought of as a unique
 set of edges with their own numbering.

 The entity associations contain the information of how these elements,faces,
 edges,points are connected.

 The global entity association for dimension 3 to dimension 2, which we'll write
 as G(3,2), indicates the relationship between entities of dimension 3 to the
 entities of dimension 2. Since the dimension 3 elements are made of the
 dimension 2 elements (much as PH cells are expressed in Blueprint), the G(3,2)
 association will consist of the indices of the faces that make up element 0,
 followed by the indices of the faces that make up element 1, and so on.

           element 0 faces   element 1 faces ...
 G(3,2) = {0, 1, 2, 3, 4, 5}{6, 7, 8, 9, 2, 10}...
                    |
                    |
                    face 3 in unique face list (topo[2] connectivity)

 There are other GLOBAL entity associations G(e,a) too. G(3,3) is the
 relationship of elements to elements, which is self so any input to the map
 is also the output. The G(3,1) map represents the indices of the 1D edges
 that make up the 3D element. The G(3,0) map represents all of the points
 contained by the 3D element.

 G(e,a) is shorthand for "Global(entity_dim, association_dim)".

 When e==a, the relationship is self.
 When e>a, entity e contains/references entity a.
 When a>e, entity a is referenced by parents of dimension e.

 In the above example for G(3,2), we can see that face 2 is shared by elements 0
 and 1. If we want to know which edges make up face 2, we can consult G(2,1).

           face 0      face 1      face 2      face 3
 G(2,1) = {0, 1, 2, 3}{3, 4, 5, 6}{2, 7, 8, 4}{1, 9, 10, 7}...
                                   |
                                   |
                                   face 2 in unique edge list (topo[1] connectivity)

 If we want to know which points make up the edges for face 2, consult G(1,0).
 This is the map of edges to points (also top_dims[1] connectivity).

           edge number
           |
           0    1    2    3    4     5     6    7     8
 G(1,0) = {0,3}{3,4}{4,1}{1,0}{1,10}{10,9}{9,0}{4,13}{13,10}...
                     |         |                |     |
                     |         |                |     |
                     ----------------------------------
                     |
                     edges that are in face 2. 


 What does G(e,a) mean?

                               Association (a)
  ---------------------------------------------------------------------------
     | 3              | 2               | 1               | 0
  ---------------------------------------------------------------------------
E  3 |                | face ids in     | edge ids for    | point ids in
n    |    self        | element i       | element i       | element i
t    |                | BBC*            |                 | dim[3] connectivity
i  --|-----------------------------------------------------------------------
t  2 | element ids    |                 | edge ids in     | point ids in 
y    | that contain   |      self       | face i          | face i
     | face i         |                 | BBC *           | dim[2] connectivity
e  --|-----------------------------------------------------------------------
   1 | element ids    | face ids that   |                 | point ids in
     | that contain   | contain edge i  |      self       | edge i
     | edge i         |                 |                 | dim[1] connectivity
   --|-----------------------------------------------------------------------
   0 | element ids    | face ids that   | edge ids that   |
     | that contain   | contain point i | contain point i |     self
     | point i        |                 |                 |
  ---------------------------------------------------------------------------

 *BBC = built by cascade

 There is another mapping concept called LOCAL mapping L(e,a) where entity
 gets its own unique index ordering. The local maps look essentially like an
 increasing series of integers, though the number of repetitions for an
 entity depends on e,a.

 Example:
 L(3,2) = {0,1,2,3,4,5},{6,7,8,9,10,11},...

               6 faces 
           ----------------
           |              |
 L(2,3) = {0}{0}{0}{0}{0}{0}{1}{1}{1}{1}{1}{1}...

 */
class CONDUIT_BLUEPRINT_API TopologyMetadata : public TopologyMetadataBase
{
public:
    //-----------------------------------------------------------------------
    /**
     @brief Legacy constructor, which builds all of the topology levels in the shape
            cascade as well as all associations.

     @param topology     The input topology node.
     @param coordset     The input coordset associated with the topology.
     */
    TopologyMetadata(const conduit::Node &topology, const conduit::Node &coordset);

    //-----------------------------------------------------------------------
    /**
     @brief Constructor for the NewTopologyMetadata class. This constructor
            lets the caller be more selective about which topology levels and 
            associations are created, possibly saving time.

     @param topology     The input topology node.
     @param coordset     The input coordset associated with the topology.
     @param lowest_dim   The lowest level of shape cascade that we're interested in.
                         If we only want faces, for example, then we don't need to
                         refine to edges or points.
     @param desired      A vector of (entity_dim,assoc_dim) pairs that indicate
                         the associations that will be requested by the client.
     */
    TopologyMetadata(const conduit::Node &topology,
                     const conduit::Node &coordset,
                     size_t lowest_dim,
                     const std::vector<std::pair<size_t,size_t> > &desired_maps);

    //-----------------------------------------------------------------------
    virtual ~TopologyMetadata();

    //-----------------------------------------------------------------------
    /**
     @brief Get the highest shape dimension.
     @return The highest shape dimension.
     */
    int dimension() const;

    //-----------------------------------------------------------------------
    /**
     @brief Get the topology for the dimension.
     @param dim The dimension.
     @return The topology node.
     */
    const conduit::Node &get_topology(size_t dim);

    //-----------------------------------------------------------------------
    /**
     @brief Get the topology for the dimension in the provided node.
     @param dim The dimension.
     @param dest The destination node.
     */
    void get_topology(size_t dim, conduit::Node &dest);

    //-----------------------------------------------------------------------
    /**
     @brief Get the length for a topology. Any topologies that were not
            produced will have length 0.
     @return The topology length.
     */
    index_t get_topology_length(size_t dim) const;

    //-----------------------------------------------------------------------
    /**
     @brief Get a global association for a particular entity_id in the entity_dim
            dimension. This lets us ask a question like: which face ids are in
            element 5? Or, which faces contain edge 12?

     @param entity_id The entity id whose information we want to obtain. This
                      value must be valid for the entity dimension in question.
     @param entity_dim The dimension of the entity whose information we want.
     @param assoc_dim  The dimension of the association we want.

     Example: Which faces are in element 5?
              get_global_association(5,  // element 5
                                     3,  // elements are 3 dimensional
                                     2); // faces are 2 dimensional

     Example 2: Which faces contain edge 2?
              get_global_association(2,  // edge 2
                                     1,  // edges are 1 dimensional
                                     2); // faces are 2 dimensional

     @note The data are returned as a vector_view, which is an object that acts
           like a vector but does not have to copy bulk data. The data it points
           to for a given return are in the associations. We use vector_view
           instead of DataArray in order provide a std::vector-like container
           (with iterators, etc) that points to externally allocated data.
     */
    conduit::vector_view<index_t>
    get_global_association(index_t entity_id, index_t entity_dim, index_t assoc_dim) const;

    //-----------------------------------------------------------------------
    /**
     @brief Get a local association for a particular entity_id in the entity_dim
            dimension.

     @param entity_id The entity id whose information we want to obtain. This
                      value must be valid for the entity dimension in question.
     @param entity_dim The dimension of the entity whose information we want.
     @param assoc_dim  The dimension of the association we want.

     @note The data are returned as a range_vector, which is a class that
           represents a range implicitly and provides std::vector-like
           functionality and iterators.
    */
    conduit::range_vector<index_t>
    get_local_association(index_t entity_id, index_t entity_dim, index_t assoc_dim) const;

    //-----------------------------------------------------------------------
    /**
     @brief Get whether the association e,a was requested when the object
            was initialized. If so, the association will exist. Otherwise,
            the association does not exist.

     @param entity_dim The starting dimension.
     @param assoc_dim  The destination dimension.

     @return True if the association exists; False otherwise.
     */
    bool association_requested(index_t entity_dim, index_t assoc_dim) const;

    //-----------------------------------------------------------------------
    /**
     @brief Return all of the data for an association by COPYING it into the 
            provided Conduit node.

     @param type Whether we want GLOBAL or LOCAL data.
     @param src_dim The source dimension of the desired association.
     @param dst_dim The destination dimension of the desired association.
     @param[out] map_node The Conduit node that will contain the copied association data.

     @note This method guarantees that all bulk arrays for values, sizes, and
           offsets will be index_t.

     @return True if the map exists(was requested); False otherwise.
     */
    bool get_dim_map(IndexType type, index_t src_dim, index_t dst_dim, Node &map_node) const;

    //-----------------------------------------------------------------------
    /**
     @brief Gets the length of the topology as specified by dimension. If
            dim is -1 then the length of all topologies are summed.

     @param dim The dimension whose length we want. Or if -1 then all
                dimensions are assumed.
     @return The topology length for the requested dimension(s).
     */
    index_t get_length(index_t dim = -1) const;

    //-----------------------------------------------------------------------
    /**
     @brief The the preferred integer storage data type.
     @return The preferred integer storage data type.
     */
    const DataType &get_int_dtype() const;

    //-----------------------------------------------------------------------
    /**
     @brief The the preferred float storage data type.
     @return The preferred float storage data type.
     */
    const DataType &get_float_dtype() const;

    //-----------------------------------------------------------------------
    /**
     @brief Gets the total number of embeddings for each entity at the top level
            to the embedding level.
     @param entity_dim The entity dimension.
     @param embed_dim The embedding dimension.
     @return The total number of embeddings.
     */
    index_t get_embed_length(index_t entity_dim, index_t embed_dim) const;

    //-----------------------------------------------------------------------
    /**
     @brief Make a representation of the metadata in the supplied node.
     @param rep The node that will contain the representation.
     */
    void make_node(conduit::Node &rep) const;

    //-----------------------------------------------------------------------
    /**
     @brief Turn the metadata into a JSON string and return it.
     @return A string that contains a JSON representation of the metadata.
     */
    std::string to_json() const;

    //-----------------------------------------------------------------------
    /**
     @brief Return a vector that lets us map local entity ids to global ids.
     @param dim The dimension of the map we want.
     @return The map for the requested dimension.
     */
    const std::vector<index_t> &get_local_to_global_map(index_t dim) const;

private:
    class Implementation;
    Implementation *impl;
};


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::reference --
//-----------------------------------------------------------------------------
namespace reference
{

// This is the original TopologyMetadata, which is remaining as a reference
// implementation.

class CONDUIT_BLUEPRINT_API TopologyMetadata : public TopologyMetadataBase
{
public:
    // The 'IndexType' indicates the index space to be used when referring to
    // entities within this topological cascade. The types have the following
    // meanings:
    //
    // - GLOBAL: The unique index for the entity relative to the entire topology.
    //   Though a point may be shared by many lines/faces/cells, it will only
    //   have one global index. This is most commonly used for entity identification.
    // - LOCAL: The index of the entity relative to a cascade context. A point
    //   will have one local index for each line/face/cell that it participates
    //   in along the cascade. This is most commonly used to determine entity orientation.
    //
    // To clarify, consider the following example, with the following local and
    // global identifiers:
    //
    // - GLOBAL Scheme: Each entity has 1 unique identifier depending on FIFO
    //   cascade iteration:
    //
    //   p3               p4              p5
    //   +----------------+----------------+
    //   |       l2       |       l6       |
    //   |                |                |
    //   |                |                |
    //   |l3     f0       |l1     f1     l5|
    //   |                |                |
    //   |                |                |
    //   |       l0       |       l4       |
    //   +----------------+----------------+
    //   p0               p1              p2
    //
    // - LOCAL Scheme: Each entity has an identifier for each occurence within
    //   the cascade:
    //
    //    p5            p4 p13          p12
    //   +----------------+----------------+
    //   |p6     l2     p3|p14    l6    p11|
    //   |                |                |
    //   |                |                |
    //   |l3     f0     l1|l7     f1     l5|
    //   |                |                |
    //   |                |                |
    //   |p7     l0     p2|p15    l4    p10|
    //   +----------------+----------------+
    //    p0            p1 p8            p9
    //

    TopologyMetadata(const conduit::Node &topology, const conduit::Node &coordset);

    void add_entity_assoc(IndexType type, index_t e0_id, index_t e0_dim, index_t e1_id, index_t e1_dim);

    std::vector<index_t> &get_entity_assocs(IndexType type, index_t entity_id, index_t entity_dim, index_t assoc_dim);
    const std::vector<index_t> &get_entity_assocs(IndexType type, index_t entity_id, index_t entity_dim, index_t assoc_dim) const;
    void get_dim_map(IndexType type, index_t src_dim, index_t dst_dim, Node &map_node) const;
    void get_point_data(IndexType type, index_t point_id, Node &data) const;

    index_t get_length(index_t dim = -1) const;
    index_t get_embed_length(index_t entity_dim, index_t embed_dim) const;

    std::string to_json() const;

    void expand_assoc_capacity(IndexType type, index_t idx, index_t dim);
    inline index_t next_global_id(index_t dim) const
    {
        index_t tdim1 = topo_shape.dim + 1;
        return dim_geassocs_maps[dim].size() / tdim1;
    }
    inline index_t next_local_id(index_t dim) const
    {
        index_t tdim1 = topo_shape.dim + 1;
        return dim_leassocs_maps[dim].size() / tdim1;
    }

    // Compatibility methods
    TopologyMetadata(const conduit::Node &topology,
                     const conduit::Node &coordset,
                     size_t lowest_dim,
                     const std::vector<std::pair<size_t,size_t> > &desired_maps);

    int dimension() const { return topo_cascade.dim; }

    const conduit::Node &get_topology(size_t dim)
    {
        if(dim > 3) { CONDUIT_ERROR("Invalid dimension."); }
        return dim_topos[dim];
    }

    void get_topology(size_t dim, conduit::Node &dest)
    {
        if(dim > 3) { CONDUIT_ERROR("Invalid dimension."); }
        dest.reset();
        dest.set(dim_topos[dim]);
    }

    index_t get_topology_length(size_t dim) const
    {
        if(dim > 3) { CONDUIT_ERROR("Invalid dimension."); }
        return conduit::blueprint::mesh::utils::topology::length(dim_topos[dim]);
    }

    const std::vector<index_t> &
    get_global_association(index_t entity_id, index_t entity_dim, index_t assoc_dim) const
    {
        return get_entity_assocs(GLOBAL, entity_id, entity_dim, assoc_dim);
    }

    const std::vector<index_t> &
    get_local_association(index_t entity_id, index_t entity_dim, index_t assoc_dim) const
    {
        return get_entity_assocs(LOCAL, entity_id, entity_dim, assoc_dim);
    }

    const DataType &get_int_dtype() const { return int_dtype; }

    const DataType &get_float_dtype() const { return float_dtype; }

    const std::vector<index_t> &get_local_to_global_map(index_t dim) const
    {
        return dim_le2ge_maps[dim];
    }

    // Data.
    const conduit::Node *topo, *cset;
    const conduit::DataType int_dtype, float_dtype;
    const ShapeCascade topo_cascade;
    const ShapeType topo_shape;

    /*
      dim_geassocs_maps[dim][element * (topodims+1) + dim] -> associates vector
          [dim 0]
          [dim 1]--->elements
          [dim 2]      [ei=0, dim=0]
          [dim 3]      [ei=0, dim=1]
                       [ei=0, dim=2]--->associates {1,3,5}
                       [ei=0, dim=3]
                       [ei=1, dim=0]
                       [ei=1, dim=1]
                       ...
                       [ei=nelem-1, dim=3]

     */

    // per-dimension topology nodes (mapped onto 'cset' coordinate set)
    std::vector< conduit::Node > dim_topos;
    // per-dimension maps from global entity ids to per-dimension global associate ids
    std::vector< std::vector< std::vector<index_t> > > dim_geassocs_maps;
    // per-dimension maps from local entity ids to per-dimension local associate ids
    std::vector< std::vector< std::vector<index_t> > > dim_leassocs_maps;
    // per-dimension mapping from local entity ids to global entity ids (delegates)
    std::vector< std::vector<index_t> > dim_le2ge_maps;
};

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::reference --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
