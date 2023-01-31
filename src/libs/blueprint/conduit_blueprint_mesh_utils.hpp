// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_util.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_UTILS_HPP
#define CONDUIT_BLUEPRINT_MESH_UTILS_HPP

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
#include "conduit_blueprint_mesh_utils_detail.hpp"
#include "conduit_blueprint_exports.h"

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

//-----------------------------------------------------------------------------
/// blueprint mesh utility constants
//-----------------------------------------------------------------------------

static const DataType DEFAULT_INT_DTYPE = DataType::int32(1);
static const DataType DEFAULT_UINT_DTYPE = DataType::uint32(1);
static const DataType DEFAULT_FLOAT_DTYPE = DataType::float32(1);
static const std::vector<DataType> DEFAULT_INT_DTYPES = {DEFAULT_INT_DTYPE, DEFAULT_UINT_DTYPE};
static const std::vector<DataType> DEFAULT_NUMBER_DTYPES = {DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE, DEFAULT_UINT_DTYPE};

static const std::vector<DataType> INT_DTYPES = {DataType::int32(1), DataType::int64(1)};
static const std::vector<DataType> FLOAT_DTYPES = {DataType::float32(1), DataType::float64(1)};

static const std::vector<std::string> ASSOCIATIONS = {"vertex", "element"};
static const std::vector<std::string> BOOLEANS = {"true", "false"};
static const std::vector<std::string> NESTSET_TYPES = {"parent", "child"};

static const std::vector<std::string> COORDINATE_AXES = {"x", "y", "z", "r", "z", "theta", "phi"};
static const std::vector<std::string> CARTESIAN_AXES = {"x", "y", "z"};
static const std::vector<std::string> CYLINDRICAL_AXES = {"r", "z"};
static const std::vector<std::string> SPHERICAL_AXES = {"r", "theta", "phi"};
static const std::vector<std::string> LOGICAL_AXES = {"i", "j", "k"};

static const std::vector<std::string> COORD_TYPES = {"uniform", "rectilinear", "explicit"};
static const std::vector<std::string> COORD_SYSTEMS = {"cartesian", "cylindrical", "spherical"};

static const std::vector<std::string> TOPO_TYPES = {"points", "uniform", "rectilinear", "structured", "unstructured"};

// Note: To add a new topo shape type, you must do the following:
//  1) Add an entry to TOPO_SHAPES, TOPO_SHAPE_IDS, TOPO_SHAPE_DIMS, TOPO_SHAPE_INDEX_COUNTS, 
//     TOPO_SHAPE_EMBED_TYPES, TOPO_SHAPE_EMBED_COUNTS, and TOPO_SHAPE_EMBEDDINGS. These arrays
//     are indexed by the same values, so be very careful to add elements in the same place
//     in each array.
//  2) If you are adding elements in the middle of these arrays, then make sure that for 
//     TOPO_SHAPE_EMBED_TYPES, you update the indices of any shapes that come after the ones
//     you are adding.
//  3) Head over to conduit_blueprint_mesh_utils_iterate_elements.hpp and find the enum class ShapeId.
//     Add an element for your shape there, and update the others if adding in the middle.

static const std::vector<std::string> TOPO_SHAPES = {"point", "line", "tri", "quad", 
    "tet", "hex", "wedge", "pyramid", "polygonal", "polyhedral", "mixed"};

// "p" is for point
// "l" is for line
// "f" is for face
// "c" is for cell
static const std::vector<std::string> TOPO_SHAPE_IDS = {/*point*/ "p", /*line*/ "l", /*tri*/ "f", /*quad*/ "f", 
    /*tet*/ "c", /*hex*/ "c", /*wedge*/ "c", /*pyramid*/ "c", /*polygonal*/ "f", /*polyhedral*/ "c"};

// The dimensions for each element in TOPO_SHAPES
static const std::vector<index_t> TOPO_SHAPE_DIMS = {/*point*/ 0, /*line*/ 1, /*tri*/ 2, /*quad*/ 2, 
    /*tet*/ 3, /*hex*/ 3, /*wedge*/ 3, /*pyramid*/ 3, /*polygonal*/ 2, /*polyhedral*/ 3, /*mixed*/ -1};

// How many points are in each element in TOPO_SHAPES
static const std::vector<index_t> TOPO_SHAPE_INDEX_COUNTS = {/*point*/ 1, /*line*/ 2, /*tri*/ 3, /*quad*/ 4, 
    /*tet*/ 4, /*hex*/ 8, /*wedge*/ 6, /*pyramid*/ 5, /*polygonal*/ -1, /*polyhedral*/ -1, /*mixed*/ -1};

// For each element in TOPO_SHAPES, the index into TOPO_SHAPES of the underlying shape. 
// Points have no underlying shape so they get -1.
// Lines have points under the hood so they get 0.
// Triangles are made of lines so they get 1.
// Hexahedrons are made of quads so they get 3.
static const std::vector<index_t> TOPO_SHAPE_EMBED_TYPES = {/*point*/ -1, /*line*/ 0, /*tri*/ 1, /*quad*/ 1, 
    /*tet*/ 2, /*hex*/ 3, /*wedge*/ 2, /*pyramid*/ 2, /*polygonal*/ 1, /*polyhedral*/ 8, /*mixed*/ -1};

// How many of those underlying shapes are there?
// Lines are made of two points so they get 2.
// Triangles are made of three lines so they get 3.
// Hexahedrons are made of 6 quads so they get 6.
// Wedges are made of two end caps (tris) plus three quad sides each split into two tris, so they get 8.
// Pyramids are made of four triangular sides plus a quad base split into two tris, so they get 6.
static const std::vector<index_t> TOPO_SHAPE_EMBED_COUNTS = {/*point*/ 0, /*line*/ 2, /*tri*/ 3, /*quad*/ 4, 
    /*tet*/ 4, /*hex*/ 6, /*wedge*/ 8, /*pyramid*/ 6, /*polygonal*/ -1, /*polyhedral*/ -1, /*mixed*/ -1};

// TODO(JRC): These orientations currently assume the default Conduit-Blueprit
// windings are used for the input geometry, which happens to be the case
// for all example geometry but cannot be assumed for all inputs. In order
// for these arrangements to be used generally, the winding feature needs to
// be implemented and used to perform index space transforms.
static const index_t TOPO_POINT_EMBEDDING[1][1] = {
    {0}};
static const index_t TOPO_LINE_EMBEDDING[2][1] = {
    {0}, {1}};
static const index_t TOPO_TRI_EMBEDDING[3][2] = {
    {0, 1}, {1, 2}, {2, 0}};
static const index_t TOPO_QUAD_EMBEDDING[4][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}};
static const index_t TOPO_TET_EMBEDDING[4][3] = {
    {0, 2, 1}, {0, 1, 3},
    {0, 3, 2}, {1, 2, 3}};
static const index_t TOPO_HEX_EMBEDDING[6][4] = {
    {0, 3, 2, 1}, {0, 1, 5, 4}, {1, 2, 6, 5},
    {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}};
static const index_t TOPO_WEDGE_EMBEDDING[8][3] = {
    {0, 1, 2}, {0, 1, 3}, {1, 3, 4}, {1, 2, 4},
    {2, 4, 5}, {2, 0, 5}, {0, 5, 3}, {3, 4, 5}};
static const index_t TOPO_PYRAMID_EMBEDDING[6][3] = {
    {0, 1, 2}, {3, 2, 0}, {0, 1, 4},
    {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
static const std::vector<const index_t*> TOPO_SHAPE_EMBEDDINGS = {
    &TOPO_POINT_EMBEDDING[0][0], &TOPO_LINE_EMBEDDING[0][0],
    &TOPO_TRI_EMBEDDING[0][0], &TOPO_QUAD_EMBEDDING[0][0],
    &TOPO_TET_EMBEDDING[0][0], &TOPO_HEX_EMBEDDING[0][0],
    &TOPO_WEDGE_EMBEDDING[0][0], &TOPO_PYRAMID_EMBEDDING[0][0],
    NULL, NULL};

//-----------------------------------------------------------------------------
/// blueprint mesh utility structures
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
struct CONDUIT_BLUEPRINT_API ShapeType
{
public:
    ShapeType();
    ShapeType(const index_t type_id);
    ShapeType(const std::string &type_name);
    ShapeType(const conduit::Node &topology);

    bool is_poly() const;
    bool is_polygonal() const;
    bool is_polyhedral() const;
    bool is_valid() const;

    std::string type;
    index_t id, dim, indices;
    index_t embed_id, embed_count, *embedding;

private:
    void init(const index_t type_id);
    void init(const std::string &type_name);
};

//---------------------------------------------------------------------------//
struct CONDUIT_BLUEPRINT_API ShapeCascade
{
public:
    ShapeCascade(const conduit::Node &topology);
    ShapeCascade(const ShapeType &shape_type);

    index_t get_num_embedded(const index_t level) const;
    const ShapeType& get_shape(const index_t level = -1) const;

    ShapeType dim_types[4];
    index_t dim;
private:
    void init(const ShapeType &shape_type);
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

 An example is helpful to understand the concept of the maps.

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

 The entity maps contain the information of how these elements,faces,edges,points
 are connected.

 The global entity map for dimension 3 to dimension 2, which we'll write as G(3,2),
 indicates the relationship between entities of dimension 3 to the entities of
 dimension 2. Since the dimension 3 elements are made of the dimension 2 elements
 (much as PH cells are expressed in Blueprint), the G(3,2) map will consist
 of the indices of the faces that make up element 0, followed by the indices of the
 faces that make up element 1, and so on.

           element 0 faces   element 1 faces ...
 G(3,2) = {0, 1, 2, 3, 4, 5}{6, 7, 8, 9, 2, 10}...
                    |
                    |
                    face 3 in unique face list (topo[2] connectivity)

 There are other global entity maps G(e,a) too. G(3,3) is the relationship of the elements
 to elements, which is self so any input to the map is also the output. The G(3,1) map
 represents the indices of the 1D edges that make up the 3D element. The G(3,0) map
 represents all of the points contained by the 3D element, values which for most
 cases are contained in the 3D connectivity.

 G(e,a) is shorthand for "Global(entity_dim, association_dim)".

 When e==a, the relationship is self.
 When e>a, the relationship is that entity e contains/references entity a.
 When a>e, the relationship is entity a is referenced by parents of dimension e.

 In the above example for G(3,2), we can see that face 2 is shared by elements 0
 and 1. If we want to know which edges make up face 2, we can consult G(2,1).

           face 0      face 1      face 2      face 3
 G(2,1) = {0, 1, 2, 3}{3, 4, 5, 6}{2, 7, 8, 4}{1, 9, 10, 7}...
                                   |
                                   |
                                   face 2 in unique edge list (topo[1] connectivity)

 If we want to know which points make up the edges for face 2, consult G(1,0). This is
 the map of lines to points (also top_dims[1] connectivity).

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

 */
class CONDUIT_BLUEPRINT_API TopologyMetadata : public detail::NewTopologyMetadata
{
public:
    using ParentClass = detail::NewTopologyMetadata;
    using IndexType = detail::NewTopologyMetadata::IndexType;
//    enum IndexType { GLOBAL = 0, LOCAL = 1 };

    //-----------------------------------------------------------------------
    /**
     @brief Legacy constructor, which builds all of the topology levels in the shape
            cascade as well as all associations.

     @param topology     The input topology node.
     @param coordset     The input coordset associated with the topology.
     */
    TopologyMetadata(const conduit::Node &topology, const conduit::Node &coordset) :
        ParentClass(topology, coordset)
    {
    }

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
                        const std::vector<std::pair<size_t,size_t> > &desired_maps) :
        ParentClass(topology, coordset, lowest_dim, desired_maps)
    {
    }

    //-----------------------------------------------------------------------
    /**
     @brief Get the highest shape dimension.
     @return The highest shape dimension.
     */
    int dimension() const { return ParentClass::dimension(); }

    //-----------------------------------------------------------------------
    /**
     @brief Get the topologies array.
     @return The topologies array.
     */
    const conduit::Node *get_topologies() const
    {
        return ParentClass::get_topologies();
    }

    //-----------------------------------------------------------------------
    /**
     @brief Get the legnths arrays for the topologies. Any topologies that
            were not produced will have length 0.
     @return The topology lengths array.
     */
    const index_t *get_topology_lengths() const
    {
        return ParentClass::get_topology_lengths();
    }

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
           to for a given return are in the associations.
     */
    conduit::vector_view<index_t>
    get_global_association(index_t entity_id, index_t entity_dim, index_t assoc_dim) const
    {
        return ParentClass::get_global_association(entity_id, entity_dim, assoc_dim);
    }

    conduit::range_vector<index_t>
    get_local_association(index_t entity_id, index_t entity_dim, index_t assoc_dim) const
    {
        return ParentClass::get_local_association(entity_id, entity_dim, assoc_dim);
    }

    //-----------------------------------------------------------------------
    /**
     @brief Get whether the association e,a was requested when the object
            was initialized. If so, the association will exist. Otherwise,
            the association does not exist.

     @param entity_dim The starting dimension.
     @param assoc_dim  The destination dimension.

     @return True if the association exists; False otherwise.
     */
    bool association_requested(index_t entity_dim, index_t assoc_dim) const
    {
        return ParentClass::association_requested(entity_dim, assoc_dim);
    }

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
    bool get_dim_map(IndexType type, index_t src_dim, index_t dst_dim, Node &map_node) const
    {
        return ParentClass::get_dim_map(type, src_sim, dst_dim, map_node);
    }
};


//-----------------------------------------------------------------------------
/// blueprint mesh utility functions
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Node CONDUIT_BLUEPRINT_API link_nodes(const Node &lhs, const Node &rhs);

//-----------------------------------------------------------------------------
DataType CONDUIT_BLUEPRINT_API find_widest_dtype(const Node &node, const DataType &default_dtype);
//-----------------------------------------------------------------------------
DataType CONDUIT_BLUEPRINT_API find_widest_dtype(const Node &node, const std::vector<DataType> &default_dtypes);

//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API const Node * find_reference_node(const Node &node, const std::string &ref_key);
//-----------------------------------------------------------------------------
index_t CONDUIT_BLUEPRINT_API find_domain_id(const Node &node);

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::connectivity --
//-----------------------------------------------------------------------------
namespace connectivity
{
    //-------------------------------------------------------------------------
    typedef std::vector<index_t> ElemType;
    typedef std::map< index_t, std::vector<index_t> > SubelemMap;

    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API make_element_2d(std::vector<index_t>& elem,
                                               index_t element,
                                               index_t iwidth);
    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API make_element_3d(ElemType& connect,
                                               index_t element,
                                               index_t iwidth,
                                               index_t jwidth,
                                               index_t kwidth,
                                               SubelemMap& faces);

    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API create_elements_2d(const Node& ref_win,
                                                  index_t i_lo,
                                                  index_t j_lo,
                                                  index_t iwidth,
                                                  std::map<index_t, std::vector<index_t> >& elems);
    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API create_elements_3d(const Node& ref_win,
                                                  index_t i_lo,
                                                  index_t j_lo,
                                                  index_t k_lo,
                                                  index_t iwidth,
                                                  index_t jwidth,
                                                  index_t kwidth,
                                                  std::map<index_t, ElemType>& elems,
                                                  SubelemMap& faces);

    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API connect_elements_2d(const Node& ref_win,
                                                   index_t i_lo,
                                                   index_t j_lo,
                                                   index_t iwidth,
                                                   index_t ratio,
                                                   index_t& new_vertex,
                                                   std::map<index_t, std::vector<index_t> >& elems);
    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API connect_elements_3d(const Node& ref_win,
                                                   index_t i_lo,
                                                   index_t j_lo,
                                                   index_t k_lo,
                                                   index_t iwidth,
                                                   index_t jwidth,
                                                   index_t& new_vertex,
                                                   std::map<index_t, ElemType>& elems);
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::connectivity --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::coordset --
//-----------------------------------------------------------------------------
namespace coordset
{
    //-------------------------------------------------------------------------
    index_t CONDUIT_BLUEPRINT_API dims(const conduit::Node &coordset);

    //-------------------------------------------------------------------------
    index_t CONDUIT_BLUEPRINT_API length(const conduit::Node &coordset);

    //-----------------------------------------------------------------------------
    std::vector<std::string> CONDUIT_BLUEPRINT_API axes(const Node &coordset);

    //-----------------------------------------------------------------------------
    std::string CONDUIT_BLUEPRINT_API coordsys(const Node &n);

    //-----------------------------------------------------------------------------
    /**
    @brief Updates array d to hold the number of verticies in each dimension
        for the given coordset. Explicit coordsets will just report their
        number_of_elements() in d[0].
    */
    void logical_dims(const conduit::Node &n, index_t *d, index_t maxdims);

    //-----------------------------------------------------------------------------
    /**
    @brief Reads the coordset's data and determines min/max for each axis.
    NOTE: This simply takes the min/max of each data array for recilinear/explicit,
    are there any special considerations for cylindrical and spherical coordinates?
    For uniform it calculates min/max based off of origin/spacing/dims.
    @return A vector of float64 in the format {d0min, d0max, ... , dNmin, dNmax}
    */
    std::vector<float64> CONDUIT_BLUEPRINT_API extents(const Node &n);

    namespace uniform
    {
        /**
        @brief Reads the given uniform coordset and extracts to spacing
               to an index_t vector
        */
        std::vector<double> CONDUIT_BLUEPRINT_API spacing(const Node &n);

        std::vector<index_t> CONDUIT_BLUEPRINT_API origin(const Node &n);
    }

    std::string CONDUIT_BLUEPRINT_API coordsys(const Node &coordset);

    //-------------------------------------------------------------------------
    // -- begin conduit::blueprint::mesh::utils::coordset::_explicit --
    //-------------------------------------------------------------------------
    namespace _explicit
    {
        //-------------------------------------------------------------------------
        std::vector<float64> CONDUIT_BLUEPRINT_API coords(const Node &coordset, const index_t i);
    }
    //-------------------------------------------------------------------------
    // -- end conduit::blueprint::mesh::utils::coordset::_explicit --
    //-------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::coordset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::topology --
//-----------------------------------------------------------------------------
namespace topology
{
    //-------------------------------------------------------------------------
    index_t CONDUIT_BLUEPRINT_API dims(const conduit::Node &topo);

    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API logical_dims(const Node &n, index_t *d, index_t maxdims);

    //-------------------------------------------------------------------------
    index_t CONDUIT_BLUEPRINT_API length(const conduit::Node &topo);

    // return the coordset for this topology
    const Node & coordset(const conduit::Node &topo);

    //-------------------------------------------------------------------------
    /**
     * @brief Reindexes the vertices in a topology to be associated with a new
     * coordset, based on a global vertex ID numbering.
     * The old coordset must be a subset of the new coordset.
     */
    void CONDUIT_BLUEPRINT_API reindex_coords(const conduit::Node& topo,
                                              const conduit::Node& new_coordset,
                                              const conduit::Node& old_gvids,
                                              const conduit::Node& new_gvids,
                                              conduit::Node& out_topo);

    //-------------------------------------------------------------------------
    // -- begin conduit::blueprint::mesh::utils::topology::unstructured --
    //-------------------------------------------------------------------------
    namespace unstructured
    {
        //-------------------------------------------------------------------------
        // Generates element offsets for given topo
        void CONDUIT_BLUEPRINT_API generate_offsets(const Node &topo,
                                                    Node &dest);

        //-------------------------------------------------------------------------
        // Generates element and subelement offsets for given topo
        void CONDUIT_BLUEPRINT_API generate_offsets(const Node &topo,
                                                    Node &dest_ele_offsets,
                                                    Node &dest_subele_offsets);

        //-------------------------------------------------------------------------
        // Adds offsets to given topo
        void CONDUIT_BLUEPRINT_API generate_offsets_inline(Node &topo);

        //-------------------------------------------------------------------------
        std::vector<index_t> CONDUIT_BLUEPRINT_API points(const Node &topo,
                                                          const index_t i);
    }
    //-------------------------------------------------------------------------
    // -- end conduit::blueprint::mesh::utils::topology::unstructured --
    //-------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::topology --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::adjset --
//-----------------------------------------------------------------------------
namespace adjset
{
    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API canonicalize(Node &adjset);
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::adjset --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils --
//-----------------------------------------------------------------------------

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
