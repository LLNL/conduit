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
static const std::vector<std::string> TOPO_SHAPES = {"point", "line", "tri", "quad", "tet", "hex", "polygonal", "polyhedral"};
static const std::vector<std::string> TOPO_SHAPE_IDS = {"p", "l", "f", "f", "c", "c", "f", "c"};
static const std::vector<index_t> TOPO_SHAPE_DIMS = {0, 1, 2, 2, 3, 3, 2, 3};
static const std::vector<index_t> TOPO_SHAPE_INDEX_COUNTS = {1, 2, 3, 4, 4, 8, -1, -1};
static const std::vector<index_t> TOPO_SHAPE_EMBED_TYPES = {-1, 0, 1, 1, 2, 3, 1, 6};
static const std::vector<index_t> TOPO_SHAPE_EMBED_COUNTS = {0, 2, 3, 4, 4, 6, -1, -1};

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
static const std::vector<const index_t*> TOPO_SHAPE_EMBEDDINGS = {
    &TOPO_POINT_EMBEDDING[0][0], &TOPO_LINE_EMBEDDING[0][0],
    &TOPO_TRI_EMBEDDING[0][0], &TOPO_QUAD_EMBEDDING[0][0],
    &TOPO_TET_EMBEDDING[0][0], &TOPO_HEX_EMBEDDING[0][0],
    NULL, NULL};

//-----------------------------------------------------------------------------
/// blueprint mesh utility structures
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
struct ShapeType
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
struct ShapeCascade
{
    ShapeCascade(const conduit::Node &topology);

    index_t get_num_embedded(const index_t level) const;
    const ShapeType& get_shape(const index_t level = -1) const;

    ShapeType dim_types[4];
    index_t dim;
};

//---------------------------------------------------------------------------//
struct TopologyMetadata
{
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
    enum IndexType { GLOBAL = 0, LOCAL = 1 };

    TopologyMetadata(const conduit::Node &topology, const conduit::Node &coordset);

    void add_entity_assoc(IndexType type, index_t e0_id, index_t e0_dim, index_t e1_id, index_t e1_dim);

    const std::vector<index_t> &get_entity_assocs(IndexType type, index_t entity_id, index_t entity_dim, index_t assoc_dim) const;
    void get_dim_map(IndexType type, index_t src_dim, index_t dst_dim, Node &map_node) const;
    void get_entity_data(IndexType type, index_t entity_id, index_t entity_dim, Node &data) const;
    void get_point_data(IndexType type, index_t point_id, Node &data) const;

    index_t get_length(index_t dim = -1) const;
    index_t get_embed_length(index_t entity_dim, index_t embed_dim) const;

    std::string to_json() const;

    const conduit::Node *topo, *cset;
    const conduit::DataType int_dtype, float_dtype;
    const ShapeCascade topo_cascade;
    const ShapeType topo_shape;

    // per-dimension topology nodes (mapped onto 'cset' coordinate set)
    std::vector< conduit::Node > dim_topos;
    // per-dimension maps from an entity's point id set to its global entity id
    std::vector< std::map< std::set<index_t>, index_t > > dim_geid_maps;
    // per-dimension maps from global entity ids to per-dimension global associate ids
    std::vector< std::vector< std::vector< std::pair< std::vector<index_t>, std::set<index_t> > > > > dim_geassocs_maps;
    // per-dimension maps from local entity ids to per-dimension local associate ids
    std::vector< std::vector< std::vector< std::pair< std::vector<index_t>, std::set<index_t> > > > > dim_leassocs_maps;
    // per-dimension mapping from local entity ids to global entity ids (delegates)
    std::vector< std::vector<index_t> > dim_le2ge_maps;
};

//-----------------------------------------------------------------------------
/// blueprint mesh utility functions
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Node link_nodes(const Node &lhs, const Node &rhs);

//-----------------------------------------------------------------------------
DataType find_widest_dtype(const Node &node, const DataType &default_dtype);
//-----------------------------------------------------------------------------
DataType find_widest_dtype(const Node &node, const std::vector<DataType> &default_dtypes);

//-----------------------------------------------------------------------------
bool find_reference_node(const Node &node, const std::string &ref_key, Node &ref);

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::coordset --
//-----------------------------------------------------------------------------
namespace coordset
{
    //-------------------------------------------------------------------------
    index_t dims(const conduit::Node &n);

    //-------------------------------------------------------------------------
    index_t length(const conduit::Node &n);

    //-----------------------------------------------------------------------------
    std::vector<std::string> axes(const Node &n);

    //-----------------------------------------------------------------------------
    std::string coordsys(const Node &n);
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::coorset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::topology --
//-----------------------------------------------------------------------------
namespace topology
{
    //-------------------------------------------------------------------------
    index_t dims(const conduit::Node &n);

    //-------------------------------------------------------------------------
    index_t length(const conduit::Node &n);

    //-------------------------------------------------------------------------
    // -- begin conduit::blueprint::mesh::utils::topology::unstructured --
    //-------------------------------------------------------------------------
    namespace unstructured
    {
        // TODO(JRC): Expose this 'cache' version of the function publicly?
        //-------------------------------------------------------------------------
        void generate_offsets(Node &n,
                              Node &dest);

        //-------------------------------------------------------------------------
        void generate_offsets(const Node &n,
                              Node &dest);
    }
    //-------------------------------------------------------------------------
    // -- end conduit::blueprint::mesh::utils::topology::unstructured --
    //-------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::topology --
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
