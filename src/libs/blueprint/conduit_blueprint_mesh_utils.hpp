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
///
/// class: conduit::blueprint::mesh::utils::NDIndex
///
/// description:
///  General purpose index for strided structured meshes.
///
//-----------------------------------------------------------------------------
class CONDUIT_BLUEPRINT_API NDIndex
{
public:
    //-----------------------------------------------------------------------------
    //
    // -- conduit::blueprint::mesh::utils::NDIndex public members --
    //
    //-----------------------------------------------------------------------------

    //-----------------------------------------------------------------------------
    /// NDIndex Construction and Destruction
    //-----------------------------------------------------------------------------
    // Copy constructor.
    NDIndex(const NDIndex& idx);

    /// Primary index constructor.
    NDIndex(const Node* node);

    /// Primary index constructor.  The argument node should contain numeric
    /// children, all of length equal to the array's dimensionality:
    ///     - shape (required) specifies the shape of the array to index
    ///     - offset (optional) specifies where the data starts in the array
    ///     - stride (optional) specifies the extent of the array storing data
    ///
    /// The shape is required.  It is an error if shape is omitted.
    /// If offset is not specified, it defaults to 0 in each dimension.
    /// If stride is not specified, it defaults to
    /// \code
    /// stride[0] = 1
    /// stride[i] = stride[i-1] * (offset[i-1] + shape[i-1])
    /// \endcode
    /// Node that offset is specified in terms of logical index, not
    /// flatindex, and stride is specified in terms of flatindex.  Also note
    /// that the default stride holds an assumption that the data is laid
    /// out in C-style, with fastest-varying dimension left-most.  Users may
    /// specify a custom stride to index Fortran-style arrays, where the
    /// fastest-varying index is right-most.
    ///
    /// Here are a few examples:
    ///
    /// - A 6x4 array
    ///   \code
    ///   shape: [6, 4]
    ///   \endcode
    /// - A 6x4 array with two extra elements at the end of each row
    ///   \code
    ///   shape: [6, 4]
    ///   stride: [1, 8]
    ///   \endcode
    /// - A 6x4 array with two elements of padding on the low end of a
    ///   dimension and one element of padding on the high end
    ///   \code
    ///   shape: [6, 4]
    ///   offset: [2, 2]
    ///   stride: [1, 9]
    ///   \endcode
    /// - A 6x4x5 array with two elements of padding on the low end of each
    ///   dimension and one element of padding on the high end (adds third
    ///   dimension to previous)
    ///   \code
    ///   shape: [6, 4, 5]
    ///   offset: [2, 2, 2]
    ///   stride: [1, 9, 63]
    ///   \endcode
    /// - A Fortran 6x4x5 array with two elements of padding on the low
    ///   end of each dimension and one element of padding on the high end
    ///   (previous example changed to column-major)
    ///   \code
    ///   shape: [6, 4, 5]
    ///   offset: [2, 2, 2]
    ///   stride: [63, 7, 1]
    ///   \endcode
    NDIndex(const Node& node);

    /// Array constructor
    NDIndex(const index_t dim, const index_t* shape, const index_t* offset = NULL, const index_t* stride = NULL);

    /// Destructor
    ~NDIndex() { };

    /// Assignment operator.
    NDIndex& operator=(const NDIndex& itr);

    //-----------------------------------------------------------------------------
    /// Retrieve a flat-index: public interface.
    //-----------------------------------------------------------------------------
    template<typename T, typename... Ts>
    index_t     index(T idx, Ts... idxs) const;
    template<typename T>
    index_t     index(T idx) const;

    /// With default argument, returns the number of ranks or dimensions for
    /// this NDIndex.  With dim >= 0, returns the extent of this NDIndex
    /// for dimension dim.
    index_t     shape(index_t dim = -1) const;

    /// Returns the logical index in dimension dim where the data starts.
    index_t     offset(index_t dim) const;

    /// Returns the stride along dimension dim.
    index_t     stride(index_t dim) const;

    //-----------------------------------------------------------------------------
    /// Human readable info about this iterator
    //-----------------------------------------------------------------------------
    void        info(Node& res) const;

private:

    //-----------------------------------------------------------------------------
    //
    // -- conduit::blueprint::mesh::utils::NDIndex private members --
    //
    //-----------------------------------------------------------------------------

    /// Accessors for shape, offset, and stride
    index_t_accessor m_shape_acc;
    index_t_accessor m_offset_acc;
    index_t_accessor m_stride_acc;

    /// Dimension (length of shape, offset, and stride nodes)
    index_t m_dim;
};

template<typename T, typename... Ts>
index_t
NDIndex::index(T idx, Ts... idxs) const
{
    index_t depth = m_dim - sizeof...(idxs) - 1;
    index_t component = (offset(depth) + idx) * stride(depth);
    return component + index(idxs...);
}

template<typename T>
index_t
NDIndex::index(T idx) const
{
    index_t depth = m_dim - 1;
    index_t component = (offset(depth) + idx) * stride(depth);
    return component;
}

inline
index_t
NDIndex::shape(index_t dim) const
{
    int retval = 0;
    if (dim < 0)
    {
        retval = m_dim;
    }
    else
    {
        retval = m_shape_acc[dim];
    }
    return retval;
}

inline
index_t
NDIndex::offset(index_t dim) const
{
    if (m_offset_acc.number_of_elements() < 1)
    {
        return 0;
    }
    return m_offset_acc[dim];
}

inline
index_t
NDIndex::stride(index_t dim) const
{
    if (m_stride_acc.number_of_elements() < 1)
    {
        index_t acc = 1;
        for (int d = 0; d < dim && d < m_dim; ++d)
        {
            acc = acc * (m_shape_acc[d] + offset(d));
        }
        return acc;
    }
    return m_stride_acc[dim];
}



//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::O2MIndex --
//-----------------------------------------------------------------------------

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
