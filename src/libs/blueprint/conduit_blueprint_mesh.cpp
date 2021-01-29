// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh.cpp
///
//-----------------------------------------------------------------------------

#if defined(CONDUIT_PLATFORM_WINDOWS)
#define NOMINMAX
#undef min
#undef max
#include "windows.h"
#endif

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <set>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mcarray.hpp"
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_mesh.hpp"
#include "conduit_log.hpp"

using namespace conduit;
// Easier access to the Conduit logging functions
using namespace conduit::utils;
// access conduit path helper
using ::conduit::utils::join_path;

//-----------------------------------------------------------------------------
namespace conduit { namespace blueprint { namespace mesh {
//-----------------------------------------------------------------------------

    bool verify_single_domain(const conduit::Node &n, conduit::Node &info);
    bool verify_multi_domain(const conduit::Node &n, conduit::Node &info);

    static const std::string association_list[2] = {"vertex", "element"};
    static const std::vector<std::string> associations(association_list,
        association_list + sizeof(association_list) / sizeof(association_list[0]));

    static const std::string boolean_list[2] = {"true", "false"};
    static const std::vector<std::string> booleans(boolean_list,
        boolean_list + sizeof(boolean_list) / sizeof(boolean_list[0]));

    static const std::string coord_type_list[3] = {"uniform", "rectilinear", "explicit"};
    static const std::vector<std::string> coord_types(coord_type_list,
        coord_type_list + sizeof(coord_type_list) / sizeof(coord_type_list[0]));

    static const std::string coord_system_list[3] = {"cartesian", "cylindrical", "spherical"};
    static const std::vector<std::string> coord_systems(coord_system_list,
        coord_system_list + sizeof(coord_system_list) / sizeof(coord_system_list[0]));

    static const std::string topo_type_list[5] = {"points", "uniform",
        "rectilinear", "structured", "unstructured"};
    static const std::vector<std::string> topo_types(topo_type_list,
        topo_type_list + sizeof(topo_type_list) / sizeof(topo_type_list[0]));

    static const std::string topo_shape_list[8] = {"point", "line",
        "tri", "quad", "tet", "hex", "polygonal", "polyhedral"};
    static const std::vector<std::string> topo_shapes(topo_shape_list,
        topo_shape_list + sizeof(topo_shape_list) / sizeof(topo_shape_list[0]));

    static const std::string topo_shape_id_list[8] = {"p", "l",
        "f", "f", "c", "c", "f", "c"};
    static const std::vector<std::string> topo_shape_ids(topo_shape_id_list,
        topo_shape_id_list + sizeof(topo_shape_id_list) / sizeof(topo_shape_id_list[0]));

    static const index_t topo_shape_dim_list[8] = {0, 1,
        2, 2, 3, 3, 2, 3};
    static const std::vector<index_t> topo_shape_dims(
        topo_shape_dim_list, topo_shape_dim_list +
        sizeof(topo_shape_dim_list) / sizeof(topo_shape_dim_list[0]));

    static const index_t topo_shape_index_count_list[8] = {1, 2,
        3, 4, 4, 8, -1, -1};
    static const std::vector<index_t> topo_shape_index_counts(
        topo_shape_index_count_list, topo_shape_index_count_list +
        sizeof(topo_shape_index_count_list) / sizeof(topo_shape_index_count_list[0]));

    static const index_t topo_shape_embed_type_list[8] = {-1, 0,
        1, 1, 2, 3, 1, 6};
    static const std::vector<index_t> topo_shape_embed_types(
        topo_shape_embed_type_list, topo_shape_embed_type_list +
        sizeof(topo_shape_embed_type_list) / sizeof(topo_shape_embed_type_list[0]));

    static const index_t topo_shape_embed_count_list[8] = {0, 2,
        3, 4, 4, 6, -1, -1};
    static const std::vector<index_t> topo_shape_embed_counts(
        topo_shape_embed_count_list, topo_shape_embed_count_list +
        sizeof(topo_shape_embed_count_list) / sizeof(topo_shape_embed_count_list[0]));

    // TODO(JRC): These orientations currently assume the default Conduit-Blueprit
    // windings are used for the input geometry, which happens to be the case
    // for all example geometry but cannot be assumed for all inputs. In order
    // for these arrangements to be used generally, the winding feature needs to
    // be implemented and used to perform index space transforms.
    static const index_t topo_point_embedding[1][1] = {
        {0}};
    static const index_t topo_line_embedding[2][1] = {
        {0}, {1}};
    static const index_t topo_tri_embedding[3][2] = {
        {0, 1}, {1, 2}, {2, 0}};
    static const index_t topo_quad_embedding[4][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}};
    static const index_t topo_tet_embedding[4][3] = {
        {0, 2, 1}, {0, 1, 3},
        {0, 3, 2}, {1, 2, 3}};
    static const index_t topo_hex_embedding[6][4] = {
        {0, 3, 2, 1}, {0, 1, 5, 4}, {1, 2, 6, 5},
        {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}};

    static const index_t* topo_shape_embedding_list[8] = {
        &topo_point_embedding[0][0], &topo_line_embedding[0][0],
        &topo_tri_embedding[0][0], &topo_quad_embedding[0][0],
        &topo_tet_embedding[0][0], &topo_hex_embedding[0][0],
        NULL, NULL};
    static const std::vector<const index_t*> topo_shape_embeddings(
        topo_shape_embedding_list, topo_shape_embedding_list +
        sizeof(topo_shape_embedding_list) / sizeof(topo_shape_embedding_list[0]));

    static const std::string coordinate_axis_list[7] = {"x", "y", "z", "r", "z", "theta", "phi"};
    static const std::vector<std::string> coordinate_axes(coordinate_axis_list,
        coordinate_axis_list + sizeof(coordinate_axis_list) / sizeof(coordinate_axis_list[0]));

    static const std::string cartesian_axis_list[3] = {"x", "y", "z"};
    static const std::vector<std::string> cartesian_axes(cartesian_axis_list,
        cartesian_axis_list + sizeof(cartesian_axis_list) / sizeof(cartesian_axis_list[0]));

    static const std::string cylindrical_axis_list[2] = {"r", "z"};
    static const std::vector<std::string> cylindrical_axes(cylindrical_axis_list,
        cylindrical_axis_list + sizeof(cylindrical_axis_list) / sizeof(cylindrical_axis_list[0]));

    static const std::string spherical_axis_list[7] = {"r", "theta", "phi"};
    static const std::vector<std::string> spherical_axes(spherical_axis_list,
        spherical_axis_list + sizeof(spherical_axis_list) / sizeof(spherical_axis_list[0]));

    static const std::string logical_axis_list[3] = {"i", "j", "k"};
    static const std::vector<std::string> logical_axes(logical_axis_list,
        logical_axis_list + sizeof(logical_axis_list) / sizeof(logical_axis_list[0]));

    static const std::string nestset_type_list[4] = {"parent", "child"};
    static const std::vector<std::string> nestset_types(nestset_type_list,
        nestset_type_list + sizeof(nestset_type_list) / sizeof(nestset_type_list[0]));

    static const DataType default_int_dtype(DataType::INT32_ID, 1);
    static const DataType default_uint_dtype(DataType::UINT32_ID, 1);
    static const DataType default_float_dtype(DataType::FLOAT64_ID, 1);

    static const DataType default_int_dtype_list[2] = {default_int_dtype, default_uint_dtype};
    static const std::vector<DataType> default_int_dtypes(default_int_dtype_list,
        default_int_dtype_list + sizeof(default_int_dtype_list) / sizeof(default_int_dtype_list[0]));

    static const DataType default_number_dtype_list[3] = {default_float_dtype,
        default_int_dtype, default_uint_dtype};
    static const std::vector<DataType> default_number_dtypes(default_number_dtype_list,
        default_number_dtype_list + sizeof(default_number_dtype_list) /
        sizeof(default_number_dtype_list[0]));
} } }

//-----------------------------------------------------------------------------
// -- begin internal helpers --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// - begin internal helper types -
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
struct ShapeType
{
    ShapeType()
    {
        init(-1);
    }

    ShapeType(const index_t type_id)
    {
        init(type_id);
    }

    ShapeType(const std::string &type_name)
    {
        init(type_name);
    }

    ShapeType(const conduit::Node &topology)
    {
        init(-1);

        if(topology["type"].as_string() == "unstructured" &&
            topology["elements"].has_child("shape"))
        {
            init(topology["elements/shape"].as_string());
        }
    };

    void init(const std::string &type_name)
    {
        init(-1);

        for(index_t i = 0; i < (index_t)blueprint::mesh::topo_shapes.size(); i++)
        {
            if(type_name == blueprint::mesh::topo_shapes[i])
            {
                init(i);
            }
        }
    }

    void init(index_t type_id)
    {
        if(type_id < 0 || type_id >= (index_t)blueprint::mesh::topo_shapes.size())
        {
            type = "";
            id = dim = indices = embed_id = embed_count = -1;
            embedding = NULL;
        }
        else
        {
            type = blueprint::mesh::topo_shapes[type_id];
            id = type_id;
            dim = blueprint::mesh::topo_shape_dims[type_id];
            indices = blueprint::mesh::topo_shape_index_counts[type_id];

            embed_id = blueprint::mesh::topo_shape_embed_types[type_id];
            embed_count = blueprint::mesh::topo_shape_embed_counts[type_id];
            embedding = const_cast<index_t*>(blueprint::mesh::topo_shape_embeddings[type_id]);
        }
    }

    bool is_poly() const { return embedding == NULL; }
    bool is_polygonal() const { return embedding == NULL && dim == 2; }
    bool is_polyhedral() const { return embedding == NULL && dim == 3; }
    bool is_valid() const { return id >= 0; }

    std::string type;
    index_t id, dim, indices;
    index_t embed_id, embed_count, *embedding;
};

//---------------------------------------------------------------------------//
struct ShapeCascade
{
    ShapeCascade(const conduit::Node &topology)
    {
        ShapeType base_type(topology);
        dim = base_type.dim;

        dim_types[base_type.dim] = base_type;
        for(index_t di = base_type.dim - 1; di >= 0; di--)
        {
            dim_types[di] = ShapeType(dim_types[di + 1].embed_id);
        }
    }

    index_t get_num_embedded(index_t level) const
    {
        index_t num_embedded = -1;

        if(!get_shape().is_poly())
        {
            num_embedded = 1;
            for(index_t di = level + 1; di <= dim; di++)
            {
                num_embedded *= dim_types[di].embed_count;
            }
        }

        return num_embedded;
    }

    const ShapeType& get_shape(index_t level = -1) const
    {
        return dim_types[level < 0 ? dim : level];
    }

    ShapeType dim_types[4];
    index_t dim;
};

//-----------------------------------------------------------------------------
// - end internal helper types -
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// - begin internal potpourri functions -
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void grid_ijk_to_id(const index_t *ijk,
                    const index_t *dims,
                    index_t &grid_id)
{
    grid_id = 0;
    for(index_t d = 0; d < 3; d++)
    {
        index_t doffset = ijk[d];
        for(index_t dd = 0; dd < d; dd++)
        {
            doffset *= dims[dd];
        }

        grid_id += doffset;
    }
}

//-----------------------------------------------------------------------------
void grid_id_to_ijk(const index_t id,
                    const index_t *dims,
                    index_t *grid_ijk)
{
    index_t dremain = id;
    for(index_t d = 3; d-- > 0;)
    {
        index_t dstride = 1;
        for(index_t dd = 0; dd < d; dd++)
        {
            dstride *= dims[dd];
        }

        grid_ijk[d] = dremain / dstride;
        dremain = dremain % dstride;
    }
}

//-----------------------------------------------------------------------------
std::vector<index_t> intersect_sets(const std::set<index_t> &s1,
                                    const std::set<index_t> &s2)
{
    std::vector<index_t> si(std::max(s1.size(), s2.size()));
    std::vector<index_t>::iterator si_end = std::set_intersection(
        s1.begin(), s1.end(), s2.begin(), s2.end(), si.begin());
    return std::vector<index_t>(si.begin(), si_end);
}

//-----------------------------------------------------------------------------
std::vector<index_t> intersect_sets(const std::vector<index_t> &v1,
                                    const std::vector<index_t> &v2)
{
    std::vector<index_t> res;
    for(index_t i1 = 0; i1 < (index_t)v1.size(); i1++)
    {
        for(index_t i2 = 0; i2 < (index_t)v2.size(); i2++)
        {
            if(v1[i1] == v2[i2])
            {
                res.push_back(v1[i1]);
            }
        }
    }
    return std::vector<index_t>(std::move(res));
}

//-----------------------------------------------------------------------------
std::vector<index_t> subtract_sets(const std::vector<index_t> &v1,
                                   const std::vector<index_t> &v2)
{
    std::vector<index_t> res;
    for(index_t i1 = 0; i1 < (index_t)v1.size(); i1++)
    {
        bool vi1_found = false;
        for(index_t i2 = 0; i2 < (index_t)v2.size() && !vi1_found; i2++)
        {
            vi1_found |= v1[i1] == v2[i2];
        }

        if(!vi1_found)
        {
            res.push_back(v1[i1]);
        }
    }
    return std::vector<index_t>(std::move(res));
}

//-----------------------------------------------------------------------------
// - end internal potpourri functions -
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// - begin internal helper functions -
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool verify_field_exists(const std::string &protocol,
                         const conduit::Node &node,
                         conduit::Node &info,
                         const std::string &field_name = "")
{
    bool res = true;

    if(field_name != "")
    {
        if(!node.has_child(field_name))
        {
            log::error(info, protocol, "missing child" + log::quote(field_name, 1));
            res = false;
        }

        log::validation(info[field_name], res);
    }

    return res;
}

//-----------------------------------------------------------------------------
bool verify_integer_field(const std::string &protocol,
                          const conduit::Node &node,
                          conduit::Node &info,
                          const std::string &field_name = "")
{
    Node &field_info = (field_name != "") ? info[field_name] : info;

    bool res = verify_field_exists(protocol, node, info, field_name);
    if(res)
    {
        const Node &field_node = (field_name != "") ? node[field_name] : node;

        if(!field_node.dtype().is_integer())
        {
            log::error(info, protocol, log::quote(field_name) + "is not an integer (array)");
            res = false;
        }
    }

    log::validation(field_info, res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_number_field(const std::string &protocol,
                         const conduit::Node &node,
                         conduit::Node &info,
                         const std::string &field_name = "")
{
    Node &field_info = (field_name != "") ? info[field_name] : info;

    bool res = verify_field_exists(protocol, node, info, field_name);
    if(res)
    {
        const Node &field_node = (field_name != "") ? node[field_name] : node;

        if(!field_node.dtype().is_number())
        {
            log::error(info, protocol, log::quote(field_name) + "is not a number");
            res = false;
        }
    }

    log::validation(field_info, res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_string_field(const std::string &protocol,
                         const conduit::Node &node,
                         conduit::Node &info,
                         const std::string &field_name = "")
{
    Node &field_info = (field_name != "") ? info[field_name] : info;

    bool res = verify_field_exists(protocol, node, info, field_name);
    if(res)
    {
        const Node &field_node = (field_name != "") ? node[field_name] : node;

        if(!field_node.dtype().is_string())
        {
            log::error(info, protocol, log::quote(field_name) + "is not a string");
            res = false;
        }
    }

    log::validation(field_info, res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_object_field(const std::string &protocol,
                         const conduit::Node &node,
                         conduit::Node &info,
                         const std::string &field_name = "",
                         const bool allow_list = false,
                         const index_t num_children = 0)
{
    Node &field_info = (field_name != "") ? info[field_name] : info;

    bool res = verify_field_exists(protocol, node, info, field_name);
    if(res)
    {
        const Node &field_node = (field_name != "") ? node[field_name] : node;

        if(!(field_node.dtype().is_object() ||
            (allow_list && field_node.dtype().is_list())))
        {
            log::error(info, protocol, log::quote(field_name) + "is not an object" +
                                       (allow_list ? " or a list" : ""));
            res = false;
        }
        else if(field_node.number_of_children() == 0)
        {
            log::error(info,protocol, "has no children");
            res = false;
        }
        else if(num_children && field_node.number_of_children() != num_children)
        {
            std::ostringstream oss;
            oss << "has incorrect number of children ("
                << field_node.number_of_children()
                << " vs "
                << num_children
                << ")";
            log::error(info,protocol, oss.str());
            res = false;
        }
    }

    log::validation(field_info, res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_mcarray_field(const std::string &protocol,
                          const conduit::Node &node,
                          conduit::Node &info,
                          const std::string &field_name)
{
    Node &field_info = info[field_name];

    bool res = verify_field_exists(protocol, node, info, field_name);
    if(res)
    {
        const Node &field_node = node[field_name];
        res = blueprint::mcarray::verify(field_node,field_info);
        if(res)
        {
            log::info(info, protocol, log::quote(field_name) + "is an mcarray");
        }
        else
        {
            log::error(info, protocol, log::quote(field_name) + "is not an mcarray");
        }
    }

    log::validation(field_info, res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_mlarray_field(const std::string &protocol,
                          const conduit::Node &node,
                          conduit::Node &info,
                          const std::string &field_name,
                          const index_t min_depth,
                          const index_t max_depth,
                          const bool leaf_uniformity)
{
    Node &field_info = info[field_name];

    bool res = verify_field_exists(protocol, node, info, field_name);
    if(res)
    {
        const Node &field_node = node[field_name];
        res = blueprint::mlarray::verify(field_node,field_info,min_depth,max_depth,leaf_uniformity);
        if(res)
        {
            log::info(info, protocol, log::quote(field_name) + "is an mlarray");
        }
        else
        {
            log::error(info, protocol, log::quote(field_name) + "is not an mlarray");
        }
    }

    log::validation(field_info, res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_o2mrelation_field(const std::string &protocol,
                              const conduit::Node &node,
                              conduit::Node &info,
                              const std::string &field_name)
{
    Node &field_info = info[field_name];

    bool res = verify_field_exists(protocol, node, info, field_name);
    if(res)
    {
        const Node &field_node = node[field_name];
        res = blueprint::o2mrelation::verify(field_node,field_info);
        if(res)
        {
            log::info(info, protocol, log::quote(field_name) + "describes a one-to-many relation");
        }
        else
        {
            log::error(info, protocol, log::quote(field_name) + "doesn't describe a one-to-many relation");
        }
    }

    log::validation(field_info, res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_enum_field(const std::string &protocol,
                       const conduit::Node &node,
                       conduit::Node &info,
                       const std::string &field_name,
                       const std::vector<std::string> &enum_values )
{
    Node &field_info = (field_name != "") ? info[field_name] : info;

    bool res = verify_string_field(protocol, node, info, field_name);
    if(res)
    {
        const Node &field_node = (field_name != "") ? node[field_name] : node;

        const std::string field_value = field_node.as_string();
        bool is_field_enum = false;
        for(size_t i=0; i < enum_values.size(); i++)
        {
            is_field_enum |= (field_value == enum_values[i]);
        }

        if(is_field_enum)
        {
            log::info(info, protocol, log::quote(field_name) +
                                      "has valid value" +
                                      log::quote(field_value, 1));
        }
        else
        {
            log::error(info, protocol, log::quote(field_name) +
                                       "has invalid value" +
                                       log::quote(field_value, 1));
            res = false;
        }
    }

    log::validation(field_info, res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_reference_field(const std::string &protocol,
                            const conduit::Node &node_tree,
                            conduit::Node &info_tree,
                            const conduit::Node &node,
                            conduit::Node &info,
                            const std::string &field_name,
                            const std::string &ref_path)
{
    bool res = verify_string_field(protocol, node, info, field_name);
    if(res)
    {
        const std::string ref_name = node[field_name].as_string();

        if(!node_tree.has_child(ref_path) || !node_tree[ref_path].has_child(ref_name))
        {
            log::error(info, protocol, "reference to non-existent " + field_name +
                                        log::quote(ref_name, 1));
            res = false;
        }
        else if(info_tree[ref_path][ref_name]["valid"].as_string() != "true")
        {
            log::error(info, protocol, "reference to invalid " + field_name +
                                       log::quote(ref_name, 1));
            res = false;
        }
    }

    log::validation(info[field_name], res);
    log::validation(info, res);

    return res;
}

//-----------------------------------------------------------------------------
bool verify_poly_node(bool is_mixed_topo,
                      std::string name,
                      const conduit::Node &node,
                      conduit::Node &node_info,
                      const conduit::Node &topo,
                      conduit::Node &info,
                      bool &elems_res)
{
    const std::string protocol = "mesh::topology::unstructured";
    bool node_res = true;

    // Polygonal & Polyhedral shape
    if(node.has_child("shape") &&
       node["shape"].dtype().is_string() &&
       (node["shape"].as_string() == "polygonal" ||
       node["shape"].as_string() == "polyhedral"))
    {
        node_res &= blueprint::o2mrelation::verify(node, node_info);

        // Polyhedral - Check for subelements
        if (node["shape"].as_string() == "polyhedral")
        {
            bool subnode_res = true;
            if(!verify_object_field(protocol, topo, info, "subelements"))
            {
                subnode_res = false;
            }
            else
            {
                const Node &topo_subelems = topo["subelements"];
                Node &info_subelems = info["subelements"];
                bool has_subnames = topo_subelems.dtype().is_object();

                // Look for child "name" if mixed topology case,
                // otherwise look for "shape" variable.
                name = is_mixed_topo ? name : "shape";
                if(!topo_subelems.has_child(name))
                {
                    subnode_res = false;
                }
                // Checks for topo["subelements"]["name"]["shape"] with mixed topology,
                // or topo["subelements"]["shape"] with single topology,
                else
                {
                    const Node &sub_node  = is_mixed_topo ? topo_subelems[name] : topo_subelems;
                    Node &subnode_info =
                        !is_mixed_topo ? info_subelems :
                        has_subnames ? info["subelements"][name] :
                        info["subelements"].append();

                    if(sub_node.has_child("shape"))
                    {
                        subnode_res &= verify_field_exists(protocol, sub_node, subnode_info, "shape") &&
                        blueprint::mesh::topology::shape::verify(sub_node["shape"], subnode_info["shape"]);
                        subnode_res &= verify_integer_field(protocol, sub_node, subnode_info, "connectivity");
                        subnode_res &= sub_node["shape"].as_string() == "polygonal";
                        subnode_res &= blueprint::o2mrelation::verify(sub_node, subnode_info);
                    }
                    else
                    {
                        subnode_res = false;
                    }

                    log::validation(subnode_info,subnode_res);
                }
                log::validation(info_subelems, subnode_res);
            }
            elems_res &= subnode_res;
        }
    }
    node_res &= elems_res;
    return node_res;
}

//-----------------------------------------------------------------------------
std::string identify_coords_coordsys(const Node &coords)
{
    Node axes;
    NodeConstIterator itr = coords.children();
    while(itr.has_next())
    {
        itr.next();
        const std::string axis_name = itr.name();

        if(axis_name[0] == 'd' && axis_name.size() > 1)
        {
            axes[axis_name.substr(1, axis_name.length())];
        }
        else
        {
            axes[axis_name];
        }
    }

    std::string coordsys = "unknown";
    if(axes.has_child("theta") || axes.has_child("phi"))
    {
        coordsys = "spherical";
    }
    else if(axes.has_child("r")) // rz, or r w/o theta, phi
    {
        coordsys = "cylindrical";
    }
    else if(axes.has_child("x") || axes.has_child("y") || axes.has_child("z"))
    {
        coordsys = "cartesian";
    }
    else if(axes.has_child("i") || axes.has_child("j") || axes.has_child("k"))
    {
        coordsys = "logical";
    }
    return coordsys;
}

//-----------------------------------------------------------------------------
const Node& identify_coordset_coords(const Node &coordset)
{
    std::string coords_path = "";
    if(coordset["type"].as_string() == "uniform")
    {
        if(coordset.has_child("origin"))
        {
            coords_path = "origin";
        }
        else if(coordset.has_child("spacing"))
        {
            coords_path = "spacing";
        }
        else
        {
            coords_path = "dims";
        }
    }
    else
    {
        coords_path = "values";
    }
    return coordset[coords_path];
}

//-----------------------------------------------------------------------------
std::vector<std::string> identify_coordset_axes(const Node &coordset)
{
    // TODO(JRC): This whole set of coordinate system identification functions
    // could be revised to allow different combinations of axes to be specified
    // (e.g. (x, z), or even something like (z)).
    const Node &coords = identify_coordset_coords(coordset);
    const std::string coordset_coordsys = identify_coords_coordsys(coords);

    std::vector<std::string> coordset_axes;
    if(coordset_coordsys == "cartesian" || coordset_coordsys == "logical")
    {
        coordset_axes = conduit::blueprint::mesh::cartesian_axes;
    }
    else if(coordset_coordsys == "cylindrical")
    {
        coordset_axes = conduit::blueprint::mesh::cylindrical_axes;
    }
    else if(coordset_coordsys == "spherical")
    {
        coordset_axes = conduit::blueprint::mesh::spherical_axes;
    }

    return std::vector<std::string>(
        coordset_axes.begin(),
        coordset_axes.begin() + coords.number_of_children());
}

//-----------------------------------------------------------------------------
DataType find_widest_dtype(const Node &node,
                           const std::vector<DataType> &default_dtypes)
{
    DataType widest_dtype(default_dtypes[0].id(), 0, 0, 0, 0, default_dtypes[0].endianness());

    std::vector<const Node*> node_bag(1, &node);
    while(!node_bag.empty())
    {
        const Node *curr_node = node_bag.back(); node_bag.pop_back();
        const DataType curr_dtype = curr_node->dtype();
        if( curr_dtype.is_list() || curr_dtype.is_object() )
        {
            NodeConstIterator curr_node_it = curr_node->children();
            while(curr_node_it.has_next())
            {
                node_bag.push_back(&curr_node_it.next());
            }
        }
        else
        {
            for(index_t ti = 0; ti < (index_t)default_dtypes.size(); ti++)
            {
                const DataType &valid_dtype = default_dtypes[ti];
                bool is_valid_dtype =
                    (curr_dtype.is_floating_point() && valid_dtype.is_floating_point()) ||
                    (curr_dtype.is_signed_integer() && valid_dtype.is_signed_integer()) ||
                    (curr_dtype.is_unsigned_integer() && valid_dtype.is_unsigned_integer()) ||
                    (curr_dtype.is_string() && valid_dtype.is_string());
                if(is_valid_dtype && (widest_dtype.element_bytes() < curr_dtype.element_bytes()))
                {
                    widest_dtype.set(DataType(curr_dtype.id(), 1));
                }
            }
        }
    }

    bool no_type_found = widest_dtype.element_bytes() == 0;
    return no_type_found ? default_dtypes[0] : widest_dtype;
}

//-----------------------------------------------------------------------------
DataType find_widest_dtype(const Node &node,
                           const DataType &default_dtype)
{
    return find_widest_dtype(node, std::vector<DataType>(1, default_dtype));
}

//-----------------------------------------------------------------------------
Node link_nodes(const Node &lhs, const Node &rhs)
{
    Node linker;
    linker.append().set_external(lhs);
    linker.append().set_external(rhs);
    return linker;
}

//-----------------------------------------------------------------------------
bool find_reference_node(const Node &node, const std::string &ref_key, Node &ref)
{
    bool res = false;
    ref.reset();

    // NOTE: This segment of code is necessary to transform "topology" into
    // "topologies" while keeping all other dependency names (e.g. "coordset")
    // simply plural by just appending an "s" character.
    const std::string ref_section = (ref_key[ref_key.length()-1] != 'y') ?
        ref_key + "s" : ref_key.substr(0, ref_key.length()-1) + "ies";

    if(node.has_child(ref_key))
    {
        const std::string &ref_value = node.fetch(ref_key).as_string();

        const Node *traverse_node = node.parent();
        while(traverse_node != NULL)
        {
            if(traverse_node->has_child(ref_section))
            {
                const Node &ref_parent = traverse_node->fetch(ref_section);
                if(ref_parent.has_child(ref_value))
                {
                    ref.set_external(ref_parent[ref_value]);
                    res = true;
                }
                break;
            }
            traverse_node = traverse_node->parent();
        }
    }

    return res;
}

//-----------------------------------------------------------------------------
index_t get_coordset_length(const std::string &type,
                            const Node &coordset)
{
    index_t coordset_length = 1;

    const std::vector<std::string> csys_axes = identify_coordset_axes(coordset);
    const std::vector<std::string> &logical_axes = blueprint::mesh::logical_axes;
    for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
    {
        if(type == "uniform")
        {
            coordset_length *=
                coordset["dims"][logical_axes[i]].to_int64();
        }
        else if(type == "rectilinear")
        {
            coordset_length *=
                coordset["values"][csys_axes[i]].dtype().number_of_elements();
        }
        else // if(type == "explicit")
        {
            coordset_length =
                coordset["values"][csys_axes[i]].dtype().number_of_elements();
        }
    }

    return coordset_length;
}

//-----------------------------------------------------------------------------
// void get_offset_topology(const Node &topology,
//                          Node &otopology)
// {
//     // NOTE(JRC): Unfortunately, this method doesn't work for caching the offsets
//     // array because the given topology doesn't have the same tree context as the
//     // original, which causes procedures like 'find_reference_node' to fail.
//     otopology.reset();
//     otopology.set_external(topology);
//
//     if(topology.has_child("elements") && !topology["elements"].has_child("offsets"))
//     {
//         Node &offsets = otopology["elements/offsets"];
//         blueprint::mesh::topology::unstructured::generate_offsets(otopology, offsets);
//     }
// }

//-----------------------------------------------------------------------------
void get_topology_offsets(const Node &topology,
                          Node &offsets)
{
    // TODO(JRC): This solution suffers from performance issues when multiple
    // calls in a trace need the offset array and it isn't provided by the original
    // caller (e.g. generate_sides() will make generate offsets, and so will its
    // callee methods generate_centroids() and generate_edges()). This issue should
    // be fixed if possible (or at least made more obnoxious to callers).

    offsets.reset();

    if(topology.has_child("type") && topology["type"].as_string() == "unstructured")
    {
        if(topology["elements"].has_child("offsets") && !topology["elements/offsets"].dtype().is_empty())
        {
            offsets.set_external(topology["elements/offsets"]);
        }
        else
        {
            blueprint::mesh::topology::unstructured::generate_offsets(topology, offsets);
        }
    }
}

//-----------------------------------------------------------------------------
index_t get_topology_length(const std::string &type,
                            const Node &topology)
{
    index_t topology_length = 1;

    if(type == "uniform" || type == "rectilinear")
    {
        Node coordset;
        find_reference_node(topology, "coordset", coordset);

        const std::vector<std::string> csys_axes = identify_coordset_axes(coordset);
        const std::vector<std::string> &logical_axes = blueprint::mesh::logical_axes;
        for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
        {
            topology_length *= ((type == "uniform") ?
                coordset["dims"][logical_axes[i]].to_int64() :
                coordset["values"][csys_axes[i]].dtype().number_of_elements()) - 1;
        }
    }
    else if(type == "structured")
    {
        const Node &dims = topology["elements/dims"];

        const std::vector<std::string> &logical_axes = blueprint::mesh::logical_axes;
        for(index_t i = 0; i < (index_t)dims.number_of_children(); i++)
        {
            topology_length *= dims[logical_axes[i]].to_int64();
        }
    }
    else // if(type == "unstructured")
    {
        Node topo_offsets;
        get_topology_offsets(topology, topo_offsets);
        topology_length = topo_offsets.dtype().number_of_elements();
    }

    return topology_length;
}

//-----------------------------------------------------------------------------
// - end internal data function helpers -
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// - start internal topology helpers -
//-----------------------------------------------------------------------------

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

    // NOTE(JRC): This type current only works at forming associations within
    // an unstructured topology's hierarchy.
    TopologyMetadata(const conduit::Node &topology, const conduit::Node &coordset) :
        topo(&topology), cset(&coordset),
        int_dtype(find_widest_dtype(link_nodes(topology, coordset), conduit::blueprint::mesh::default_int_dtypes)),
        float_dtype(find_widest_dtype(link_nodes(topology, coordset), conduit::blueprint::mesh::default_float_dtype)),
        topo_cascade(topology), topo_shape(topology)
    {
        Node topo_offsets;
        get_topology_offsets(topology, topo_offsets);
        const index_t topo_num_elems = topo_offsets.dtype().number_of_elements();
        const index_t topo_num_coords = get_coordset_length("unstructured", coordset);

        // Allocate Data Templates for Outputs //

        dim_topos.resize(topo_shape.dim + 1);
        dim_geid_maps.resize(topo_shape.dim + 1);
        dim_geassocs_maps.resize(topo_shape.dim + 1);
        dim_leassocs_maps.resize(topo_shape.dim + 1);
        dim_le2ge_maps.resize(topo_shape.dim + 1);

        for(index_t di = 0; di < topo_shape.dim; di++)
        {
            Node &dim_topo = dim_topos[di];
            dim_topo.reset();
            dim_topo["type"].set("unstructured");
            dim_topo["coordset"].set(topology["coordset"].as_string());
            dim_topo["elements/shape"].set(topo_cascade.get_shape(di).type);
        }
        // NOTE: This is done so that the index values for the top-level entities
        // can be extracted by the 'get_entity_data' function before DFS
        // processing below.
        dim_topos[topo_shape.dim].set_external(topology);
        dim_topos[topo_shape.dim]["elements/offsets"].set(topo_offsets);
        std::vector< std::vector<int64> > dim_buffers(topo_shape.dim + 1);

        // Prepare Initial Values for Processing //

        // Temporary nodes to manage pointers to important information (temp) and
        // associated conversations (data).
        Node temp, data;

        // NOTE(JRC): A 'deque' is used so that queue behavior (FIFO)
        // is responsible for ordering the identifiers in the cascade of entities,
        // which more closely follows the average intuition.
        const index_t bag_num_elems = topo_num_coords + topo_num_elems;
        std::deque< std::vector<int64> > entity_index_bag(bag_num_elems);
        std::deque< index_t > entity_dim_bag(bag_num_elems, -1);
        std::deque< std::vector< std::pair<int64, int64> > > entity_parent_bag(bag_num_elems);

        // NOTE(JRC): We start with processing the points of the topology followed
        // by the top-level elements in order to ensure that order is preserved
        // relative to the original topology for these entities.
        for(index_t pi = 0; pi < topo_num_coords; pi++)
        {
            index_t bi = pi;
            entity_index_bag[bi].push_back(bi);
            entity_dim_bag[bi] = 0;
        }
        for(index_t ei = 0; ei < topo_num_elems; ei++)
        {
            index_t bi = topo_num_coords + ei;

            temp.reset();
            get_entity_data(TopologyMetadata::GLOBAL, ei, topo_shape.dim, temp);

            std::vector<int64> &elem_indices = entity_index_bag[bi];
            elem_indices.resize(temp.dtype().number_of_elements());
            data.set_external(DataType::int64(elem_indices.size()), &elem_indices[0]);
            temp.to_int64_array(data);

            entity_dim_bag[bi] = topo_shape.dim;
        }

        while(!entity_index_bag.empty())
        {
            std::vector<int64> entity_indices = entity_index_bag.front();
            entity_index_bag.pop_front();
            index_t entity_dim = entity_dim_bag.front();
            entity_dim_bag.pop_front();
            std::vector< std::pair<int64, int64> > entity_parents = entity_parent_bag.front();
            entity_parent_bag.pop_front();

            std::vector<int64> &dim_buffer = dim_buffers[entity_dim];
            std::map< std::set<index_t>, index_t > &dim_geid_map = dim_geid_maps[entity_dim];
            auto &dim_geassocs = dim_geassocs_maps[entity_dim];
            auto &dim_leassocs = dim_leassocs_maps[entity_dim];
            std::vector<index_t> &dim_le2ge_map = dim_le2ge_maps[entity_dim];
            ShapeType dim_shape = topo_cascade.get_shape(entity_dim);

            // Add Element to Topology //

            // NOTE: This code assumes that all entities can be uniquely
            // identified by the list of coordinate indices of which they
            // are comprised. This is certainly true of all implicit topologies
            // and of 2D polygonal topologies, but it may not be always the
            // case for 3D polygonal topologies.
            std::set<int64> vert_ids;
            if(!dim_shape.is_polyhedral())
            {
                vert_ids = std::set<int64>(entity_indices.begin(), entity_indices.end());
            }
            else // if(dim_shape.is_polyhedral())
            {
                const index_t elem_outer_count = entity_indices.size();
                for(index_t oi = 0; oi < elem_outer_count; oi++)
                {
                    temp.set_external((*topo)["subelements/offsets"]);
                    data.set_external(int_dtype, temp.element_ptr(entity_indices[oi]));
                    const index_t elem_inner_offset = data.to_index_t();

                    temp.set_external((*topo)["subelements/sizes"]);
                    data.set_external(int_dtype, temp.element_ptr(entity_indices[oi]));
                    const index_t elem_inner_count = data.to_index_t();

                    for(index_t ii = 0; ii < elem_inner_count; ii++)
                    {
                        temp.set_external((*topo)["subelements/connectivity"]);
                        data.set_external(int_dtype, temp.element_ptr(elem_inner_offset + ii));
                        const index_t vi = data.to_int64();
                        vert_ids.insert(vi);
                    }
                }
            }

            const index_t local_id = dim_leassocs.size();
            if(dim_geid_map.find(vert_ids) == dim_geid_map.end())
            {
                const index_t global_id = dim_geassocs.size();
                dim_buffer.insert(dim_buffer.end(), entity_indices.begin(), entity_indices.end());
                dim_geid_map[vert_ids] = global_id;
            }
            const index_t global_id = dim_geid_map.find(vert_ids)->second;

            { // create_entity(global_id, local_id, entity_dim)
                if((index_t)dim_geassocs.size() <= global_id)
                {
                    dim_geassocs.resize(global_id + 1);
                }
                if((index_t)dim_leassocs.size() <= local_id)
                {
                    dim_leassocs.resize(local_id + 1);
                }
                if((index_t)dim_le2ge_map.size() <= local_id)
                {
                    dim_le2ge_map.resize(local_id + 1);
                }
                dim_le2ge_map[local_id] = global_id;
            }

            // Add Element to Associations //

            add_entity_assoc(IndexType::GLOBAL, global_id, entity_dim, global_id, entity_dim);
            add_entity_assoc(IndexType::LOCAL, local_id, entity_dim, local_id, entity_dim);
            for(index_t pi = 0; pi < (index_t)entity_parents.size(); pi++)
            {
                index_t plevel = entity_parents.size() - pi - 1;
                const index_t parent_global_id = entity_parents[plevel].first;
                const index_t parent_local_id = entity_parents[plevel].second;
                const index_t parent_dim = entity_dim + pi + 1;
                add_entity_assoc(IndexType::GLOBAL, global_id, entity_dim, parent_global_id, parent_dim);
                add_entity_assoc(IndexType::LOCAL, local_id, entity_dim, parent_local_id, parent_dim);
            }

            // Add Embedded Elements for Further Processing //

            if(entity_dim > 0)
            {
                std::vector< std::pair<int64, int64> > embed_parents = entity_parents;
                embed_parents.push_back(std::make_pair(global_id, local_id));
                ShapeType embed_shape = topo_cascade.get_shape(entity_dim - 1);

                index_t elem_outer_count = dim_shape.is_poly() ?
                    entity_indices.size() : dim_shape.embed_count;

                // NOTE(JRC): This is horribly complicated for the poly case and needs
                // to be refactored so that it's legible. There's a lot of overlap in
                // used variables where it feels unnecessary (e.g. 'poly' being
                // shoehorned into using 'implicit' variables), for example.
                for(index_t oi = 0, ooff = 0; oi < elem_outer_count; oi++)
                {
                    index_t elem_inner_count = embed_shape.indices;

                    if (dim_shape.is_polyhedral())
                    {
                        const Node &subelem_off_const = (*topo)["subelements/offsets"];
                        const Node &subelem_size_const = (*topo)["subelements/sizes"];

                        Node subelem_off; subelem_off.set_external(subelem_off_const);
                        Node subelem_size; subelem_size.set_external(subelem_size_const);

                        temp.set_external(int_dtype,
                            subelem_off.element_ptr(entity_indices[oi]));
                        ooff = temp.to_int64();
                        temp.set_external(int_dtype,
                            subelem_size.element_ptr(entity_indices[oi]));
                        elem_inner_count = temp.to_int64();
                    }

                    std::vector<int64> embed_indices;
                    for(index_t ii = 0; ii < elem_inner_count; ii++)
                    {
                        index_t ioff = ooff + (dim_shape.is_poly() ?
                            ii : dim_shape.embedding[oi * elem_inner_count + ii]);

                        if (dim_shape.is_polyhedral())
                        {
                            const Node &subelem_conn_const = (*topo)["subelements/connectivity"];
                            Node subelem_conn; subelem_conn.set_external(subelem_conn_const);

                            temp.set_external(int_dtype,
                                subelem_conn.element_ptr(ioff));
                            embed_indices.push_back(temp.to_int64());
                        }
                        else
                        {
                            embed_indices.push_back(
                                entity_indices[ioff % entity_indices.size()]);
                        }
                    }

                    ooff += dim_shape.is_polygonal() ? 1 : 0;

                    entity_index_bag.push_back(embed_indices);
                    entity_dim_bag.push_back(embed_shape.dim);
                    entity_parent_bag.push_back(embed_parents);
                }
            }
        }

        // Move Topological Data into Per-Dim Nodes //

        for(index_t di = 0; di <= topo_shape.dim; di++)
        {
            Node &dim_conn = dim_topos[di]["elements/connectivity"];
            Node data_conn(DataType::int64(dim_buffers[di].size()),
                &(dim_buffers[di][0]), true);

            dim_conn.set(DataType(int_dtype.id(), dim_buffers[di].size()));
            data_conn.to_data_type(int_dtype.id(), dim_conn);

            // Initialize element/sizes for polygonal mesh using polyhedral's
            // subelement/sizes
            if(di == 2 && topo_shape.is_polyhedral())
            {
                Node &polygonal_size = dim_topos[di]["elements/sizes"];
                Node &polyhedral_subsize = dim_topos[3]["subelements/sizes"];
                if (polygonal_size.dtype().is_empty())
                {
                    polygonal_size = polyhedral_subsize;
                }
            }

            get_topology_offsets(dim_topos[di], dim_topos[di]["elements/offsets"]);
        }
    }

    void add_entity_assoc(IndexType type,
        index_t e0_id, index_t e0_dim,
        index_t e1_id, index_t e1_dim)
    {
        auto &assoc_maps = (type == IndexType::LOCAL) ? dim_leassocs_maps : dim_geassocs_maps;
        std::vector< std::pair< std::vector<index_t>, std::set<index_t> > > *entity_assocs[2] = {
            &assoc_maps[e0_dim][e0_id],
            &assoc_maps[e1_dim][e1_id]
        };

        for(index_t ai = 0; ai < 2; ai++)
        {
            auto &curr_assocs = *entity_assocs[ai];
            curr_assocs.resize(topo_shape.dim + 1);

            const index_t cross_id = (ai == 0) ? e1_id : e0_id;
            const index_t cross_dim = (ai == 0) ? e1_dim : e0_dim;
            auto &cross_assocs = curr_assocs[cross_dim];
            if(cross_assocs.second.find(cross_id) == cross_assocs.second.end())
            {
                cross_assocs.first.push_back(cross_id);
                cross_assocs.second.insert(cross_id);
            }
        }
    }

    const std::vector<index_t>& get_entity_assocs(IndexType type,
        index_t entity_id, index_t entity_dim, index_t assoc_dim) const
    {
        auto &dim_assocs = (type == IndexType::LOCAL) ? dim_leassocs_maps : dim_geassocs_maps;
        return dim_assocs[entity_dim][entity_id][assoc_dim].first;
    }

    void get_dim_map(IndexType type,
        index_t src_dim, index_t dst_dim, Node &map_node) const
    {
        auto &dim_assocs = (type == IndexType::LOCAL) ? dim_leassocs_maps : dim_geassocs_maps;

        std::vector<index_t> values, sizes, offsets;
        for(index_t sdi = 0, so = 0; sdi < (index_t)dim_assocs[src_dim].size(); sdi++, so += sizes.back())
        {
            const std::vector<index_t> &src_assocs = get_entity_assocs(type, sdi, src_dim, dst_dim);
            values.insert(values.end(), src_assocs.begin(), src_assocs.end());
            sizes.push_back((index_t)src_assocs.size());
            offsets.push_back(so);
        }

        std::vector<index_t>* path_data[] = { &values, &sizes, &offsets };
        std::string path_names[] = { "values", "sizes", "offsets" };
        const index_t path_count = sizeof(path_data) / sizeof(path_data[0]);
        for(index_t pi = 0; pi < path_count; pi++)
        {
            Node data;
            data.set(*path_data[pi]);
            data.to_data_type(int_dtype.id(), map_node[path_names[pi]]);
        }
    }

    void get_entity_data(IndexType type,
        index_t entity_id, index_t entity_dim, Node &data) const
    {
        Node temp;

        // NOTE(JRC): This is done in order to get around 'const' casting for
        // data pointers that won't be changed by the function anyway.
        Node dim_conn; dim_conn.set_external(dim_topos[entity_dim]["elements/connectivity"]);
        Node dim_off; dim_off.set_external(dim_topos[entity_dim]["elements/offsets"]);

        const DataType conn_dtype(dim_conn.dtype().id(), 1);
        const DataType off_dtype(dim_off.dtype().id(), 1);
        const DataType data_dtype = data.dtype().is_number() ? data.dtype() : DataType::int64(1);

        const index_t entity_gid = (type == IndexType::LOCAL) ?
            dim_le2ge_maps[entity_dim][entity_id] : entity_id;
        temp.set_external(off_dtype, dim_off.element_ptr(entity_gid));
        index_t entity_start_index = temp.to_int64();
        temp.set_external(off_dtype, dim_off.element_ptr(entity_gid + 1));
        index_t entity_end_index = (entity_gid < get_length(entity_dim) - 1) ?
            temp.to_int64() : dim_conn.dtype().number_of_elements();

        index_t entity_size = entity_end_index - entity_start_index;
        temp.set_external(DataType(conn_dtype.id(), entity_size),
            dim_conn.element_ptr(entity_start_index));
        temp.to_data_type(data_dtype.id(), data);
    }

    void get_point_data(IndexType type,
        index_t point_id, Node &data) const
    {
        const index_t point_gid = (type == IndexType::LOCAL) ?
            dim_le2ge_maps[0][point_id] : point_id;

        if(data.dtype().is_empty())
        {
            data.set(DataType::float64(3));
        }
        const DataType data_dtype(data.dtype().id(), 1);

        Node temp1, temp2;
        const std::vector<std::string> csys_axes = identify_coordset_axes(*cset);
        for(index_t di = 0; di < topo_shape.dim; di++)
        {
            temp1.set_external(float_dtype,
                (void*)(*cset)["values"][csys_axes[di]].element_ptr(point_gid));
            temp2.set_external(data_dtype, data.element_ptr(di));
            temp1.to_data_type(data_dtype.id(), temp2);
        }
    }

    index_t get_length(index_t dim=-1) const
    {
        // NOTE: The default version of 'get_length' gets the total length of all
        // unique entities in the topology. The parameterized version fetches the
        // length for just that parameter's dimension.

        index_t start_dim = (dim >= 0) ? dim : 0;
        index_t end_dim = (dim >= 0) ? dim : topo_shape.dim;

        index_t topo_length = 0;
        for(index_t di = start_dim; di <= end_dim; di++)
        {
            topo_length += get_topology_length("unstructured", dim_topos[di]);
        }

        return topo_length;
    }

    index_t get_embed_length(index_t entity_dim, index_t embed_dim) const
    {
        // NOTE: The default version of 'get_embed_length' gets the total number of
        // embeddings for each entity at the top level to the embedding level. The
        // parameterized version just fetches the number of embeddings for one
        // specific entity at the top level.

        std::vector<index_t> entity_index_bag;
        std::vector<index_t> entity_dim_bag;
        for(index_t ei = 0; ei < this->get_length(entity_dim); ei++)
        {
            entity_index_bag.push_back(ei);
            entity_dim_bag.push_back(entity_dim);
        }

        std::set<index_t> embed_set;
        index_t embed_length = 0;
        while(!entity_index_bag.empty())
        {
            index_t entity_index = entity_index_bag.back();
            entity_index_bag.pop_back();
            index_t entity_dim_back = entity_dim_bag.back();
            entity_dim_bag.pop_back();

            if(entity_dim_back == embed_dim)
            {
                if(embed_set.find(entity_index) == embed_set.end())
                {
                    embed_length++;
                }
                embed_set.insert(entity_index);
            }
            else
            {
                const std::vector<index_t> &embed_ids = get_entity_assocs(
                    TopologyMetadata::LOCAL, entity_index, entity_dim_back, entity_dim_back - 1);
                for(index_t ei = 0; ei < (index_t)embed_ids.size(); ei++)
                {
                    entity_index_bag.push_back(embed_ids[ei]);
                    entity_dim_bag.push_back(entity_dim_back - 1);
                }
            }
        }

        return embed_length;
    }

    std::string to_json() const
    {
        Node mesh;

        Node &mesh_coords = mesh["coordsets"][(*topo)["coordset"].as_string()];
        mesh_coords.set_external(*cset);

        Node &mesh_topos = mesh["topologies"];
        for(index_t di = 0; di <= topo_shape.dim; di++)
        {
            std::ostringstream oss;
            oss << "d" << di;
            mesh_topos[oss.str()].set_external(dim_topos[di]);
        }

        return mesh.to_json();
    }

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

//-------------------------------------------------------------------------
void
convert_coordset_to_rectilinear(const std::string &/*base_type*/,
                                const conduit::Node &coordset,
                                conduit::Node &dest)
{
    // bool is_base_uniform = true;

    dest.reset();
    dest["type"].set("rectilinear");

    DataType float_dtype = find_widest_dtype(coordset, blueprint::mesh::default_float_dtype);

    std::vector<std::string> csys_axes = identify_coordset_axes(coordset);
    const std::vector<std::string> &logical_axes = blueprint::mesh::logical_axes;
    for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
    {
        const std::string& csys_axis = csys_axes[i];
        const std::string& logical_axis = logical_axes[i];

        float64 dim_origin = coordset.has_child("origin") ?
            coordset["origin"][csys_axis].to_float64() : 0.0;
        float64 dim_spacing = coordset.has_child("spacing") ?
            coordset["spacing"]["d"+csys_axis].to_float64() : 1.0;
        index_t dim_len = coordset["dims"][logical_axis].to_int64();

        Node &dst_cvals_node = dest["values"][csys_axis];
        dst_cvals_node.set(DataType(float_dtype.id(), dim_len));

        Node src_cval_node, dst_cval_node;
        for(index_t d = 0; d < dim_len; d++)
        {
            src_cval_node.set(dim_origin + d * dim_spacing);
            dst_cval_node.set_external(float_dtype, dst_cvals_node.element_ptr(d));
            src_cval_node.to_data_type(float_dtype.id(), dst_cval_node);
        }
    }
}

//-------------------------------------------------------------------------
void
convert_coordset_to_explicit(const std::string &base_type,
                             const conduit::Node &coordset,
                             conduit::Node &dest)
{
    bool is_base_rectilinear = base_type == "rectilinear";
    bool is_base_uniform = base_type == "uniform";

    dest.reset();
    dest["type"].set("explicit");

    DataType float_dtype = find_widest_dtype(coordset, blueprint::mesh::default_float_dtype);

    std::vector<std::string> csys_axes = identify_coordset_axes(coordset);
    const std::vector<std::string> &logical_axes = blueprint::mesh::logical_axes;

    index_t dim_lens[3] = { 0,0,0 } , coords_len = 1;
    for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
    {
        dim_lens[i] = is_base_rectilinear ?
            coordset["values"][csys_axes[i]].dtype().number_of_elements() :
            coordset["dims"][logical_axes[i]].to_int64();
        coords_len *= dim_lens[i];
    }

    Node info;
    for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
    {
        const std::string& csys_axis = csys_axes[i];

        // NOTE: The following values are specific to the
        // rectilinear transform case.
        const Node &src_cvals_node = coordset.has_child("values") ?
            coordset["values"][csys_axis] : info;
        // NOTE: The following values are specific to the
        // uniform transform case.
        float64 dim_origin = coordset.has_child("origin") ?
            coordset["origin"][csys_axis].to_float64() : 0.0;
        float64 dim_spacing = coordset.has_child("spacing") ?
            coordset["spacing"]["d"+csys_axis].to_float64() : 1.0;

        index_t dim_block_size = 1, dim_block_count = 1;
        for(index_t j = 0; j < (index_t)csys_axes.size(); j++)
        {
            dim_block_size *= (j < i) ? dim_lens[j] : 1;
            dim_block_count *= (i < j) ? dim_lens[j] : 1;
        }

        Node &dst_cvals_node = dest["values"][csys_axis];
        dst_cvals_node.set(DataType(float_dtype.id(), coords_len));

        Node src_cval_node, dst_cval_node;
        for(index_t d = 0; d < dim_lens[i]; d++)
        {
            index_t doffset = d * dim_block_size;
            for(index_t b = 0; b < dim_block_count; b++)
            {
                index_t boffset = b * dim_block_size * dim_lens[i];
                for(index_t bi = 0; bi < dim_block_size; bi++)
                {
                    index_t ioffset = doffset + boffset + bi;
                    dst_cval_node.set_external(float_dtype,
                        dst_cvals_node.element_ptr(ioffset));

                    if(is_base_rectilinear)
                    {
                        src_cval_node.set_external(
                            DataType(src_cvals_node.dtype().id(), 1),
                            (void*)src_cvals_node.element_ptr(d));
                    }
                    else if(is_base_uniform)
                    {
                        src_cval_node.set(dim_origin + d * dim_spacing);
                    }

                    src_cval_node.to_data_type(float_dtype.id(), dst_cval_node);
                }
            }
        }
    }
}

// TODO(JRC): For all of the following topology conversion functions, it's
// possible if the user validates the topology in isolation that it can be
// good and yet the conversion will fail due to an invalid reference coordset.
// In order to eliminate this concern, it may be better to update the mesh
// verify code so that "topology::verify" verifies reference fields, which
// would enable more assurances.

//-------------------------------------------------------------------------
void
convert_topology_to_rectilinear(const std::string &/*base_type*/,
                                const conduit::Node &topo,
                                conduit::Node &dest,
                                conduit::Node &cdest)
{
    // bool is_base_uniform = true;

    dest.reset();
    cdest.reset();

    Node coordset;
    find_reference_node(topo, "coordset", coordset);
    blueprint::mesh::coordset::uniform::to_rectilinear(coordset, cdest);

    dest.set(topo);
    dest["type"].set("rectilinear");
    dest["coordset"].set(cdest.name());
}

//-------------------------------------------------------------------------
void
convert_topology_to_structured(const std::string &base_type,
                               const conduit::Node &topo,
                               conduit::Node &dest,
                               conduit::Node &cdest)
{
    bool is_base_rectilinear = base_type == "rectilinear";
    bool is_base_uniform = base_type == "uniform";

    dest.reset();
    cdest.reset();

    Node coordset;
    find_reference_node(topo, "coordset", coordset);
    if(is_base_rectilinear)
    {
        blueprint::mesh::coordset::rectilinear::to_explicit(coordset, cdest);
    }
    else if(is_base_uniform)
    {
        blueprint::mesh::coordset::uniform::to_explicit(coordset, cdest);
    }

    dest["type"].set("structured");
    dest["coordset"].set(cdest.name());
    if(topo.has_child("origin"))
    {
        dest["origin"].set(topo["origin"]);
    }

    // TODO(JRC): In this case, should we reach back into the coordset
    // and use its types to inform those of the topology?
    DataType int_dtype = find_widest_dtype(topo, blueprint::mesh::default_int_dtypes);

    std::vector<std::string> csys_axes = identify_coordset_axes(coordset);
    const std::vector<std::string> &logical_axes = blueprint::mesh::logical_axes;
    for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
    {
        Node src_dlen_node;
        src_dlen_node.set(is_base_uniform ?
            coordset["dims"][logical_axes[i]].to_int64() :
            coordset["values"][csys_axes[i]].dtype().number_of_elements());
        // NOTE: The number of elements in the topology is one less
        // than the number of points along each dimension.
        src_dlen_node.set(src_dlen_node.to_int64() - 1);

        Node &dst_dlen_node = dest["elements/dims"][logical_axes[i]];
        src_dlen_node.to_data_type(int_dtype.id(), dst_dlen_node);
    }
}

//-------------------------------------------------------------------------
void
convert_topology_to_unstructured(const std::string &base_type,
                                 const conduit::Node &topo,
                                 conduit::Node &dest,
                                 conduit::Node &cdest)
{
    bool is_base_structured = base_type == "structured";
    bool is_base_rectilinear = base_type == "rectilinear";
    bool is_base_uniform = base_type == "uniform";

    dest.reset();
    cdest.reset();

    Node coordset;
    find_reference_node(topo, "coordset", coordset);
    if(is_base_structured)
    {
        cdest.set(coordset);
    }
    else if(is_base_rectilinear)
    {
        blueprint::mesh::coordset::rectilinear::to_explicit(coordset, cdest);
    }
    else if(is_base_uniform)
    {
        blueprint::mesh::coordset::uniform::to_explicit(coordset, cdest);
    }

    dest["type"].set("unstructured");
    dest["coordset"].set(cdest.name());
    if(topo.has_child("origin"))
    {
        dest["origin"].set(topo["origin"]);
    }

    // TODO(JRC): In this case, should we reach back into the coordset
    // and use its types to inform those of the topology?
    DataType int_dtype = find_widest_dtype(topo, blueprint::mesh::default_int_dtypes);

    std::vector<std::string> csys_axes = identify_coordset_axes(coordset);
    dest["elements/shape"].set(
        (csys_axes.size() == 1) ? "line" : (
        (csys_axes.size() == 2) ? "quad" : (
        (csys_axes.size() == 3) ? "hex"  : "")));
    const std::vector<std::string> &logical_axes = blueprint::mesh::logical_axes;

    index_t edims_axes[3] = {1, 1, 1};
    if(is_base_structured)
    {
        const conduit::Node &dim_node = topo["elements/dims"];
        for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
        {
            edims_axes[i] = dim_node[logical_axes[i]].to_int();
        }
    }
    else if(is_base_rectilinear)
    {
        const conduit::Node &dim_node = coordset["values"];
        for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
        {
            edims_axes[i] =
                dim_node[csys_axes[i]].dtype().number_of_elements() - 1;
        }
    }
    else if(is_base_uniform)
    {
        const conduit::Node &dim_node = coordset["dims"];
        for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
        {
            edims_axes[i] = dim_node[logical_axes[i]].to_int() - 1;
        }
    }

    index_t vdims_axes[3] = {1, 1, 1}, num_elems = 1;
    for(index_t d = 0; d < 3; d++)
    {
        num_elems *= edims_axes[d];
        vdims_axes[d] = edims_axes[d] + 1;
    }
    index_t indices_per_elem = (index_t) pow(2, csys_axes.size());

    conduit::Node &conn_node = dest["elements/connectivity"];
    conn_node.set(DataType(int_dtype.id(), num_elems * indices_per_elem));

    Node src_idx_node, dst_idx_node;
    index_t curr_elem[3], curr_vert[3];
    for(index_t e = 0; e < num_elems; e++)
    {
        grid_id_to_ijk(e, &edims_axes[0], &curr_elem[0]);

        // NOTE(JRC): In order to get all adjacent vertices for the
        // element, we use the bitwise interpretation of each index
        // per element to inform the direction (e.g. 5, which is
        // 101 bitwise, means (z+1, y+0, x+1)).
        for(index_t i = 0, v = 0; i < indices_per_elem; i++)
        {
            memcpy(&curr_vert[0], &curr_elem[0], 3 * sizeof(index_t));
            for(index_t d = 0; d < (index_t)csys_axes.size(); d++)
            {
                curr_vert[d] += (i & (index_t)pow(2, d)) >> d;
            }
            grid_ijk_to_id(&curr_vert[0], &vdims_axes[0], v);

            src_idx_node.set(v);
            dst_idx_node.set_external(int_dtype,
                conn_node.element_ptr(e * indices_per_elem + i));
            src_idx_node.to_data_type(int_dtype.id(), dst_idx_node);
        }

        // TODO(JRC): This loop inverts quads/hexes to conform to
        // the default Blueprint ordering. Once the ordering transforms
        // are introduced, this code should be removed and replaced
        // with initializing the ordering label value.
        for(index_t p = 2; p < indices_per_elem; p += 4)
        {
            index_t p1 = e * indices_per_elem + p;
            index_t p2 = e * indices_per_elem + p + 1;

            Node t1, t2, t3;
            t1.set(int_dtype, conn_node.element_ptr(p1));
            t2.set(int_dtype, conn_node.element_ptr(p2));

            t3.set_external(int_dtype, conn_node.element_ptr(p1));
            t2.to_data_type(int_dtype.id(), t3);
            t3.set_external(int_dtype, conn_node.element_ptr(p2));
            t1.to_data_type(int_dtype.id(), t3);
        }
    }
}

// NOTE(JRC): The following two functions need to be passed the coordinate set
// and can't use 'find_reference_node' because these internal functions aren't
// guaranteed to be passed nodes that exist in the context of an existing mesh
// tree ('generate_corners' has a good example wherein an in-situ edge topology
// is used to contruct an in-situ centroid topology).

//-------------------------------------------------------------------------
void
calculate_unstructured_centroids(const conduit::Node &topo,
                                 const conduit::Node &coordset,
                                 conduit::Node &dest,
                                 conduit::Node &cdest)
{
    // NOTE(JRC): This is a stand-in implementation for the method
    // 'mesh::topology::unstructured::generate_centroids' that exists because there
    // is currently no good way in Blueprint to create mappings with sparse data.
    const std::vector<std::string> csys_axes = identify_coordset_axes(coordset);

    Node topo_offsets;
    get_topology_offsets(topo, topo_offsets);
    const index_t topo_num_elems = topo_offsets.dtype().number_of_elements();

    const ShapeCascade topo_cascade(topo);
    const ShapeType &topo_shape = topo_cascade.get_shape();

    Node topo_sizes;
    if (topo_shape.is_poly())
    {
      topo_sizes = topo["elements/sizes"];
    }

    Node topo_subconn;
    Node topo_subsizes;
    Node topo_suboffsets;
    if (topo_shape.is_polyhedral())
    {
        const Node &topo_subconn_const = topo["subelements/connectivity"];
        topo_subconn.set_external(topo_subconn_const);
        topo_subsizes = topo["subelements/sizes"];
        topo_suboffsets = topo["subelements/offsets"];
    }

    // Discover Data Types //

    DataType int_dtype, float_dtype;
    {
        conduit::Node src_node;
        src_node["topology"].set_external(topo);
        src_node["coordset"].set_external(coordset);
        int_dtype = find_widest_dtype(src_node, blueprint::mesh::default_int_dtypes);
        float_dtype = find_widest_dtype(src_node, blueprint::mesh::default_float_dtype);
    }

    const Node &topo_conn_const = topo["elements/connectivity"];
    Node topo_conn; topo_conn.set_external(topo_conn_const);
    const DataType conn_dtype(topo_conn.dtype().id(), 1);
    const DataType offset_dtype(topo_offsets.dtype().id(), 1);
    const DataType size_dtype(topo_sizes.dtype().id(), 1);

    const DataType subconn_dtype(topo_subconn.dtype().id(), 1);
    const DataType suboffset_dtype(topo_suboffsets.dtype().id(), 1);
    const DataType subsize_dtype(topo_subsizes.dtype().id(), 1);

    // Allocate Data Templates for Outputs //

    dest.reset();
    dest["type"].set("unstructured");
    dest["coordset"].set(cdest.name());
    dest["elements/shape"].set(topo_cascade.get_shape(0).type);
    dest["elements/connectivity"].set(DataType(int_dtype.id(), topo_num_elems));

    cdest.reset();
    cdest["type"].set("explicit");
    for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
    {
        cdest["values"][csys_axes[ai]].set(DataType(float_dtype.id(), topo_num_elems));
    }

    // Compute Data for Centroid Topology //

    Node data_node;
    for(index_t ei = 0; ei < topo_num_elems; ei++)
    {
        index_t esize = 0;
        if (topo_shape.is_polygonal())
        {
            data_node.set_external(size_dtype, topo_sizes.element_ptr(ei));
            esize = data_node.to_int64();
        }
        data_node.set_external(offset_dtype, topo_offsets.element_ptr(ei));
        const index_t eoffset = data_node.to_int64();

        if (topo_shape.is_polyhedral())
        {
            data_node.set_external(size_dtype, topo_sizes.element_ptr(ei));
        }
        const index_t elem_num_faces = topo_shape.is_polyhedral() ?
            data_node.to_int64() : 1;

        std::set<index_t> elem_coord_indices;
        for(index_t fi = 0, foffset = eoffset;
            fi < elem_num_faces; fi++)
        {

            index_t subelem_index = 0;
            index_t subelem_offset = 0;
            index_t subelem_size = 0;
            if (topo_shape.is_polyhedral())
            {
                data_node.set_external(conn_dtype, topo_conn.element_ptr(foffset));
                subelem_index = data_node.to_int64();
                data_node.set_external(suboffset_dtype, topo_suboffsets.element_ptr(subelem_index));
                subelem_offset = data_node.to_int64();
                data_node.set_external(subsize_dtype, topo_subsizes.element_ptr(subelem_index));
                subelem_size = data_node.to_int64();
            }

            const index_t face_num_coords =
                topo_shape.is_polyhedral() ? subelem_size :
                topo_shape.is_polygonal() ? esize :
                topo_shape.indices;

            for(index_t ci = 0; ci < face_num_coords; ci++)
            {
                if (topo_shape.is_polyhedral())
                {
                    data_node.set_external(subconn_dtype, topo_subconn.element_ptr(subelem_offset + ci));
                }
                else
                {
                    data_node.set_external(conn_dtype, topo_conn.element_ptr(foffset + ci));
                }
                elem_coord_indices.insert(data_node.to_int64());
            }
            foffset += topo_shape.is_polyhedral() ? 1 : face_num_coords;
        }

        float64 ecentroid[3] = {0.0, 0.0, 0.0};
        for(std::set<index_t>::iterator elem_cindices_it = elem_coord_indices.begin();
            elem_cindices_it != elem_coord_indices.end(); ++elem_cindices_it)
        {
            index_t ci = *elem_cindices_it;
            for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
            {
                const Node &axis_data = coordset["values"][csys_axes[ai]];
                data_node.set_external(DataType(axis_data.dtype().id(), 1),
                    const_cast<void*>(axis_data.element_ptr(ci)));
                ecentroid[ai] += data_node.to_float64() / elem_coord_indices.size();
            }
        }

        int64 ei_value = static_cast<int64>(ei);
        Node ei_data(DataType::int64(1), &ei_value, true);
        data_node.set_external(int_dtype, dest["elements/connectivity"].element_ptr(ei));
        ei_data.to_data_type(int_dtype.id(), data_node);

        for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
        {
            data_node.set_external(float_dtype,
                cdest["values"][csys_axes[ai]].element_ptr(ei));
            Node center_data(DataType::float64(), &ecentroid[ai], true);
            center_data.to_data_type(float_dtype.id(), data_node);
        }
    }
}

//-----------------------------------------------------------------------------
// - end internal data function helpers -
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- end internal helper functions --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
bool
mesh::verify(const std::string &protocol,
             const Node &n,
             Node &info)
{
    bool res = false;
    info.reset();

    if(protocol == "coordset")
    {
        res = coordset::verify(n,info);
    }
    else if(protocol == "topology")
    {
        res = topology::verify(n,info);
    }
    else if(protocol == "matset")
    {
        res = matset::verify(n,info);
    }
    else if(protocol == "specset")
    {
        res = specset::verify(n,info);
    }
    else if(protocol == "field")
    {
        res = field::verify(n,info);
    }
    else if(protocol == "adjset")
    {
        res = adjset::verify(n,info);
    }
    else if(protocol == "nestset")
    {
        res = nestset::verify(n,info);
    }
    else if(protocol == "index")
    {
        res = index::verify(n,info);
    }
    else if(protocol == "coordset/index")
    {
        res = coordset::index::verify(n,info);
    }
    else if(protocol == "topology/index")
    {
        res = topology::index::verify(n,info);
    }
    else if(protocol == "matset/index")
    {
        res = matset::index::verify(n,info);
    }
    else if(protocol == "specset/index")
    {
        res = specset::index::verify(n,info);
    }
    else if(protocol == "field/index")
    {
        res = field::index::verify(n,info);
    }
    else if(protocol == "adjset/index")
    {
        res = adjset::index::verify(n,info);
    }
    else if(protocol == "nestset/index")
    {
        res = nestset::index::verify(n,info);
    }

    return res;
}


//-----------------------------------------------------------------------------
bool
mesh::verify_single_domain(const Node &n,
                           Node &info)
{
    const std::string protocol = "mesh";
    bool res = true;
    info.reset();

    if(!verify_object_field(protocol, n, info, "coordsets"))
    {
        res = false;
    }
    else
    {
        bool cset_res = true;
        NodeConstIterator itr = n["coordsets"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();

            cset_res &= coordset::verify(chld, info["coordsets"][chld_name]);
        }

        log::validation(info["coordsets"],cset_res);
        res &= cset_res;
    }

    if(!verify_object_field(protocol, n, info, "topologies"))
    {
        res = false;
    }
    else
    {
        bool topo_res = true;
        NodeConstIterator itr = n["topologies"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            Node &chld_info = info["topologies"][chld_name];

            topo_res &= topology::verify(chld, chld_info);
            topo_res &= verify_reference_field(protocol, n, info,
                chld, chld_info, "coordset", "coordsets");
        }

        log::validation(info["topologies"],topo_res);
        res &= topo_res;
    }

    // optional: "matsets", each child must conform to "mesh::matset"
    if(n.has_path("matsets"))
    {
        if(!verify_object_field(protocol, n, info, "matsets"))
        {
            res = false;
        }
        else
        {
            bool mset_res = true;
            NodeConstIterator itr = n["matsets"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["matsets"][chld_name];

                mset_res &= matset::verify(chld, chld_info);
                mset_res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "topology", "topologies");
            }

            log::validation(info["matsets"],mset_res);
            res &= mset_res;
        }
    }

    // optional: "specsets", each child must conform to "mesh::specset"
    if(n.has_path("specsets"))
    {
        if(!verify_object_field(protocol, n, info, "specsets"))
        {
            res = false;
        }
        else
        {
            bool sset_res = true;
            NodeConstIterator itr = n["specsets"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["specsets"][chld_name];

                sset_res &= specset::verify(chld, chld_info);
                sset_res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "matset", "matsets");
            }

            log::validation(info["specsets"],sset_res);
            res &= sset_res;
        }
    }

    // optional: "fields", each child must conform to "mesh::field"
    if(n.has_path("fields"))
    {
        if(!verify_object_field(protocol, n, info, "fields"))
        {
            res = false;
        }
        else
        {
            bool field_res = true;
            NodeConstIterator itr = n["fields"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["fields"][chld_name];

                field_res &= field::verify(chld, chld_info);
                if(chld.has_child("topology"))
                {
                    field_res &= verify_reference_field(protocol, n, info,
                        chld, chld_info, "topology", "topologies");
                }
                if(chld.has_child("matset"))
                {
                    field_res &= verify_reference_field(protocol, n, info,
                        chld, chld_info, "matset", "matsets");
                }
            }

            log::validation(info["fields"],field_res);
            res &= field_res;
        }
    }

    // optional: "adjsets", each child must conform to "mesh::adjset"
    if(n.has_path("adjsets"))
    {
        if(!verify_object_field(protocol, n, info, "adjsets"))
        {
            res = false;
        }
        else
        {
            bool aset_res = true;
            NodeConstIterator itr = n["adjsets"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["adjsets"][chld_name];

                aset_res &= adjset::verify(chld, chld_info);
                aset_res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "topology", "topologies");
            }

            log::validation(info["adjsets"],aset_res);
            res &= aset_res;
        }
    }

    // optional: "nestsets", each child must conform to "mesh::nestset"
    if(n.has_path("nestsets"))
    {
        if(!verify_object_field(protocol, n, info, "nestsets"))
        {
            res = false;
        }
        else
        {
            bool nset_res = true;
            NodeConstIterator itr = n["nestsets"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["nestsets"][chld_name];

                nset_res &= nestset::verify(chld, chld_info);
                nset_res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "topology", "topologies");
            }

            log::validation(info["nestets"],nset_res);
            res &= nset_res;
        }
    }


    // one last pass to make sure if a grid_function was specified by a topo,
    // it is valid
    if (n.has_child("topologies"))
    {
        bool topo_res = true;
        NodeConstIterator itr = n["topologies"].children();
        while (itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            Node &chld_info = info["topologies"][chld_name];

            if(chld.has_child("grid_function"))
            {
                topo_res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "grid_function", "fields");
            }
        }

        log::validation(info["topologies"],topo_res);
        res &= topo_res;
    }

    log::validation(info,res);

    return res;
}


//-------------------------------------------------------------------------
bool mesh::verify_multi_domain(const Node &n,
                               Node &info)
{
    const std::string protocol = "mesh";
    bool res = true;
    info.reset();

    if(!n.dtype().is_object() && !n.dtype().is_list())
    {
        log::error(info, protocol, "not an object or a list");
        res = false;
    }
    else
    {
        NodeConstIterator itr = n.children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            res &= mesh::verify_single_domain(chld, info[chld_name]);
        }

        log::info(info, protocol, "is a multi domain mesh");
    }

    log::validation(info,res);

    return res;
}


//-----------------------------------------------------------------------------
bool
mesh::verify(const Node &n,
             Node &info)
{
    bool res = true;
    info.reset();

    // if n has the child "coordsets", we assume it is a single domain
    // mesh
    if(n.has_child("coordsets"))
    {
        res = mesh::verify_single_domain(n, info);
    }
    else
    {
       res = mesh::verify_multi_domain(n, info);
    }
    return res;
}


//-------------------------------------------------------------------------
bool mesh::is_multi_domain(const conduit::Node &n)
{
    // this is a blueprint property, we can assume it will be called
    // only when mesh verify is true. Given that - the only check
    // we need to make is the minimal check to distinguish between
    // a single domain and a multi domain tree structure.
    // checking for a child named "coordsets" mirrors the
    // top level verify check

    return !n.has_child("coordsets");
}

//-------------------------------------------------------------------------
index_t
mesh::number_of_domains(const conduit::Node &n)
{
    // this is a blueprint property, we can assume it will be called
    // only when mesh verify is true. Given that - it is easy to
    // answer the number of domains

    if(!is_multi_domain(n))
    {
        return 1;
    }
    else
    {
        return n.number_of_children();
    }
}



//-------------------------------------------------------------------------
void mesh::to_multi_domain(const conduit::Node &n,
                           conduit::Node &dest)
{
    dest.reset();

    if(mesh::is_multi_domain(n))
    {
        dest.set_external(n);
    }
    else
    {
        conduit::Node &dest_dom = dest.append();
        dest_dom.set_external(n);
    }
}


//-----------------------------------------------------------------------------
void
mesh::generate_index(const Node &mesh,
                     const std::string &ref_path,
                     index_t number_of_domains,
                     Node &index_out)
{
    index_out.reset();

    index_out["state/number_of_domains"] = number_of_domains;


    if(mesh.has_child("state"))
    {
        // check if the input mesh has state/cycle state/time
        // if so, add those to the index
        if(mesh.has_path("state/cycle"))
        {
            index_out["state/cycle"].set(mesh["state/cycle"]);
        }

        if(mesh.has_path("state/time"))
        {
            index_out["state/time"].set(mesh["state/time"]);
        }
        // state may contain other important stuff, like
        // the domain_id, so we need a way to read it
        // from the index
        index_out["state/path"] = join_path(ref_path, "state");
    }

    NodeConstIterator itr = mesh["coordsets"].children();
    while(itr.has_next())
    {
        const Node &coordset = itr.next();
        std::string coordset_name = itr.name();
        Node &idx_coordset = index_out["coordsets"][coordset_name];

        std::string coordset_type =   coordset["type"].as_string();
        idx_coordset["type"] = coordset_type;
        if(coordset_type == "uniform")
        {
            // default to cartesian, but check if origin or spacing exist
            // b/c they may name axes from cyln or sph
            if(coordset.has_child("origin"))
            {
                NodeConstIterator origin_itr = coordset["origin"].children();
                while(origin_itr.has_next())
                {
                    origin_itr.next();
                    idx_coordset["coord_system/axes"][origin_itr.name()];
                }
            }
            else if(coordset.has_child("spacing"))
            {
                NodeConstIterator spacing_itr = coordset["spacing"].children();
                while(spacing_itr.has_next())
                {
                    spacing_itr.next();
                    std::string axis_name = spacing_itr.name();

                    // if spacing names start with "d", use substr
                    // to determine axis name

                    // otherwise use spacing name directly, to avoid empty
                    // path fetch if just 'x', etc are passed
                    if(axis_name[0] == 'd' && axis_name.size() > 1)
                    {
                        axis_name = axis_name.substr(1);
                    }
                    idx_coordset["coord_system/axes"][axis_name];
                }
            }
            else
            {
                // assume cartesian
                index_t num_comps = coordset["dims"].number_of_children();

                if(num_comps > 0)
                {
                    idx_coordset["coord_system/axes/x"];
                }

                if(num_comps > 1)
                {
                    idx_coordset["coord_system/axes/y"];
                }

                if(num_comps > 2)
                {
                    idx_coordset["coord_system/axes/z"];
                }
            }
        }
        else
        {
            // use child names as axes
            NodeConstIterator values_itr = coordset["values"].children();
            while(values_itr.has_next())
            {
                values_itr.next();
                idx_coordset["coord_system/axes"][values_itr.name()];
            }
        }

        idx_coordset["coord_system/type"] = identify_coords_coordsys(idx_coordset["coord_system/axes"]);

        std::string cs_ref_path = join_path(ref_path, "coordsets");
        cs_ref_path = join_path(cs_ref_path, coordset_name);
        idx_coordset["path"] = cs_ref_path;
    }

    itr = mesh["topologies"].children();
    while(itr.has_next())
    {
        const Node &topo = itr.next();
        std::string topo_name = itr.name();
        Node &idx_topo = index_out["topologies"][topo_name];
        idx_topo["type"] = topo["type"].as_string();
        idx_topo["coordset"] = topo["coordset"].as_string();

        std::string tp_ref_path = join_path(ref_path,"topologies");
        tp_ref_path = join_path(tp_ref_path,topo_name);
        idx_topo["path"] = tp_ref_path;

        // a topology may also specify a grid_function
        if(topo.has_child("grid_function"))
        {
            idx_topo["grid_function"] = topo["grid_function"].as_string();
        }
    }

    if(mesh.has_child("matsets"))
    {
        itr = mesh["matsets"].children();
        while(itr.has_next())
        {
            const Node &matset = itr.next();
            const std::string matset_name = itr.name();
            Node &idx_matset = index_out["matsets"][matset_name];

            idx_matset["topology"] = matset["topology"].as_string();

            // support different flavors of valid matset protos
            //
            // if we have material_map (node with names to ids)
            // use it in the index
            if(matset.has_child("material_map"))
            {
                idx_matset["material_map"] = matset["material_map"];
            }
            else if(matset.has_child("materials"))
            {
                // NOTE: I believe path is deprecated ... 
                NodeConstIterator mats_itr = matset["materials"].children();
                while(mats_itr.has_next())
                {
                    mats_itr.next();
                    idx_matset["materials"][mats_itr.name()];
                }
            }
            else if(matset.has_child("volume_fractions"))
            {
                // we don't have material_map (node with names to ids)
                // so mapping is implied from node order, construct
                // an actual map that follows the implicit order
                NodeConstIterator mats_itr = matset["volume_fractions"].children();
                index_t mat_id = 0;
                while(mats_itr.has_next())
                {
                    mats_itr.next();
                    idx_matset["material_map"][mats_itr.name()] = mat_id;
                    mat_id ++;
                }
            }
            else // surprise!
            {
                CONDUIT_ERROR("blueprint::mesh::generate_index: "
                              "Invalid matset flavor."
                              "Input node does not conform to mesh blueprint.");
            }

            std::string ms_ref_path = join_path(ref_path, "matsets");
            ms_ref_path = join_path(ms_ref_path, matset_name);
            idx_matset["path"] = ms_ref_path;
        }
    }

    if(mesh.has_child("specsets"))
    {
        itr = mesh["specsets"].children();
        while(itr.has_next())
        {
            const Node &specset = itr.next();
            const std::string specset_name = itr.name();
            Node &idx_specset = index_out["specsets"][specset_name];

            idx_specset["matset"] = specset["matset"].as_string();
            // TODO(JRC): Is the 'materials' entry necessary given that it will
            // always match the 'materials' entry in the 'matset' list?
            NodeConstIterator specs_itr = specset["matset_values"].child(0).children();
            while(specs_itr.has_next())
            {
                specs_itr.next();
                idx_specset["species"][specs_itr.name()];
            }

            std::string ms_ref_path = join_path(ref_path, "specsets");
            ms_ref_path = join_path(ms_ref_path, specset_name);
            idx_specset["path"] = ms_ref_path;
        }
    }

    if(mesh.has_child("fields"))
    {
        itr = mesh["fields"].children();
        while(itr.has_next())
        {
            const Node &fld = itr.next();
            std::string fld_name = itr.name();
            Node &idx_fld = index_out["fields"][fld_name];

            index_t ncomps = 1;
            if(fld.has_child("values"))
            {
                if(fld["values"].dtype().is_object())
                {
                    ncomps = fld["values"].number_of_children();
                }
            }
            else
            {
                if(fld["matset_values"].child(0).dtype().is_object())
                {
                    ncomps = fld["matset_values"].child(0).number_of_children();
                }
            }
            idx_fld["number_of_components"] = ncomps;

            if(fld.has_child("topology"))
            {
                idx_fld["topology"] = fld["topology"].as_string();
            }
            if(fld.has_child("matset"))
            {
                idx_fld["matset"] = fld["matset"].as_string();
            }

            if(fld.has_child("association"))
            {
                idx_fld["association"] = fld["association"];
            }
            else
            {
                idx_fld["basis"] = fld["basis"];
            }

            std::string fld_ref_path = join_path(ref_path,"fields");
            fld_ref_path = join_path(fld_ref_path, fld_name);
            idx_fld["path"] = fld_ref_path;
        }
    }

    if(mesh.has_child("adjsets"))
    {
        itr = mesh["adjsets"].children();
        while(itr.has_next())
        {
            const Node &adjset = itr.next();
            const std::string adj_name = itr.name();
            Node &idx_adjset = index_out["adjsets"][adj_name];

            // TODO(JRC): Determine whether or not any information from the
            // "neighbors" and "values" sections need to be included in the index.
            idx_adjset["association"] = adjset["association"].as_string();
            idx_adjset["topology"] = adjset["topology"].as_string();

            std::string adj_ref_path = join_path(ref_path,"adjsets");
            adj_ref_path = join_path(adj_ref_path, adj_name);
            idx_adjset["path"] = adj_ref_path;
        }
    }

    if(mesh.has_child("nestsets"))
    {
        itr = mesh["nestsets"].children();
        while(itr.has_next())
        {
            const Node &nestset = itr.next();
            const std::string nest_name = itr.name();
            Node &idx_nestset = index_out["nestsets"][nest_name];

            // TODO(JRC): Determine whether or not any information from the
            // "domain_id" or "ratio" sections need to be included in the index.
            idx_nestset["association"] = nestset["association"].as_string();
            idx_nestset["topology"] = nestset["topology"].as_string();

            std::string adj_ref_path = join_path(ref_path,"nestsets");
            adj_ref_path = join_path(adj_ref_path, nest_name);
            idx_nestset["path"] = adj_ref_path;
        }
    }
}


//-----------------------------------------------------------------------------
// blueprint::mesh::logical_dims protocol interface
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool
mesh::logical_dims::verify(const Node &dims,
                           Node &info)
{
    const std::string protocol = "mesh::logical_dims";
    bool res = true;
    info.reset();

    res &= verify_integer_field(protocol, dims, info, "i");
    if(dims.has_child("j"))
    {
        res &= verify_integer_field(protocol, dims, info, "j");
    }
    if(dims.has_child("k"))
    {
        res &= verify_integer_field(protocol, dims, info, "k");
    }

    log::validation(info, res);

    return res;
}


//-----------------------------------------------------------------------------
// blueprint::mesh::association protocol interface
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool
mesh::association::verify(const Node &assoc,
                          Node &info)
{
    const std::string protocol = "mesh::association";
    bool res = true;
    info.reset();

    res &= verify_enum_field(protocol, assoc, info, "", mesh::associations);

    log::validation(info, res);

    return res;
}


//-----------------------------------------------------------------------------
// blueprint::mesh::coordset protocol interface
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// blueprint::mesh::coordset::verify protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::coordset::uniform::origin::verify(const Node &origin,
                                        Node &info)
{
    const std::string protocol = "mesh::coordset::uniform::origin";
    bool res = true;
    info.reset();

    for(size_t i = 0; i < mesh::coordinate_axes.size(); i++)
    {
        const std::string &coord_axis = mesh::coordinate_axes[i];
        if(origin.has_child(coord_axis))
        {
            res &= verify_number_field(protocol, origin, info, coord_axis);
        }
    }

    log::validation(info, res);

    return res;
}



//-----------------------------------------------------------------------------
bool
mesh::coordset::uniform::spacing::verify(const Node &spacing,
                                         Node &info)
{
    const std::string protocol = "mesh::coordset::uniform::spacing";
    bool res = true;
    info.reset();

    for(size_t i = 0; i < mesh::coordinate_axes.size(); i++)
    {
        const std::string &coord_axis = mesh::coordinate_axes[i];
        const std::string coord_axis_spacing = "d" + coord_axis;
        if(spacing.has_child(coord_axis_spacing))
        {
            res &= verify_number_field(protocol, spacing, info, coord_axis_spacing);
        }
    }

    log::validation(info,res);

    return res;
}

//-----------------------------------------------------------------------------
bool
mesh::coordset::uniform::verify(const Node &coordset,
                                Node &info)
{
    const std::string protocol = "mesh::coordset::uniform";
    bool res = true;
    info.reset();

    res &= verify_enum_field(protocol, coordset, info, "type",
        std::vector<std::string>(1, "uniform"));

    res &= verify_object_field(protocol, coordset, info, "dims") &&
           mesh::logical_dims::verify(coordset["dims"], info["dims"]);

    if(coordset.has_child("origin"))
    {
        log::optional(info, protocol, "has origin");
        res &= mesh::coordset::uniform::origin::verify(coordset["origin"],
                                                       info["origin"]);
    }

    if(coordset.has_child("spacing"))
    {
        log::optional(info,protocol, "has spacing");
        res &= mesh::coordset::uniform::spacing::verify(coordset["spacing"],
                                                        info["spacing"]);
    }

    log::validation(info,res);

    return res;
}

//-----------------------------------------------------------------------------
bool
mesh::coordset::rectilinear::verify(const Node &coordset,
                                    Node &info)
{
    const std::string protocol = "mesh::coordset::rectilinear";
    bool res = true;
    info.reset();

    res &= verify_enum_field(protocol, coordset, info, "type",
        std::vector<std::string>(1, "rectilinear"));

    if(!verify_object_field(protocol, coordset, info, "values", true))
    {
        res = false;
    }
    else
    {
        NodeConstIterator itr = coordset["values"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            if(!chld.dtype().is_number())
            {
                log::error(info, protocol, "value child " + log::quote(chld_name) +
                                           " is not a number array");
                res = false;
            }
        }
    }

    log::validation(info,res);

    return res;
}


//-----------------------------------------------------------------------------
bool
mesh::coordset::_explicit::verify(const Node &coordset,
                                 Node &info)
{
    const std::string protocol = "mesh::coordset::explicit";
    bool res = true;
    info.reset();

    res &= verify_enum_field(protocol, coordset, info, "type",
        std::vector<std::string>(1, "explicit"));

    res &= verify_mcarray_field(protocol, coordset, info, "values");

    log::validation(info,res);

    return res;
}


//-----------------------------------------------------------------------------
bool
mesh::coordset::verify(const Node &coordset,
                       Node &info)
{
    const std::string protocol = "mesh::coordset";
    bool res = true;
    info.reset();

    res &= verify_field_exists(protocol, coordset, info, "type") &&
           mesh::coordset::type::verify(coordset["type"], info["type"]);

    if(res)
    {
        const std::string type_name = coordset["type"].as_string();

        if(type_name == "uniform")
        {
            res = mesh::coordset::uniform::verify(coordset,info);
        }
        else if(type_name == "rectilinear")
        {
            res = mesh::coordset::rectilinear::verify(coordset,info);
        }
        else if(type_name == "explicit")
        {
            res = mesh::coordset::_explicit::verify(coordset,info);
        }
    }

    log::validation(info,res);

    return res;
}


//-----------------------------------------------------------------------------
index_t
mesh::coordset::dims(const Node &coordset)
{
    std::vector<std::string> csys_axes = identify_coordset_axes(coordset);
    return (index_t)csys_axes.size();
}


//-------------------------------------------------------------------------
void
mesh::coordset::uniform::to_rectilinear(const conduit::Node &coordset,
                                        conduit::Node &dest)
{
    convert_coordset_to_rectilinear("uniform", coordset, dest);
}


//-------------------------------------------------------------------------
void
mesh::coordset::uniform::to_explicit(const conduit::Node &coordset,
                                     conduit::Node &dest)
{
    convert_coordset_to_explicit("uniform", coordset, dest);
}


//-------------------------------------------------------------------------
void
mesh::coordset::rectilinear::to_explicit(const conduit::Node &coordset,
                                         conduit::Node &dest)
{
    convert_coordset_to_explicit("rectilinear", coordset, dest);
}


//-----------------------------------------------------------------------------
// blueprint::mesh::coordset::type protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::coordset::type::verify(const Node &type,
                             Node &info)
{
    const std::string protocol = "mesh::coordset::type";
    bool res = true;
    info.reset();

    res &= verify_enum_field(protocol, type, info, "", mesh::coord_types);

    log::validation(info,res);

    return res;
}


//-----------------------------------------------------------------------------
// blueprint::mesh::coordset::coord_system protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::coordset::coord_system::verify(const Node &coord_sys,
                                     Node &info)
{
    const std::string protocol = "mesh::coordset::coord_system";
    bool res = true;
    info.reset();

    std::string coord_sys_str = "unknown";
    if(!verify_enum_field(protocol, coord_sys, info, "type", mesh::coord_systems))
    {
        res = false;
    }
    else
    {
        coord_sys_str = coord_sys["type"].as_string();
    }

    if(!verify_object_field(protocol, coord_sys, info, "axes"))
    {
        res = false;
    }
    else if(coord_sys_str != "unknown")
    {
        NodeConstIterator itr = coord_sys["axes"].children();
        while(itr.has_next())
        {
            itr.next();
            const std::string axis_name = itr.name();

            bool axis_name_ok = true;
            if(coord_sys_str == "cartesian")
            {
                axis_name_ok = axis_name == "x" || axis_name == "y" ||
                               axis_name == "z";
            }
            else if(coord_sys_str == "cylindrical")
            {
                axis_name_ok = axis_name == "r" || axis_name == "z";
            }
            else if(coord_sys_str == "spherical")
            {
                axis_name_ok = axis_name == "r" || axis_name == "theta" ||
                               axis_name == "phi";
            }

            if(!axis_name_ok)
            {
                log::error(info, protocol, "unsupported " + coord_sys_str +
                                           " axis name: " + axis_name);
                res = false;
            }
        }
    }

    log::validation(info,res);

    return res;
}


//-----------------------------------------------------------------------------
// blueprint::mesh::coordset::index protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::coordset::index::verify(const Node &coordset_idx,
                              Node &info)
{
    const std::string protocol = "mesh::coordset::index";
    bool res = true;
    info.reset();

    res &= verify_field_exists(protocol, coordset_idx, info, "type") &&
           mesh::coordset::type::verify(coordset_idx["type"], info["type"]);
    res &= verify_string_field(protocol, coordset_idx, info, "path");
    res &= verify_object_field(protocol, coordset_idx, info, "coord_system") &&
           coordset::coord_system::verify(coordset_idx["coord_system"], info["coord_system"]);

    log::validation(info,res);

    return res;
}


//-----------------------------------------------------------------------------
// blueprint::mesh::topology protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::verify(const Node &topo,
                       Node &info)
{
    const std::string protocol = "mesh::topology";
    bool res = true;
    info.reset();

    if(!(verify_field_exists(protocol, topo, info, "type") &&
         mesh::topology::type::verify(topo["type"], info["type"])))
    {
        res = false;
    }
    else
    {
        const std::string topo_type = topo["type"].as_string();

        if(topo_type == "points")
        {
            res &= mesh::topology::points::verify(topo,info);
        }
        else if(topo_type == "uniform")
        {
            res &= mesh::topology::uniform::verify(topo,info);
        }
        else if(topo_type == "rectilinear")
        {
            res &= mesh::topology::rectilinear::verify(topo,info);
        }
        else if(topo_type == "structured")
        {
            res &= mesh::topology::structured::verify(topo,info);
        }
        else if(topo_type == "unstructured")
        {
            res &= mesh::topology::unstructured::verify(topo,info);
        }
    }

    if(topo.has_child("grid_function"))
    {
        log::optional(info, protocol, "includes grid_function");
        res &= verify_string_field(protocol, topo, info, "grid_function");
    }

    log::validation(info,res);

    return res;

}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::points protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::points::verify(const Node & topo,
                               Node &info)
{
    const std::string protocol = "mesh::topology::points";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, topo, info, "coordset");

    res &= verify_enum_field(protocol, topo, info, "type",
        std::vector<std::string>(1, "points"));

    // if needed in the future, can be used to verify optional info for
    // implicit 'points' topology

    log::validation(info,res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::uniform protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::uniform::verify(const Node & topo,
                                Node &info)
{
    const std::string protocol = "mesh::topology::uniform";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, topo, info, "coordset");

    res &= verify_enum_field(protocol, topo, info, "type",
        std::vector<std::string>(1, "uniform"));

    // future: will be used to verify optional info from "elements"
    // child of a uniform topology

    log::validation(info,res);

    return res;
}


//-------------------------------------------------------------------------
void
mesh::topology::uniform::to_rectilinear(const conduit::Node &topo,
                                        conduit::Node &dest,
                                        conduit::Node &cdest)
{
    convert_topology_to_rectilinear("uniform", topo, dest, cdest);
}


//-------------------------------------------------------------------------
void
mesh::topology::uniform::to_structured(const conduit::Node &topo,
                                       conduit::Node &dest,
                                       conduit::Node &cdest)
{
    convert_topology_to_structured("uniform", topo, dest, cdest);
}


//-------------------------------------------------------------------------
void
mesh::topology::uniform::to_unstructured(const conduit::Node &topo,
                                         conduit::Node &dest,
                                         conduit::Node &cdest)
{
    convert_topology_to_unstructured("uniform", topo, dest, cdest);
}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::rectilinear protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::rectilinear::verify(const Node &topo,
                                    Node &info)
{
    const std::string protocol = "mesh::topology::rectilinear";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, topo, info, "coordset");

    res &= verify_enum_field(protocol, topo, info, "type",
        std::vector<std::string>(1, "rectilinear"));

    // future: will be used to verify optional info from "elements"
    // child of a rectilinear topology

    log::validation(info,res);

    return res;
}


//-------------------------------------------------------------------------
void
mesh::topology::rectilinear::to_structured(const conduit::Node &topo,
                                           conduit::Node &dest,
                                           conduit::Node &cdest)
{
    convert_topology_to_structured("rectilinear", topo, dest, cdest);
}


//-------------------------------------------------------------------------
void
mesh::topology::rectilinear::to_unstructured(const conduit::Node &topo,
                                             conduit::Node &dest,
                                             conduit::Node &cdest)
{
    convert_topology_to_unstructured("rectilinear", topo, dest, cdest);
}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::structured protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::structured::verify(const Node &topo,
                                   Node &info)
{
    const std::string protocol = "mesh::topology::structured";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, topo, info, "coordset");

    res &= verify_enum_field(protocol, topo, info, "type",
        std::vector<std::string>(1, "structured"));

    if(!verify_object_field(protocol, topo, info, "elements"))
    {
        res = false;
    }
    else
    {
        const Node &topo_elements = topo["elements"];
        Node &info_elements = info["elements"];

        bool elements_res =
            verify_object_field(protocol, topo_elements, info_elements, "dims") &&
            mesh::logical_dims::verify(topo_elements["dims"], info_elements["dims"]);

        log::validation(info_elements,elements_res);
        res &= elements_res;
    }

    // FIXME: Add some verification code here for the optional origin in the
    // structured topology.

    log::validation(info,res);

    return res;
}


//-------------------------------------------------------------------------
void
mesh::topology::structured::to_unstructured(const conduit::Node &topo,
                                            conduit::Node &dest,
                                            conduit::Node &cdest)
{
    convert_topology_to_unstructured("structured", topo, dest, cdest);
}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::unstructured protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::unstructured::verify(const Node &topo,
                                     Node &info)
{
    const std::string protocol = "mesh::topology::unstructured";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, topo, info, "coordset");

    res &= verify_enum_field(protocol, topo, info, "type",
        std::vector<std::string>(1, "unstructured"));

    if(!verify_object_field(protocol, topo, info, "elements"))
    {
        res = false;
    }
    else
    {
        const Node &topo_elems = topo["elements"];
        Node &info_elems = info["elements"];

        bool elems_res = true;
        bool subelems_res = true;

        // single shape case
        if(topo_elems.has_child("shape"))
        {
            elems_res &= verify_field_exists(protocol, topo_elems, info_elems, "shape") &&
                   mesh::topology::shape::verify(topo_elems["shape"], info_elems["shape"]);
            elems_res &= verify_integer_field(protocol, topo_elems, info_elems, "connectivity");

            // Verify if node is polygonal or polyhedral
            elems_res &= verify_poly_node (false, "", topo_elems, info_elems, topo, info, elems_res);
        }
        // shape stream case
        else if(topo_elems.has_child("element_types"))
        {
            // TODO
        }
        // mixed shape case
        else if(topo_elems.number_of_children() != 0)
        {
            bool has_names = topo_elems.dtype().is_object();

            NodeConstIterator itr = topo_elems.children();
            while(itr.has_next())
            {
                const Node &chld  = itr.next();
                std::string name = itr.name();
                Node &chld_info = has_names ? info["elements"][name] :
                    info["elements"].append();

                bool chld_res = true;
                chld_res &= verify_field_exists(protocol, chld, chld_info, "shape") &&
                       mesh::topology::shape::verify(chld["shape"], chld_info["shape"]);
                chld_res &= verify_integer_field(protocol, chld, chld_info, "connectivity");

                // Verify if child is polygonal or polyhedral
                chld_res &= verify_poly_node (true, name, chld, chld_info, topo, info, elems_res);

                log::validation(chld_info,chld_res);
                elems_res &= chld_res;
            }
        }
        else
        {
            log::error(info,protocol,"invalid child 'elements'");
            res = false;
        }

        log::validation(info_elems,elems_res);
        res &= elems_res;
        res &= subelems_res;
    }

    log::validation(info,res);

    return res;
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::to_polygonal(const Node &topo,
                                           Node &dest)
{
    dest.reset();

    const ShapeCascade topo_cascade(topo);
    const ShapeType topo_shape(topo_cascade.get_shape());
    const DataType int_dtype = find_widest_dtype(topo, blueprint::mesh::default_int_dtypes);

    if(topo_shape.is_poly())
    {
        dest.set(topo);
    }
    else // if(!topo_shape.is_poly())
    {
        const Node &topo_conn_const = topo["elements/connectivity"];
        Node topo_conn; topo_conn.set_external(topo_conn_const);
        const DataType topo_dtype(topo_conn.dtype().id(), 1);
        const index_t topo_indices = topo_conn.dtype().number_of_elements();
        const index_t topo_elems = topo_indices / topo_shape.indices;
        const bool is_topo_3d = topo_shape.dim == 3;

        Node topo_templ;
        topo_templ.set_external(topo);
        topo_templ.remove("elements");
        dest.set(topo_templ);
        dest["elements/shape"].set(is_topo_3d ? "polyhedral" : "polygonal");

        Node temp;
        if (!is_topo_3d) // polygonal
        {
            // NOTE(JRC): The derived polygonal topology simply inherits the
            // original implicit connectivity and adds sizes/offsets, which
            // means that it inherits the orientation/winding of the source as well.
            temp.set_external(topo_conn);
            temp.to_data_type(int_dtype.id(), dest["elements/connectivity"]);

            std::vector<int64> poly_size_data(topo_elems, topo_shape.indices);
            temp.set_external(poly_size_data);
            temp.to_data_type(int_dtype.id(), dest["elements/sizes"]);

            generate_offsets(dest, dest["elements/offsets"]);
        }
        else // if(is_topo_3d) // polyhedral
        {
            // NOTE(JRC): Polyhedral topologies are a bit more complicated
            // because the derivation comes from the embedding. The embedding
            // is statically RHR positive, but this can be turned negative by
            // an initially RHR negative element.
            const ShapeType embed_shape = topo_cascade.get_shape(topo_shape.dim - 1);

            std::vector<int64> polyhedral_conn_data(topo_elems * topo_shape.embed_count);
            std::vector<int64> polygonal_conn_data;
            std::vector<int64> face_indices(embed_shape.indices);

            // Generate each polyhedral element by generating its constituent
            // polygonal faces. Also, make sure that faces connecting the same
            // set of vertices aren't duplicated; reuse the ID generated by the
            // first polyhedral element to create the polygonal face.
            for (index_t ei = 0; ei < topo_elems; ei++)
            {
                index_t data_off = topo_shape.indices * ei;
                index_t polyhedral_off = topo_shape.embed_count * ei;

                for (index_t fi = 0; fi < topo_shape.embed_count; fi++)
                {
                    for (index_t ii = 0; ii < embed_shape.indices; ii++)
                    {
                        index_t inner_data_off = data_off +
                          topo_shape.embedding[fi * embed_shape.indices + ii];
                        temp.set_external(topo_dtype,
                            topo_conn.element_ptr(inner_data_off));
                        face_indices[ii] = temp.to_int64();
                    }

                    bool face_exists = false;
                    index_t face_index = polygonal_conn_data.size() / embed_shape.indices;
                    for (index_t poly_i = 0; poly_i < face_index; poly_i++)
                    {
                        index_t face_off = poly_i * embed_shape.indices;
                        face_exists |= std::is_permutation(polygonal_conn_data.begin() + face_off,
                                                           polygonal_conn_data.begin() + face_off + embed_shape.indices,
                                                           face_indices.begin());
                        face_index = face_exists ? poly_i : face_index;
                    }

                    polyhedral_conn_data[polyhedral_off + fi] = face_index;
                    if (!face_exists)
                    {
                        polygonal_conn_data.insert(polygonal_conn_data.end(),
                            face_indices.begin(), face_indices.end());
                    }
                }
            }

            temp.set_external(polyhedral_conn_data);
            temp.to_data_type(int_dtype.id(), dest["elements/connectivity"]);

            std::vector<int64> polyhedral_size_data(topo_elems, topo_shape.embed_count);
            temp.set_external(polyhedral_size_data);
            temp.to_data_type(int_dtype.id(), dest["elements/sizes"]);

            temp.set_external(polygonal_conn_data);
            temp.to_data_type(int_dtype.id(), dest["subelements/connectivity"]);

            std::vector<int64> polygonal_size_data(polygonal_conn_data.size() / embed_shape.indices,
                                                   embed_shape.indices);
            temp.set_external(polygonal_size_data);
            temp.to_data_type(int_dtype.id(), dest["subelements/sizes"]);

            dest["subelements/shape"].set("polygonal");

            // BHAN - For polyhedral, writes offsets for
            // "elements/offsets" and "subelements/offsets"
            generate_offsets(dest, dest["elements/offsets"]);
        }
    }
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_points(const Node &topo,
                                              Node &dest,
                                              Node &s2dmap,
                                              Node &d2smap)
{
    // TODO(JRC): Revise this function so that it works on every base topology
    // type and then move it to "mesh::topology::{uniform|...}::generate_points".
    Node coordset;

    find_reference_node(topo, "coordset", coordset);

    TopologyMetadata topo_data(topo, coordset);
    dest.reset();
    dest.set(topo_data.dim_topos[0]);

    const index_t src_dim = topo_data.topo_cascade.dim, dst_dim = 0;
    topo_data.get_dim_map(TopologyMetadata::GLOBAL, src_dim, dst_dim, s2dmap);
    topo_data.get_dim_map(TopologyMetadata::GLOBAL, dst_dim, src_dim, d2smap);
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_lines(const Node &topo,
                                             Node &dest,
                                             Node &s2dmap,
                                             Node &d2smap)
{
    // TODO(JRC): Revise this function so that it works on every base topology
    // type and then move it to "mesh::topology::{uniform|...}::generate_lines".
    Node coordset;
    find_reference_node(topo, "coordset", coordset);

    TopologyMetadata topo_data(topo, coordset);
    dest.reset();
    dest.set(topo_data.dim_topos[1]);

    const index_t src_dim = topo_data.topo_cascade.dim, dst_dim = 1;
    topo_data.get_dim_map(TopologyMetadata::GLOBAL, src_dim, dst_dim, s2dmap);
    topo_data.get_dim_map(TopologyMetadata::GLOBAL, dst_dim, src_dim, d2smap);
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_faces(const Node &topo,
                                             Node &dest,
                                             Node &s2dmap,
                                             Node &d2smap)
{
    // TODO(JRC): Revise this function so that it works on every base topology
    // type and then move it to "mesh::topology::{uniform|...}::generate_faces".
    Node coordset;
    find_reference_node(topo, "coordset", coordset);

    TopologyMetadata topo_data(topo, coordset);
    dest.reset();
    dest.set(topo_data.dim_topos[2]);

    const index_t src_dim = topo_data.topo_cascade.dim, dst_dim = 2;
    topo_data.get_dim_map(TopologyMetadata::GLOBAL, src_dim, dst_dim, s2dmap);
    topo_data.get_dim_map(TopologyMetadata::GLOBAL, dst_dim, src_dim, d2smap);
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_centroids(const Node &topo,
                                                 Node &dest,
                                                 Node &cdest,
                                                 Node &s2dmap,
                                                 Node &d2smap)
{
    // TODO(JRC): Revise this function so that it works on every base topology
    // type and then move it to "mesh::topology::{uniform|...}::generate_centroids".
    Node coordset;
    find_reference_node(topo, "coordset", coordset);

    calculate_unstructured_centroids(topo, coordset, dest, cdest);

    Node map_node;
    std::vector<index_t> map_vec;
    for(index_t ei = 0; ei < get_topology_length("unstructured", topo); ei++)
    {
        map_vec.push_back(1);
        map_vec.push_back(ei);
    }
    map_node.set(map_vec);

    DataType int_dtype = find_widest_dtype(link_nodes(topo, coordset),
        blueprint::mesh::default_int_dtypes);
    s2dmap.reset();
    d2smap.reset();
    map_node.to_data_type(int_dtype.id(), s2dmap);
    map_node.to_data_type(int_dtype.id(), d2smap);
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_sides(const Node &topo,
                                             Node &dest,
                                             Node &cdest,
                                             Node &s2dmap,
                                             Node &d2smap)
{
    // Retrieve Relevent Coordinate/Topology Metadata //

    Node coordset;
    find_reference_node(topo, "coordset", coordset);
    const std::vector<std::string> csys_axes = identify_coordset_axes(coordset);

    const ShapeCascade topo_cascade(topo);
    const ShapeType topo_shape = topo_cascade.get_shape();
    const ShapeType line_shape = topo_cascade.get_shape(1);
    const ShapeType side_shape(topo_shape.dim == 3 ? "tet" : "tri");
    if(topo_shape.dim < 2)
    {
        CONDUIT_ERROR("Failed to generate side mesh for input; " <<
            "input tology must be topologically 2D or 3D.");
    }

    // Extract Derived Coordinate/Topology Data //

    const TopologyMetadata topo_data(topo, coordset);
    const DataType &int_dtype = topo_data.int_dtype;
    const DataType &float_dtype = topo_data.float_dtype;

    std::vector<conduit::Node> dim_cent_topos(topo_shape.dim + 1);
    std::vector<conduit::Node> dim_cent_coords(topo_shape.dim + 1);

    for(index_t di = 0; di <= topo_shape.dim; di++)
    {
        // NOTE: No centroids are generate for the lines of the geometry
        // because they aren't included in the final sides topology.
        if(di == line_shape.dim) { continue; }

        calculate_unstructured_centroids(
            topo_data.dim_topos[di], coordset,
            dim_cent_topos[di], dim_cent_coords[di]);
    }

    // Allocate Data Templates for Outputs //

    const index_t topo_num_elems = topo_data.get_length(topo_shape.dim);
    const index_t sides_num_coords =
        topo_data.get_length() - topo_data.get_length(line_shape.dim);
    const index_t sides_num_elems =
        topo_data.get_embed_length(topo_shape.dim, line_shape.dim);
    const index_t sides_elem_degree = (topo_shape.dim - line_shape.dim) + 2;

    dest.reset();
    dest["type"].set("unstructured");
    dest["coordset"].set(cdest.name());
    dest["elements/shape"].set(side_shape.type);
    dest["elements/connectivity"].set(DataType(int_dtype.id(),
        side_shape.indices * sides_num_elems));

    cdest.reset();
    cdest["type"].set("explicit");
    for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
    {
        cdest["values"][csys_axes[ai]].set(DataType(float_dtype.id(),
            sides_num_coords));
    }

    // Populate Data Arrays w/ Calculated Coordinates //

    std::vector<index_t> dim_coord_offsets(topo_shape.dim + 1);
    for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
    {
        Node dst_data;
        Node &dst_axis = cdest["values"][csys_axes[ai]];

        for(index_t di = 0, doffset = 0; di <= topo_shape.dim; di++)
        {
            dim_coord_offsets[di] = doffset;

            // NOTE: The centroid ordering for the positions is different
            // from the base ordering, which messes up all subsequent indexing.
            // We must use the coordinate set associated with the base topology.
            const Node &cset = (di != 0) ? dim_cent_coords[di] : coordset;
            if(!cset.dtype().is_empty())
            {
                const Node &cset_axis = cset["values"][csys_axes[ai]];
                index_t cset_length = cset_axis.dtype().number_of_elements();

                dst_data.set_external(DataType(float_dtype.id(), cset_length),
                    dst_axis.element_ptr(doffset));
                cset_axis.to_data_type(float_dtype.id(), dst_data);
                doffset += cset_length;
            }
        }
    }

    // Compute New Elements/Fields for Side Topology //

    int64 elem_index = 0, side_index = 0;
    int64 s2d_val_index = 0, d2s_val_index = 0;
    int64 s2d_elem_index = 0, d2s_elem_index = 0;

    std::vector<int64> side_data_raw(sides_elem_degree);

    Node misc_data;
    Node raw_data(DataType::int64(1));
    Node elem_index_data(DataType::int64(1), &elem_index, true);
    Node side_index_data(DataType::int64(1), &side_index, true);
    Node side_data(DataType::int64(sides_elem_degree), &side_data_raw[0], true);

    s2dmap.reset();
    s2dmap["values"].set(DataType(int_dtype.id(), sides_num_elems));
    s2dmap["sizes"].set(DataType(int_dtype.id(), topo_num_elems));
    s2dmap["offsets"].set(DataType(int_dtype.id(), topo_num_elems));

    d2smap.reset();
    d2smap["values"].set(DataType(int_dtype.id(), sides_num_elems));
    d2smap["sizes"].set(DataType(int_dtype.id(), sides_num_elems));
    d2smap["offsets"].set(DataType(int_dtype.id(), sides_num_elems));

    Node &dest_conn = dest["elements/connectivity"];
    for(; elem_index < (int64)topo_num_elems; elem_index++)
    {
        std::deque< index_t > elem_embed_stack(1, elem_index);
        std::deque< index_t > elem_edim_stack(1, topo_shape.dim);
        std::deque< std::vector<index_t> > elem_eparent_stack(1);

        int64 s2d_start_index = s2d_val_index;

        while(!elem_embed_stack.empty())
        {
            index_t embed_index = elem_embed_stack.front();
            elem_embed_stack.pop_front();
            index_t embed_dim = elem_edim_stack.front();
            elem_edim_stack.pop_front();
            std::vector<index_t> embed_parents = elem_eparent_stack.front();
            elem_eparent_stack.pop_front();

            // NOTE(JRC): We iterate using local index values so that we
            // get the correct orientations for per-element lines.
            const std::vector<index_t> &embed_ids = topo_data.get_entity_assocs(
                TopologyMetadata::LOCAL, embed_index, embed_dim, embed_dim - 1);
            if(embed_dim > line_shape.dim)
            {
                embed_parents.push_back(embed_index);
                for(index_t ei = 0; ei < (index_t)embed_ids.size(); ei++)
                {
                    elem_embed_stack.push_back(embed_ids[ei]);
                    elem_edim_stack.push_back(embed_dim - 1);
                    elem_eparent_stack.push_back(embed_parents);
                }
            }
            else // if(embed_dim == line_shape.dim)
            {
                // NOTE(JRC): Side ordering retains original element orientation
                // by creating elements as follows:
                // - 2D: Face-Line Start => Face-Line End => Face Center
                // - 3D: Cell-Face-Line Start => Cell-Face-Line End => Cell-Face Center => Cell Center
                for(index_t ei = 0; ei < (index_t)embed_ids.size(); ei++)
                {
                    index_t point_id = topo_data.dim_le2ge_maps[embed_dim - 1][embed_ids[ei]];
                    side_data_raw[ei] = point_id;
                }
                for(index_t pi = 0; pi < (index_t)embed_parents.size(); pi++)
                {
                    index_t parent_index = embed_parents[embed_parents.size() - pi - 1];
                    index_t parent_dim = embed_dim + pi + 1;
                    index_t parent_id = topo_data.dim_le2ge_maps[parent_dim][parent_index];
                    side_data_raw[2 + pi] = dim_coord_offsets[parent_dim] + parent_id;
                }

                misc_data.set_external(DataType(int_dtype.id(), sides_elem_degree),
                    dest_conn.element_ptr(sides_elem_degree * side_index));
                side_data.to_data_type(int_dtype.id(), misc_data);

                misc_data.set_external(DataType(int_dtype.id(), 1),
                    s2dmap["values"].element_ptr(s2d_val_index++));
                side_index_data.to_data_type(int_dtype.id(), misc_data);

                misc_data.set_external(DataType(int_dtype.id(), 1),
                    d2smap["values"].element_ptr(d2s_val_index++));
                elem_index_data.to_data_type(int_dtype.id(), misc_data);

                int64 side_num_elems = 1;
                raw_data.set(side_num_elems);
                misc_data.set_external(DataType(int_dtype.id(), 1),
                    d2smap["sizes"].element_ptr(d2s_elem_index++));
                raw_data.to_data_type(int_dtype.id(), misc_data);

                side_index++;
            }
        }

        int64 elem_num_sides = s2d_val_index - s2d_start_index;
        raw_data.set(elem_num_sides);
        misc_data.set_external(DataType(int_dtype.id(), 1),
            s2dmap["sizes"].element_ptr(s2d_elem_index++));
        raw_data.to_data_type(int_dtype.id(), misc_data);
    }

    // TODO(JRC): Implement these counts in-line instead of being lazy and
    // taking care of it at the end of the function w/ a helper.
    Node info;
    blueprint::o2mrelation::generate_offsets(s2dmap, info);
    blueprint::o2mrelation::generate_offsets(d2smap, info);
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_corners(const Node &topo,
                                               Node &dest,
                                               Node &cdest,
                                               Node &s2dmap,
                                               Node &d2smap)
{
    // Retrieve Relevent Coordinate/Topology Metadata //

    Node coordset;
    find_reference_node(topo, "coordset", coordset);
    const std::vector<std::string> csys_axes = identify_coordset_axes(coordset);

    const ShapeCascade topo_cascade(topo);
    const ShapeType topo_shape = topo_cascade.get_shape();
    const bool is_topo_3d = topo_shape.dim == 3;
    const ShapeType point_shape = topo_cascade.get_shape(0);
    const ShapeType line_shape = topo_cascade.get_shape(1);
    const ShapeType face_shape = topo_cascade.get_shape(2);
    const ShapeType corner_shape(topo_shape.dim == 3 ? "polyhedral" : "polygonal");
    if(topo_shape.dim < 2)
    {
        CONDUIT_ERROR("Failed to generate corner mesh for input; " <<
            "input tology must be topologically 2D or 3D.");
    }

    // Extract Derived Coordinate/Topology Data //

    const TopologyMetadata topo_data(topo, coordset);
    const index_t topo_num_elems = topo_data.get_length(topo_shape.dim);
    const DataType &int_dtype = topo_data.int_dtype;
    const DataType &float_dtype = topo_data.float_dtype;

    std::vector<conduit::Node> dim_cent_topos(topo_shape.dim + 1);
    std::vector<conduit::Node> dim_cent_coords(topo_shape.dim + 1);
    for(index_t di = 0; di <= topo_shape.dim; di++)
    {
        calculate_unstructured_centroids(
            topo_data.dim_topos[di], coordset,
            dim_cent_topos[di], dim_cent_coords[di]);
    }

    // Allocate Data Templates for Outputs //

    const index_t corners_num_coords = topo_data.get_length();
    const index_t corners_face_degree = 4;

    dest.reset();
    dest["type"].set("unstructured");
    dest["coordset"].set(cdest.name());
    dest["elements/shape"].set(corner_shape.type);
    if (is_topo_3d)
    {
        dest["subelements/shape"].set("polygonal");
    }
    // TODO(JRC): I wasn't able to find a good way to compute the connectivity
    // length a priori because of the possibility of polygonal 3D inputs, but
    // having this information would improve the performance of the method.
    // dest["elements/connectivity"].set(DataType(int_dtype.id(), ???);

    cdest.reset();
    cdest["type"].set("explicit");
    for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
    {
        cdest["values"][csys_axes[ai]].set(DataType(float_dtype.id(),
            corners_num_coords));
    }

    s2dmap.reset();
    d2smap.reset();

    // Populate Data Arrays w/ Calculated Coordinates //

    std::vector<index_t> dim_coord_offsets(topo_shape.dim + 1);
    for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
    {
        Node dst_data;
        Node &dst_axis = cdest["values"][csys_axes[ai]];

        for(index_t di = 0, doffset = 0; di <= topo_shape.dim; di++)
        {
            dim_coord_offsets[di] = doffset;

            // NOTE: The centroid ordering for the positions is different
            // from the base ordering, which messes up all subsequent indexing.
            // We must use the coordinate set associated with the base topology.
            const Node &cset = (di != 0) ? dim_cent_coords[di] : coordset;
            const Node &cset_axis = cset["values"][csys_axes[ai]];
            index_t cset_length = cset_axis.dtype().number_of_elements();

            dst_data.set_external(DataType(float_dtype.id(), cset_length),
                dst_axis.element_ptr(doffset));
            cset_axis.to_data_type(float_dtype.id(), dst_data);
            doffset += cset_length;
        }
    }

    // Compute New Elements/Fields for Corner Topology //

    std::vector<int64> conn_data_raw, size_data_raw;
    std::vector<int64> subconn_data_raw, subsize_data_raw;
    std::vector<int64> s2d_idx_data_raw, s2d_size_data_raw;
    std::vector<int64> d2s_idx_data_raw, d2s_size_data_raw;
    std::map< std::set<index_t>, index_t > subconn_topo_set;

    for(index_t elem_index = 0, corner_index = 0; elem_index < topo_num_elems; elem_index++)
    {
        // per-face, per-line orientations for this element, i.e. {(f_gi, l_gj) => (v_gk, v_gl)}
        std::map< std::pair<index_t, index_t>, std::pair<index_t, index_t> > elem_orient;
        { // establish the element's internal line constraints
            const std::vector<index_t> &elem_faces = topo_data.get_entity_assocs(
                TopologyMetadata::LOCAL, elem_index, topo_shape.dim, face_shape.dim);
            for(index_t fi = 0; fi < (index_t)elem_faces.size(); fi++)
            {
                const index_t face_lid = elem_faces[fi];
                const index_t face_gid = topo_data.dim_le2ge_maps[face_shape.dim][face_lid];

                const std::vector<index_t> &face_lines = topo_data.get_entity_assocs(
                    TopologyMetadata::LOCAL, face_lid, face_shape.dim, line_shape.dim);
                for(index_t li = 0; li < (index_t)face_lines.size(); li++)
                {
                    const index_t line_lid = face_lines[li];
                    const index_t line_gid = topo_data.dim_le2ge_maps[line_shape.dim][line_lid];

                    const std::vector<index_t> &line_points = topo_data.get_entity_assocs(
                        TopologyMetadata::LOCAL, line_lid, line_shape.dim, point_shape.dim);
                    const index_t start_gid = topo_data.dim_le2ge_maps[point_shape.dim][line_points[0]];
                    const index_t end_gid = topo_data.dim_le2ge_maps[point_shape.dim][line_points[1]];

                    elem_orient[std::make_pair(face_gid, line_gid)] =
                        std::make_pair(start_gid, end_gid);
                }
            }
        }

        const std::vector<index_t> &elem_lines = topo_data.get_entity_assocs(
            TopologyMetadata::GLOBAL, elem_index, topo_shape.dim, line_shape.dim);
        const std::vector<index_t> &elem_faces = topo_data.get_entity_assocs(
            TopologyMetadata::GLOBAL, elem_index, topo_shape.dim, face_shape.dim);

        // NOTE(JRC): Corner ordering retains original element orientation
        // by creating elements as follows:
        //
        // - for a given element, determine how its co-faces and co-lines are
        //   oriented, and set these as constraints
        // - based on these constraints, create the co-line/co-face centroid
        //   corner lines, which add a new set of contraints
        // - finally, if the topology is 3D, create the co-face/cell centroid
        //   corner lines based on all previous constraints, and then collect
        //   these final lines into corner faces
        //
        // To better demonstrate this algorithm, here's a simple 2D example:
        //
        // - Top-Level Element/Constraints (See Arrows)
        //
        //   p2      l2      p3
        //   +<---------------+
        //   |                ^
        //   |                |
        //   |                |
        // l3|       f0       |l1
        //   |                |
        //   |                |
        //   v                |
        //   +--------------->+
        //   p0      l0      p1
        //
        // - Consider Corner f0/p0 and Centroids; Impose Top-Level Constraints
        //
        //   p2      l2      p3
        //   +----------------+
        //   |                |
        //   |                |
        //   |       f0       |
        // l3+       +        |l1
        //   |                |
        //   |                |
        //   v                |
        //   +------>+--------+
        //   p0      l0      p1
        //
        // - Create Face/Line Connections Based on Top-Level Constraints
        //
        //   p2      l2      p3
        //   +----------------+
        //   |                |
        //   |                |
        //   |       f0       |
        // l3+<------+        |l1
        //   |       ^        |
        //   |       |        |
        //   v       |        |
        //   +------>+--------+
        //   p0      l0      p1
        //

        // per-elem, per-point corners, informed by cell-face-line orientation constraints
        const std::vector<index_t> &elem_points = topo_data.get_entity_assocs(
            TopologyMetadata::GLOBAL, elem_index, topo_shape.dim, point_shape.dim);
        for(index_t pi = 0; pi < (index_t)elem_points.size(); pi++, corner_index++)
        {
            const index_t point_index = elem_points[pi];

            const std::vector<index_t> &point_faces = topo_data.get_entity_assocs(
                TopologyMetadata::GLOBAL, point_index, point_shape.dim, face_shape.dim);
            const std::vector<index_t> &point_lines = topo_data.get_entity_assocs(
                TopologyMetadata::GLOBAL, point_index, point_shape.dim, line_shape.dim);
            const std::vector<index_t> elem_point_faces = intersect_sets(
                elem_faces, point_faces);
            const std::vector<index_t> elem_point_lines = intersect_sets(
                elem_lines, point_lines);

            // per-corner face vertex orderings, informed by 'corner_orient'
            std::vector< std::vector<index_t> > corner_faces(
                // # of faces per corner: len(v.faces & c.faces) * (2 if is_3d else 1)
                elem_point_faces.size() * (is_topo_3d ? 2 : 1),
                // # of vertices per face: 4 (all faces are quads in corner topology)
                std::vector<index_t>(corners_face_degree, 0));
            // per-face, per-line orientations for this corner, i.e. {(f_gi, l_gj) => bool}
            std::map< std::pair<index_t, index_t>, bool > corner_orient;
            // flags for the 'corner_orient' map; if TO_FACE, line is (l_gj, f_gi);
            // if FROM_FACE, line is (f_gi, l_gj)
            const static bool TO_FACE = true, FROM_FACE = false;

            // generate oriented corner-to-face faces using internal line constraints
            for(index_t fi = 0; fi < (index_t)elem_point_faces.size(); fi++)
            {
                const index_t face_index = elem_point_faces[fi];

                const std::vector<index_t> &elem_face_lines = topo_data.get_entity_assocs(
                    TopologyMetadata::GLOBAL, face_index, face_shape.dim, line_shape.dim);
                const std::vector<index_t> corner_face_lines = intersect_sets(
                    elem_face_lines, point_lines);

                std::vector<index_t> &corner_face = corner_faces[fi];
                {
                    corner_face[0] = point_index;
                    corner_face[2] = face_index;

                    const index_t first_line_index = corner_face_lines.front();
                    const index_t second_line_index = corner_face_lines.back();
                    const auto first_line_pair = std::make_pair(face_index, first_line_index);
                    const auto second_line_pair = std::make_pair(face_index, second_line_index);

                    const bool is_first_forward = elem_orient[first_line_pair].first == point_index;
                    corner_face[1] = is_first_forward ? first_line_index : second_line_index;
                    corner_face[3] = is_first_forward ? second_line_index : first_line_index;
                    corner_orient[first_line_pair] = is_first_forward ? TO_FACE : FROM_FACE;
                    corner_orient[second_line_pair] = is_first_forward ? FROM_FACE : TO_FACE;

                    // NOTE(JRC): The non-corner points are centroids and thus
                    // need to be offset relative to their dimensional position.
                    corner_face[0] += dim_coord_offsets[point_shape.dim];
                    corner_face[1] += dim_coord_offsets[line_shape.dim];
                    corner_face[3] += dim_coord_offsets[line_shape.dim];
                    corner_face[2] += dim_coord_offsets[face_shape.dim];
                }
            }
            // generate oriented line-to-cell faces using corner-to-face constraints from above
            for(index_t li = 0; li < (index_t)elem_point_lines.size() && is_topo_3d; li++)
            {
                const index_t line_index = elem_point_lines[li];

                const std::vector<index_t> &line_faces = topo_data.get_entity_assocs(
                    TopologyMetadata::GLOBAL, line_index, line_shape.dim, face_shape.dim);
                const std::vector<index_t> corner_line_faces = intersect_sets(
                    elem_faces, line_faces);

                std::vector<index_t> &corner_face = corner_faces[elem_point_faces.size() + li];
                {
                    corner_face[0] = line_index;
                    corner_face[2] = elem_index;

                    const index_t first_face_index = corner_line_faces.front();
                    const index_t second_face_index = corner_line_faces.back();
                    const auto first_face_pair = std::make_pair(first_face_index, line_index);
                    // const auto second_face_pair = std::make_pair(second_face_index, line_index);

                    // NOTE(JRC): The current corner face will use the co-edge of the existing
                    // edge in 'corner_orient', so we flip the orientation for the local use.
                    const bool is_first_forward = !corner_orient[first_face_pair];
                    corner_face[1] = is_first_forward ? first_face_index : second_face_index;
                    corner_face[3] = is_first_forward ? second_face_index : first_face_index;

                    // NOTE(JRC): The non-corner points are centroids and thus
                    // need to be offset relative to their dimensional position.
                    corner_face[0] += dim_coord_offsets[line_shape.dim];
                    corner_face[1] += dim_coord_offsets[face_shape.dim];
                    corner_face[3] += dim_coord_offsets[face_shape.dim];
                    corner_face[2] += dim_coord_offsets[topo_shape.dim];
                }
            }

            if(!is_topo_3d)
            {
                const std::vector<index_t> &corner_face = corner_faces.front();
                size_data_raw.push_back(corner_face.size());
                conn_data_raw.insert(conn_data_raw.end(),
                    corner_face.begin(), corner_face.end());
            }
            else // if(is_topo_3d)
            {
                size_data_raw.push_back(corner_faces.size());
                for(index_t fi = 0; fi < (index_t)corner_faces.size(); fi++)
                {
                    const std::vector<index_t> &corner_face = corner_faces[fi];
                    // TODO(JRC): For now, we retain the behavior of storing only
                    // unique faces in the subconnectivity for 3D corners, but
                    // this can be easily changed by modifying the logic below.
                    const std::set<index_t> corner_face_set(corner_face.begin(), corner_face.end());
                    if(subconn_topo_set.find(corner_face_set) == subconn_topo_set.end())
                    {
                        const index_t next_face_index = subconn_topo_set.size();
                        subconn_topo_set[corner_face_set] = next_face_index;
                        subsize_data_raw.push_back(corner_face_set.size());
                        subconn_data_raw.insert(subconn_data_raw.end(),
                            corner_face.begin(), corner_face.end());
                    }
                    const index_t face_index = subconn_topo_set.find(corner_face_set)->second;
                    conn_data_raw.push_back(face_index);
                }
            }

            s2d_idx_data_raw.push_back(corner_index);
            d2s_size_data_raw.push_back(1);
            d2s_idx_data_raw.push_back(elem_index);
        }

        s2d_size_data_raw.push_back(elem_points.size());
    }

    Node raw_data, info;
    {
        raw_data.set_external(
            DataType::int64(conn_data_raw.size()),
            conn_data_raw.data());
        raw_data.to_data_type(int_dtype.id(),
            dest["elements/connectivity"]);
        raw_data.set_external(
            DataType::int64(size_data_raw.size()),
            size_data_raw.data());
        raw_data.to_data_type(int_dtype.id(),
            dest["elements/sizes"]);

        if (is_topo_3d)
        {
            raw_data.set_external(
                DataType::int64(subconn_data_raw.size()),
                subconn_data_raw.data());
            raw_data.to_data_type(int_dtype.id(),
                dest["subelements/connectivity"]);
            raw_data.set_external(
                DataType::int64(subsize_data_raw.size()),
                subsize_data_raw.data());
            raw_data.to_data_type(int_dtype.id(),
                dest["subelements/sizes"]);
        }

        raw_data.set_external(
            DataType::int64(s2d_idx_data_raw.size()),
            s2d_idx_data_raw.data());
        raw_data.to_data_type(int_dtype.id(), s2dmap["values"]);
        raw_data.set_external(
            DataType::int64(s2d_size_data_raw.size()),
            s2d_size_data_raw.data());
        raw_data.to_data_type(int_dtype.id(), s2dmap["sizes"]);

        raw_data.set_external(
            DataType::int64(d2s_idx_data_raw.size()),
            d2s_idx_data_raw.data());
        raw_data.to_data_type(int_dtype.id(), d2smap["values"]);
        raw_data.set_external(
            DataType::int64(d2s_size_data_raw.size()),
            d2s_size_data_raw.data());
        raw_data.to_data_type(int_dtype.id(), d2smap["sizes"]);

        // TODO(JRC): Implement these counts in-line instead of being lazy and
        // taking care of it at the end of the function w/ a helper.
        generate_offsets(dest, dest["elements/offsets"]);
        blueprint::o2mrelation::generate_offsets(s2dmap, info);
        blueprint::o2mrelation::generate_offsets(d2smap, info);
    }
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_offsets(const Node &topo,
                                               Node &dest)
{
    const ShapeType topo_shape(topo);
    const DataType int_dtype = find_widest_dtype(topo, blueprint::mesh::default_int_dtypes);
    const Node &topo_conn = topo["elements/connectivity"];

    const DataType topo_dtype(topo_conn.dtype().id(), 1, 0, 0,
        topo_conn.dtype().element_bytes(), topo_conn.dtype().endianness());

    if(!topo_shape.is_poly())
    {
        dest.reset();

        const index_t num_topo_shapes =
            topo_conn.dtype().number_of_elements() / topo_shape.indices;

        Node shape_node(DataType::int64(num_topo_shapes));
        int64_array shape_array = shape_node.as_int64_array();
        for(index_t s = 0; s < num_topo_shapes; s++)
        {
            shape_array[s] = s * topo_shape.indices;
        }
        shape_node.to_data_type(int_dtype.id(), dest);
    }
    else if(topo_shape.type == "polygonal")
    {
        dest.reset();

        const Node &topo_size = topo["elements/sizes"];
        std::vector<int64> shape_array;
        index_t i = 0;
        index_t s = 0;
        while(i < topo_size.dtype().number_of_elements())
        {
            const Node index_node(int_dtype,
                const_cast<void*>(topo_size.element_ptr(i)), true);
            shape_array.push_back(s);
            s += index_node.to_int64();
            i++;
        }

        Node shape_node;
        shape_node.set_external(shape_array);
        shape_node.to_data_type(int_dtype.id(), dest);
    }
    else if(topo_shape.type == "polyhedral")
    {
        Node &dest_elem_off = const_cast<Node &>(topo)["elements/offsets"];
        Node &dest_subelem_off = const_cast<Node &>(topo)["subelements/offsets"];

        const Node& topo_elem_size = topo["elements/sizes"];
        const Node& topo_subelem_size = topo["subelements/sizes"];

        Node elem_node;
        Node subelem_node;

        std::vector<int64> shape_array;
        index_t ei = 0;
        index_t es = 0;
        while(ei < topo_elem_size.dtype().number_of_elements())
        {
            const Node index_node(int_dtype,
                const_cast<void*>(topo_elem_size.element_ptr(ei)), true);
            shape_array.push_back(es);
            es += index_node.to_int64();
            ei++;
        }

        elem_node.set_external(shape_array);
        elem_node.to_data_type(int_dtype.id(), dest_elem_off);
        elem_node.to_data_type(int_dtype.id(), dest);

        shape_array.clear();
        ei = 0;
        es = 0;
        while(ei < topo_subelem_size.dtype().number_of_elements())
        {
            const Node index_node(int_dtype,
                const_cast<void*>(topo_subelem_size.element_ptr(ei)), true);
            shape_array.push_back(es);
            es += index_node.to_int64();
            ei++;
        }

        subelem_node.set_external(shape_array);
        subelem_node.to_data_type(int_dtype.id(), dest_subelem_off);
    }
}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::index protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::index::verify(const Node &topo_idx,
                              Node &info)
{
    const std::string protocol = "mesh::topology::index";
    bool res = true;
    info.reset();

    res &= verify_field_exists(protocol, topo_idx, info, "type") &&
           mesh::topology::type::verify(topo_idx["type"], info["type"]);
    res &= verify_string_field(protocol, topo_idx, info, "coordset");
    res &= verify_string_field(protocol, topo_idx, info, "path");

    if (topo_idx.has_child("grid_function"))
    {
        log::optional(info, protocol, "includes grid_function");
        res &= verify_string_field(protocol, topo_idx, info, "grid_function");
    }

    log::validation(info,res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::type protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::type::verify(const Node &type,
                             Node &info)
{
    const std::string protocol = "mesh::topology::type";
    bool res = true;
    info.reset();

    res &= verify_enum_field(protocol, type, info, "", mesh::topo_types);

    log::validation(info,res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::shape protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::shape::verify(const Node &shape,
                              Node &info)
{
    const std::string protocol = "mesh::topology::shape";
    bool res = true;
    info.reset();

    res &= verify_enum_field(protocol, shape, info, "", mesh::topo_shapes);

    log::validation(info,res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::matset protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::matset::verify(const Node &matset,
                     Node &info)
{
    const std::string protocol = "mesh::matset";
    bool res = true, vfs_res = true;
    info.reset();

    res &= verify_string_field(protocol, matset, info, "topology");
    res &= vfs_res &= verify_field_exists(protocol, matset, info, "volume_fractions");

    if(vfs_res)
    {
        if(!matset["volume_fractions"].dtype().is_number() &&
            !matset["volume_fractions"].dtype().is_object())
        {
            log::error(info, protocol, "'volume_fractions' isn't the correct type");
            res &= vfs_res &= false;
        }
        else if(matset["volume_fractions"].dtype().is_number() &&
            verify_number_field(protocol, matset, info, "volume_fractions"))
        {
            log::info(info, protocol, "detected uni-buffer matset");

            vfs_res &= verify_integer_field(protocol, matset, info, "material_ids");
            // TODO(JRC): Add a more in-depth verifier for 'material_map' that
            // verifies that it's one level deep and that each child child houses
            // an integer-style array.
            vfs_res &= verify_object_field(protocol, matset, info, "material_map");
            vfs_res &= blueprint::o2mrelation::verify(matset, info);

            res &= vfs_res;
        }
        else if(matset["volume_fractions"].dtype().is_object() &&
            verify_object_field(protocol, matset, info, "volume_fractions"))
        {
            log::info(info, protocol, "detected multi-buffer matset");

            const Node &vfs = matset["volume_fractions"];
            Node &vfs_info = info["volume_fractions"];

            NodeConstIterator mat_it = vfs.children();
            while(mat_it.has_next())
            {
                const Node &mat = mat_it.next();
                const std::string &mat_name = mat_it.name();

                if(mat.dtype().is_object())
                {
                    vfs_res &= verify_o2mrelation_field(protocol, vfs, vfs_info, mat_name);
                }
                else
                {
                    vfs_res &= verify_number_field(protocol, vfs, vfs_info, mat_name);
                }
            }

            res &= vfs_res;
            log::validation(vfs_info, vfs_res);
        }
    }

    if(matset.has_child("element_ids"))
    {
        bool eids_res = true;

        if(vfs_res)
        {
            if(!matset["element_ids"].dtype().is_integer() &&
                !matset["element_ids"].dtype().is_object())
            {
                log::error(info, protocol, "'element_ids' isn't the correct type");
                res &= eids_res &= false;
            }
            else if(matset["element_ids"].dtype().is_object() &&
                matset["volume_fractions"].dtype().is_object())
            {
                const std::vector<std::string> &vf_mats = matset["volume_fractions"].child_names();
                const std::vector<std::string> &eid_mats = matset["element_ids"].child_names();
                const std::set<std::string> vf_matset(vf_mats.begin(), vf_mats.end());
                const std::set<std::string> eid_matset(eid_mats.begin(), eid_mats.end());
                if(vf_matset != eid_matset)
                {
                    log::error(info, protocol, "'element_ids' hierarchy must match 'volume_fractions'");
                    eids_res &= false;
                }

                const Node &eids = matset["element_ids"];
                Node &eids_info = info["element_ids"];

                NodeConstIterator mat_it = eids.children();
                while(mat_it.has_next())
                {
                    const std::string &mat_name = mat_it.next().name();
                    eids_res &= verify_integer_field(protocol, eids, eids_info, mat_name);
                }

                res &= eids_res;
                log::validation(eids_info, eids_res);
            }
            else if(matset["element_ids"].dtype().is_integer() &&
                matset["volume_fractions"].dtype().is_number())
            {
                res &= eids_res &= verify_integer_field(protocol, matset, info, "element_ids");
            }
            else
            {
                log::error(info, protocol, "'element_ids' hierarchy must match 'volume_fractions'");
                res &= eids_res &= false;
            }
        }
    }

    log::validation(info, res);

    return res;
}

//-------------------------------------------------------------------------
bool
mesh::matset::is_multi_buffer(const Node &matset)
{
    return matset.child("volume_fractions").dtype().is_object();
}

//-------------------------------------------------------------------------
bool
mesh::matset::is_uni_buffer(const Node &matset)
{
    return matset.child("volume_fractions").dtype().is_number();
}

//-------------------------------------------------------------------------
bool
mesh::matset::is_element_dominant(const Node &matset)
{
    return !matset.has_child("element_ids");
}

//-------------------------------------------------------------------------
bool
mesh::matset::is_material_dominant(const Node &matset)
{
    return matset.has_child("element_ids");
}

//-----------------------------------------------------------------------------
// blueprint::mesh::matset::index protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::matset::index::verify(const Node &matset_idx,
                            Node &info)
{
    const std::string protocol = "mesh::matset::index";
    bool res = true;
    info.reset();

    // TODO(JRC): Determine whether or not extra verification needs to be
    // performed on the "materials" field.

    res &= verify_string_field(protocol, matset_idx, info, "topology");

    // 2021-1-29 cyrush:
    // prefer new "material_map" index spec, vs old materials
    if(matset_idx.has_child("material_map"))
    {
        res &= verify_object_field(protocol, matset_idx, info, "material_map");
    }
    else
    {
        res &= verify_object_field(protocol, matset_idx, info, "materials");
    }

    res &= verify_string_field(protocol, matset_idx, info, "path");

    log::validation(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::field protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::field::verify(const Node &field,
                    Node &info)
{
    const std::string protocol = "mesh::field";
    bool res = true;
    info.reset();

    bool has_assoc = field.has_child("association");
    bool has_basis = field.has_child("basis");
    if(!has_assoc && !has_basis)
    {
        log::error(info, protocol, "missing child 'association' or 'basis'");
        res = false;
    }
    if(has_assoc)
    {
        res &= mesh::association::verify(field["association"], info["association"]);
    }
    if(has_basis)
    {
        res &= mesh::field::basis::verify(field["basis"], info["basis"]);
    }

    bool has_topo = field.has_child("topology");
    bool has_matset = field.has_child("matset");
    bool has_topo_values = field.has_child("values");
    bool has_matset_values = field.has_child("matset_values");
    if(!has_topo && !has_matset)
    {
        log::error(info, protocol, "missing child 'topology' or 'matset'");
        res = false;
    }

    if(has_topo ^ has_topo_values)
    {
        std::ostringstream oss;
        oss << "'" << (has_topo ? "topology" : "values") <<"'"
            << " is present, but its companion "
            << "'" << (has_topo ? "values" : "topology") << "'"
            << " is missing";
        log::error(info, protocol, oss.str());
        res = false;
    }
    else if(has_topo && has_topo_values)
    {
        res &= verify_string_field(protocol, field, info, "topology");
        res &= verify_mlarray_field(protocol, field, info, "values", 0, 1, false);
    }

    if(has_matset ^ has_matset_values)
    {
        std::ostringstream oss;
        oss << "'" << (has_matset ? "matset" : "matset_values") <<"'"
            << " is present, but its companion "
            << "'" << (has_matset ? "matset_values" : "matset") << "'"
            << " is missing";
        log::error(info, protocol, oss.str());
        res = false;
    }
    else if(has_matset && has_matset_values)
    {
        res &= verify_string_field(protocol, field, info, "matset");
        res &= verify_mlarray_field(protocol, field, info, "matset_values", 0, 2, false);
    }

    // TODO(JRC): Enable 'volume_dependent' once it's confirmed to be a required
    // entry for fields.
    // res &= verify_enum_field(protocol, field, info, "volume_dependent", mesh::booleans);

    log::validation(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::field::basis protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::field::basis::verify(const Node &basis,
                           Node &info)
{
    const std::string protocol = "mesh::field::basis";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, basis, info);

    log::validation(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::field::index protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::field::index::verify(const Node &field_idx,
                           Node &info)
{
    const std::string protocol = "mesh::field::index";
    bool res = true;
    info.reset();

    bool has_assoc = field_idx.has_child("association");
    bool has_basis = field_idx.has_child("basis");
    if(!has_assoc && !has_basis)
    {
        log::error(info, protocol, "missing child 'association' or 'basis'");
        res = false;
    }
    if(has_assoc)
    {
        res &= mesh::association::verify(field_idx["association"], info["association"]);
    }
    if(has_basis)
    {
        res &= mesh::field::basis::verify(field_idx["basis"], info["basis"]);
    }

    bool has_topo = field_idx.has_child("topology");
    bool has_matset = field_idx.has_child("matset");
    if(!has_topo && !has_matset)
    {
        log::error(info, protocol, "missing child 'topology' or 'matset'");
        res = false;
    }
    if(has_topo)
    {
        res &= verify_string_field(protocol, field_idx, info, "topology");
    }
    if(has_matset)
    {
        res &= verify_string_field(protocol, field_idx, info, "matset");
    }

    res &= verify_integer_field(protocol, field_idx, info, "number_of_components");
    res &= verify_string_field(protocol, field_idx, info, "path");

    log::validation(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::specset protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::specset::verify(const Node &specset,
                      Node &info)
{
    const std::string protocol = "mesh::specset";
    bool res = true;
    info.reset();

    // TODO(JRC): Enable 'volume_dependent' once it's confirmed to be a required
    // entry for specsets.
    // res &= verify_enum_field(protocol, specset, info, "volume_dependent", mesh::booleans);
    res &= verify_string_field(protocol, specset, info, "matset");
    if(!verify_object_field(protocol, specset, info, "matset_values"))
    {
        res &= false;
    }
    else
    {
        bool specmats_res = true;
        index_t specmats_len = 0;

        const Node &specmats = specset["matset_values"];
        Node &specmats_info = info["matset_values"];
        NodeConstIterator specmats_it = specmats.children();
        while(specmats_it.has_next())
        {
            const Node &specmat = specmats_it.next();
            const std::string specmat_name = specmat.name();
            if(!verify_mcarray_field(protocol, specmats, specmats_info, specmat_name))
            {
                specmats_res &= false;
            }
            else
            {
                const index_t specmat_len = specmat.child(0).dtype().number_of_elements();
                if(specmats_len == 0)
                {
                    specmats_len = specmat_len;
                }
                else if(specmats_len != specmat_len)
                {
                    log::error(specmats_info, protocol,
                        log::quote(specmat_name) + " has mismatched length " +
                        "relative to other material mcarrays in this specset");
                    specmats_res &= false;
                }
            }
        }

        log::validation(specmats_info, specmats_res);
        res &= specmats_res;
    }

    log::validation(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::specset::index::verify protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::specset::index::verify(const Node &specset_idx,
                             Node &info)
{
    const std::string protocol = "mesh::specset::index";
    bool res = true;
    info.reset();

    // TODO(JRC): Determine whether or not extra verification needs to be
    // performed on the "species" field.

    res &= verify_string_field(protocol, specset_idx, info, "matset");
    res &= verify_object_field(protocol, specset_idx, info, "species");
    res &= verify_string_field(protocol, specset_idx, info, "path");

    log::validation(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::adjset protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::adjset::verify(const Node &adjset,
                     Node &info)
{
    const std::string protocol = "mesh::adjset";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, adjset, info, "topology");
    res &= verify_field_exists(protocol, adjset, info, "association") &&
           mesh::association::verify(adjset["association"], info["association"]);

    if(!verify_object_field(protocol, adjset, info, "groups"))
    {
        res = false;
    }
    else
    {
        bool groups_res = true;
        NodeConstIterator itr = adjset["groups"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            Node &chld_info = info["groups"][chld_name];

            bool group_res = true;
            group_res &= verify_integer_field(protocol, chld, chld_info, "neighbors");
            if(chld.has_child("values"))
            {
                group_res &= verify_integer_field(protocol, chld,
                    chld_info, "values");
            }
            else if(chld.has_child("windows"))
            {

                group_res &= verify_object_field(protocol, chld,
                    chld_info, "windows");

                bool windows_res = true;
                NodeConstIterator witr = chld["windows"].children();
                while(witr.has_next())
                {
                    const Node &wndw = witr.next();
                    const std::string wndw_name = witr.name();
                    Node &wndw_info = chld_info["windows"][wndw_name];

                    bool window_res = true;
                    window_res &= verify_field_exists(protocol, wndw,
                        wndw_info, "origin") &&
                        mesh::logical_dims::verify(wndw["origin"],
                            wndw_info["origin"]);
                    window_res &= verify_field_exists(protocol, wndw,
                        wndw_info, "dims") &&
                        mesh::logical_dims::verify(wndw["dims"],
                            wndw_info["dims"]);
                    window_res &= verify_field_exists(protocol, wndw,
                        wndw_info, "ratio") &&
                        mesh::logical_dims::verify(wndw["ratio"],
                            wndw_info["ratio"]);

                    // verify that dimensions for "origin" and
                    // "dims" and "ratio" are the same
                    if(window_res)
                    {
                        index_t window_dim = wndw["origin"].number_of_children();
                        window_res &= !wndw.has_child("dims") ||
                            verify_object_field(protocol, wndw,
                                wndw_info, "dims", false, window_dim);
                        window_res &= !wndw.has_child("ratio") ||
                            verify_object_field(protocol, wndw,
                                wndw_info, "ratio", false, window_dim);
                    }

                    log::validation(wndw_info,window_res);
                    windows_res &= window_res;
                }

                log::validation(chld_info["windows"],windows_res);
                res &= windows_res;

                if(chld.has_child("orientation"))
                {
                    group_res &= verify_integer_field(protocol, chld,
                        chld_info, "orientation");
                }
            }

            log::validation(chld_info,group_res);
            groups_res &= group_res;
        }

        log::validation(info["groups"],groups_res);
        res &= groups_res;
    }

    log::validation(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::adjset::index protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::adjset::index::verify(const Node &adj_idx,
                            Node &info)
{
    const std::string protocol = "mesh::adjset::index";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, adj_idx, info, "topology");
    res &= verify_field_exists(protocol, adj_idx, info, "association") &&
           mesh::association::verify(adj_idx["association"], info["association"]);
    res &= verify_string_field(protocol, adj_idx, info, "path");

    log::validation(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::nestset protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::nestset::verify(const Node &nestset,
                      Node &info)
{
    const std::string protocol = "mesh::nestset";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, nestset, info, "topology");
    res &= verify_field_exists(protocol, nestset, info, "association") &&
           mesh::association::verify(nestset["association"], info["association"]);

    if(!verify_object_field(protocol, nestset, info, "windows"))
    {
        res = false;
    }
    else
    {
        bool windows_res = true;
        NodeConstIterator itr = nestset["windows"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            Node &chld_info = info["windows"][chld_name];

            bool window_res = true;
            window_res &= verify_integer_field(protocol, chld, chld_info, "domain_id");
            window_res &= verify_field_exists(protocol, chld, chld_info, "domain_type") &&
                mesh::nestset::type::verify(chld["domain_type"], chld_info["domain_type"]);

            window_res &= verify_field_exists(protocol, chld, chld_info, "ratio") &&
                mesh::logical_dims::verify(chld["ratio"], chld_info["ratio"]);
            window_res &= !chld.has_child("origin") ||
                mesh::logical_dims::verify(chld["origin"], chld_info["origin"]);
            window_res &= !chld.has_child("dims") ||
                mesh::logical_dims::verify(chld["dims"], chld_info["dims"]);

            // one last pass: verify that dimensions for "ratio", "origin", and
            // "dims" are all the same
            if(window_res)
            {
                index_t window_dim = chld["ratio"].number_of_children();
                window_res &= !chld.has_child("origin") ||
                    verify_object_field(protocol, chld, chld_info, "origin", false, window_dim);
                window_res &= !chld.has_child("dims") ||
                    verify_object_field(protocol, chld, chld_info, "dims", false, window_dim);
            }

            log::validation(chld_info,window_res);
            windows_res &= window_res;
        }

        log::validation(info["windows"],windows_res);
        res &= windows_res;
    }

    log::validation(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::nestset::index protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::nestset::index::verify(const Node &nest_idx,
                            Node &info)
{
    const std::string protocol = "mesh::nestset::index";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, nest_idx, info, "topology");
    res &= verify_field_exists(protocol, nest_idx, info, "association") &&
           mesh::association::verify(nest_idx["association"], info["association"]);
    res &= verify_string_field(protocol, nest_idx, info, "path");

    log::validation(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::type protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::nestset::type::verify(const Node &type,
                            Node &info)
{
    const std::string protocol = "mesh::nestset::type";
    bool res = true;
    info.reset();

    res &= verify_enum_field(protocol, type, info, "", mesh::nestset_types);

    log::validation(info,res);

    return res;
}


//-----------------------------------------------------------------------------
// blueprint::mesh::index protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::index::verify(const Node &n,
                    Node &info)
{
    const std::string protocol = "mesh::index";
    bool res = true;
    info.reset();

    if(!verify_object_field(protocol, n, info, "coordsets"))
    {
        res = false;
    }
    else
    {
        bool cset_res = true;
        NodeConstIterator itr = n["coordsets"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            cset_res &= coordset::index::verify(chld, info["coordsets"][chld_name]);
        }

        log::validation(info["coordsets"],cset_res);
        res &= cset_res;
    }

    if(!verify_object_field(protocol, n, info, "topologies"))
    {
        res = false;
    }
    else
    {
        bool topo_res = true;
        NodeConstIterator itr = n["topologies"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            Node &chld_info = info["topologies"][chld_name];

            topo_res &= topology::index::verify(chld, chld_info);
            topo_res &= verify_reference_field(protocol, n, info,
                chld, chld_info, "coordset", "coordsets");
        }

        log::validation(info["topologies"],topo_res);
        res &= topo_res;
    }

    // optional: "matsets", each child must conform to
    // "mesh::index::matset"
    if(n.has_path("matsets"))
    {
        if(!verify_object_field(protocol, n, info, "matsets"))
        {
            res = false;
        }
        else
        {
            bool mset_res = true;
            NodeConstIterator itr = n["matsets"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["matsets"][chld_name];

                mset_res &= matset::index::verify(chld, chld_info);
                mset_res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "topology", "topologies");
            }

            log::validation(info["matsets"],mset_res);
            res &= mset_res;
        }
    }

    // optional: "specsets", each child must conform to
    // "mesh::index::specset"
    if(n.has_path("specsets"))
    {
        if(!verify_object_field(protocol, n, info, "specsets"))
        {
            res = false;
        }
        else
        {
            bool sset_res = true;
            NodeConstIterator itr = n["specsets"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["specsets"][chld_name];

                sset_res &= specset::index::verify(chld, chld_info);
                sset_res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "matset", "matsets");
            }

            log::validation(info["specsets"],sset_res);
            res &= sset_res;
        }
    }

    // optional: "fields", each child must conform to
    // "mesh::index::field"
    if(n.has_path("fields"))
    {
        if(!verify_object_field(protocol, n, info, "fields"))
        {
            res = false;
        }
        else
        {
            bool field_res = true;
            NodeConstIterator itr = n["fields"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["fields"][chld_name];

                field_res &= field::index::verify(chld, chld_info);
                if(chld.has_child("topology"))
                {
                    field_res &= verify_reference_field(protocol, n, info,
                        chld, chld_info, "topology", "topologies");
                }
                if(chld.has_child("matset"))
                {
                    field_res &= verify_reference_field(protocol, n, info,
                        chld, chld_info, "matset", "matsets");
                }
            }

            log::validation(info["fields"],field_res);
            res &= field_res;
        }
    }

    // optional: "adjsets", each child must conform to
    // "mesh::index::adjsets"
    if(n.has_path("adjsets"))
    {
        if(!verify_object_field(protocol, n, info, "adjsets"))
        {
            res = false;
        }
        else
        {
            bool aset_res = true;
            NodeConstIterator itr = n["adjsets"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["adjsets"][chld_name];

                aset_res &= adjset::index::verify(chld, chld_info);
                aset_res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "topology", "topologies");
            }

            log::validation(info["adjsets"],aset_res);
            res &= aset_res;
        }
    }

    // optional: "nestsets", each child must conform to
    // "mesh::index::nestsets"
    if(n.has_path("nestsets"))
    {
        if(!verify_object_field(protocol, n, info, "nestsets"))
        {
            res = false;
        }
        else
        {
            bool nset_res = true;
            NodeConstIterator itr = n["nestsets"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["nestsets"][chld_name];

                nset_res &= nestset::index::verify(chld, chld_info);
                nset_res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "topology", "topologies");
            }

            log::validation(info["nestsets"],nset_res);
            res &= nset_res;
        }
    }

    log::validation(info, res);

    return res;
}


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

