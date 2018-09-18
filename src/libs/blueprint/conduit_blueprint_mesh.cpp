//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh.cpp
///
//-----------------------------------------------------------------------------

#if defined(CONDUIT_PLATFORM_WINDOWS)
#define NOMINMAX
#undef min
#undef max
#include "Windows.h"
#endif

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mcarray.hpp"
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
        {0, 1, 2, 3}, {0, 1, 5, 4}, {1, 2, 6, 5},
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
// -- begin internal helper functions --
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
                          const std::string &field_name)
{
    Node &field_info = info[field_name];

    bool res = verify_field_exists(protocol, node, info, field_name);
    if(res)
    {
        const Node &field_node = node[field_name];
        res = blueprint::mlarray::verify(field_node,field_info);
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
std::string identify_coords_coordsys(const Node &coords)
{
    Node axes;
    NodeConstIterator itr = coords.children();
    while(itr.has_next())
    {
        itr.next();
        const std::string axis_name = itr.name();

        if(axis_name[0] == 'd')
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
    DataType subtree_dtype(DataType::empty().id(), 0);

    Node info;
    if(!node.dtype().is_object() && !node.dtype().is_object())
    {
        DataType curr_dtype(node.dtype().id(), 1);
        for(index_t t = 0; t < (index_t)default_dtypes.size(); t++)
        {
            DataType default_dtype = default_dtypes[t];
            if((curr_dtype.is_float() && default_dtype.is_float()) ||
                (curr_dtype.is_signed_integer() && default_dtype.is_signed_integer()) ||
                (curr_dtype.is_unsigned_integer() && default_dtype.is_unsigned_integer()) ||
                (curr_dtype.is_string() && default_dtype.is_string()))
            {
                subtree_dtype.set(curr_dtype);
            }
        }
    }
    else
    {
        NodeConstIterator node_it = node.children();
        while(node_it.has_next())
        {
            const Node &child_node = node_it.next();
            DataType child_dtype = find_widest_dtype(child_node, default_dtypes);
            if(subtree_dtype.element_bytes() < child_dtype.element_bytes())
            {
                subtree_dtype.set(child_dtype);
            }
        }
    }

    return !subtree_dtype.is_empty() ? subtree_dtype : default_dtypes.front();
}

//-----------------------------------------------------------------------------
DataType find_widest_dtype(const Node &node,
                           const DataType &default_dtype)
{
    return find_widest_dtype(node, std::vector<DataType>(1, default_dtype));
}

//-----------------------------------------------------------------------------
DataType find_widest_dtype(const std::vector<const Node*> &nodes,
                           const std::vector<DataType> &default_dtypes)
{
    std::vector<DataType> widest_dtypes;
    for(index_t i = 0; i < (index_t)nodes.size(); i++)
    {
        widest_dtypes.push_back(find_widest_dtype(*nodes[i], default_dtypes));
    }

    DataType widest_dtype(widest_dtypes[0]);
    for(index_t t = 1; t < (index_t)widest_dtypes.size(); t++)
    {
        if(widest_dtype.element_bytes() < widest_dtypes[t].element_bytes())
        {
            widest_dtype.set(widest_dtypes[t]);
        }
    }

    return widest_dtype;
}

//-----------------------------------------------------------------------------
DataType find_widest_dtype(const std::vector<const Node*> &nodes,
                           const DataType &default_dtype)
{
    return find_widest_dtype(nodes, std::vector<DataType>(1, default_dtype));
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
void grid_coord_to_id(const std::vector<index_t> &grid_coord,
                      const std::vector<index_t> &grid_dims,
                      index_t &grid_id)
{
    grid_id = 0;
    for(index_t d = 0; d < (index_t)grid_dims.size(); d++)
    {
        index_t doffset = grid_coord[d];
        for(index_t dd = 0; dd < d; dd++)
        {
            doffset *= grid_dims[dd];
        }

        grid_id += doffset;
    }
}

//-----------------------------------------------------------------------------
void grid_id_to_coord(const index_t grid_id,
                      const std::vector<index_t> &grid_dims,
                      std::vector<index_t> &grid_coord)
{
    grid_coord.resize(grid_dims.size());

    index_t dremain = grid_id;
    for(index_t d = (index_t)grid_dims.size(); d-- > 0;)
    {
        index_t dstride = 1;
        for(index_t dd = 0; dd < d; dd++)
        {
            dstride *= grid_dims[dd];
        }

        grid_coord[d] = dremain / dstride;
        dremain = dremain % dstride;
    }
}

// TODO(JRC): Fold the following two functions into the above two functions.

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
            coordset["origin"][csys_axis].value() : 0.0;
        float64 dim_spacing = coordset.has_child("spacing") ?
            coordset["spacing"]["d"+csys_axis].value() : 1.0;
        index_t dim_len = coordset["dims"][logical_axis].value();

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
    index_t dim_lens[3], coords_len = 1;
    for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
    {
        coords_len *= (dim_lens[i] = is_base_rectilinear ?
            coordset["values"][csys_axes[i]].dtype().number_of_elements() :
            coordset["dims"][logical_axes[i]].value());
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
            coordset["origin"][csys_axis].value() : 0.0;
        float64 dim_spacing = coordset.has_child("spacing") ?
            coordset["spacing"]["d"+csys_axis].value() : 1.0;

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
            coordset["dims"][logical_axes[i]].value() :
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
    index_t indices_per_elem = pow(2, csys_axes.size());

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

void
get_unstructured_element(const conduit::Node &topo,
                         const index_t elem_index,
                         std::vector<int64> &elem_indices)
{
    Node topo_mutable;
    topo_mutable.set_external(topo);

    Node &topo_conn = topo_mutable["elements/connectivity"];
    Node &topo_off = topo_mutable["elements/offsets"];
    get_topology_offsets(topo_mutable, topo_off);

    const DataType conn_dtype(topo_conn.dtype().id(), 1);
    const DataType off_dtype(topo_off.dtype().id(), 1);

    // TODO(JRC): This is an expedient method, but lacks robustness
    // for cases when the connectivity array has strides and mixing.
    // This method should be made efficient and robust if possible.
    Node src_node, dst_node;
    src_node.set_external(off_dtype, topo_off.element_ptr(elem_index));
    int64 elem_start = src_node.to_int64();
    src_node.set_external(off_dtype, topo_off.element_ptr(elem_index + 1));
    int64 elem_end = (elem_index != topo_off.dtype().number_of_elements() - 1) ?
        src_node.to_int64() : topo_conn.dtype().number_of_elements();

    int64 elem_size = elem_end - elem_start;
    src_node.set_external(DataType(conn_dtype.id(), elem_size),
        topo_conn.element_ptr(elem_start));
    elem_indices.resize(elem_size);
    dst_node.set_external(DataType::int64(elem_size), &elem_indices[0]);
    src_node.to_int64_array(dst_node);
}

//-------------------------------------------------------------------------
index_t
calculate_shared_coord(const conduit::Node &edge_topo,
                       const index_t edge0,
                       const index_t edge1)
{
    Node edge_offsets; get_topology_offsets(edge_topo, edge_offsets);
    const Node &edge_conn = edge_topo["elements/connectivity"];

    Node misc_data, edge_data;
    int64 edge0_raw[2] = {-1, -1}, edge1_raw[2] = {-1, -1};

    int64 edges[2] = {edge0, edge1};
    int64 *edge_raws[2] = {&edge0_raw[0], &edge1_raw[0]};
    for(index_t e = 0; e < 2; e++)
    {
        misc_data.set_external(DataType(edge_offsets.dtype().id(), 1),
            edge_offsets.element_ptr(edges[e]));
        misc_data.set_external(DataType(edge_conn.dtype().id(), 2),
            const_cast<void*>(edge_conn.element_ptr(misc_data.to_int64())));
        edge_data.set_external(DataType::int64(2), edge_raws[e]);
        misc_data.to_data_type(edge_data.dtype().id(), edge_data);
    }

    index_t shared_index = -1;
    for(index_t edge0_index = 0; edge0_index < 2; edge0_index++)
    {
        for(index_t edge1_index = 0; edge1_index < 2; edge1_index++)
        {
            if(edge0_raw[edge0_index] == edge1_raw[edge1_index])
            {
                shared_index = edge0_raw[edge0_index];
            }
        }
    }

    return shared_index;
}

//-------------------------------------------------------------------------
void
calculate_ordered_edges(const conduit::Node &edge_topo,
                        const std::set<index_t> &elem_edges,
                        std::vector<index_t> &ordered_edges)
{
    // TODO(JRC): This is pretty inefficient and should be replaced with an
    // alterantive solution if possible.
    ordered_edges = std::vector<index_t>(elem_edges.begin(), elem_edges.end());

    for(index_t bei = 0; bei < (index_t)ordered_edges.size() - 1; bei++)
    {
        index_t base_edge_index = ordered_edges[bei];
        for(index_t cei = bei + 1; cei < (index_t)ordered_edges.size(); cei++)
        {
            index_t check_edge_index = ordered_edges[cei];
            index_t shared_coord_index = calculate_shared_coord(edge_topo,
                base_edge_index, check_edge_index);
            if(shared_coord_index >= 0)
            {
                std::swap(ordered_edges[cei], ordered_edges[bei + 1]);
                continue;
            }
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
                                 conduit::Node &cdest,
                                 std::map<index_t, index_t> &topo_to_dest,
                                 std::map<index_t, index_t> &dest_to_topo)
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

    // Discover Data Types //

    DataType int_dtype, float_dtype;
    {
        std::vector<const Node*> src_nodes;
        src_nodes.push_back(&topo);
        src_nodes.push_back(&coordset);
        int_dtype = find_widest_dtype(src_nodes, blueprint::mesh::default_int_dtypes);
        float_dtype = find_widest_dtype(src_nodes, blueprint::mesh::default_float_dtype);
    }

    const Node &topo_conn_const = topo["elements/connectivity"];
    Node topo_conn; topo_conn.set_external(topo_conn_const);
    const DataType conn_dtype(topo_conn.dtype().id(), 1);
    const DataType offset_dtype(topo_offsets.dtype().id(), 1);

    // Allocate Data Templates for Outputs //

    topo_to_dest.clear();
    dest_to_topo.clear();

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
        data_node.set_external(offset_dtype, topo_offsets.element_ptr(ei));
        const index_t eoffset = data_node.to_int64();
        data_node.set_external(conn_dtype, topo_conn.element_ptr(eoffset));
        const index_t elem_num_faces = topo_shape.is_polyhedral() ?
            data_node.to_int64() : 1;

        std::set<index_t> elem_coord_indices;
        for(index_t fi = 0, foffset = eoffset + topo_shape.is_polyhedral();
            fi < elem_num_faces; fi++)
        {
            data_node.set_external(conn_dtype, topo_conn.element_ptr(foffset));
            const index_t face_num_coords = topo_shape.is_poly() ?
                data_node.to_int64() : topo_shape.indices;
            foffset += topo_shape.is_poly();

            for(index_t ci = 0; ci < face_num_coords; ci++)
            {
                data_node.set_external(conn_dtype, topo_conn.element_ptr(foffset + ci));
                elem_coord_indices.insert(data_node.to_int64());
            }
            foffset += face_num_coords;
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
        Node ei_data(DataType::int64(), &ei_value, true);
        data_node.set_external(int_dtype, dest["elements/connectivity"].element_ptr(ei));
        ei_data.to_data_type(int_dtype.id(), data_node);

        for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
        {
            data_node.set_external(float_dtype,
                cdest["values"][csys_axes[ai]].element_ptr(ei));
            Node center_data(DataType::float64(), &ecentroid[ai], true);
            center_data.to_data_type(float_dtype.id(), data_node);
        }

        topo_to_dest[ei] = ei;
        dest_to_topo[ei] = ei;
    }
}

//-------------------------------------------------------------------------
void
extract_unstructured_entities(const conduit::Node &topo,
                              const conduit::Node &coordset,
                              index_t dest_dim,
                              bool make_unique,
                              conduit::Node &dest,
                              std::map< index_t, std::set<index_t> > &topo_to_dest,
                              std::map< index_t, std::set<index_t> > &dest_to_topo)
{
    // NOTE(JRC): This is a stand-in implementation for the set of methods
    // 'mesh::topology::unstructured::generate_{point|edge|face}' that is used
    // because there isn't a good means in Blueprint to create in-mesh mappings.
    const std::vector<std::string> csys_axes = identify_coordset_axes(coordset);

    Node topo_offsets;
    get_topology_offsets(topo, topo_offsets);
    const index_t topo_num_elems = topo_offsets.dtype().number_of_elements();

    const ShapeCascade topo_cascade(topo);
    const ShapeType &topo_shape = topo_cascade.get_shape();
    const ShapeType &dest_shape = topo_cascade.get_shape(dest_dim);

    // Discover Data Types //

    const Node &topo_conn_const = topo["elements/connectivity"];
    Node topo_conn; topo_conn.set_external(topo_conn_const);
    const DataType conn_dtype(topo_conn.dtype().id(), 1);
    const DataType offset_dtype(topo_offsets.dtype().id(), 1);
    const DataType int_dtype = find_widest_dtype(topo, blueprint::mesh::default_int_dtypes);

    // Allocate Data Templates for Outputs //

    topo_to_dest.clear();
    dest_to_topo.clear();

    dest.reset();
    dest["type"].set("unstructured");
    dest["coordset"].set(topo["coordset"].as_string());
    dest["elements/shape"].set(dest_shape.type);

    // NOTE: The absolute value is here just in case the destination shape is a
    // 'poly' type (i.e. 'polygonal' or 'polyhedral').
    const index_t topo_num_entities_approx = std::abs(topo_shape.is_poly() ?
        // explicit formula: (connectivity length) * (indices per dest entity)
        topo_conn.dtype().number_of_elements() * dest_shape.indices :
        // implicit formula: (elements) * (dest entities per element) * (indices per dest entity)
        topo_num_elems * topo_cascade.get_num_embedded(dest_dim) * dest_shape.indices);

    int64 *entity_buffer = new int64[topo_num_entities_approx];
    index_t offset_index = 0, buffer_index = 0;
    std::map< std::set<int64>, int64 > entity_map;

    // Compute Data for Edge Topology //

    Node src_node, dst_node;
    for(index_t elem_index = 0; elem_index < topo_num_elems; elem_index++)
    {
        std::vector< std::vector<int64> > entity_index_bag;
        std::vector< index_t > entity_dim_bag;

        std::vector<int64> elem_indices;
        get_unstructured_element(topo, elem_index, elem_indices);

        entity_index_bag.push_back(elem_indices);
        entity_dim_bag.push_back(topo_shape.dim);
        while(!entity_index_bag.empty())
        {
            std::vector<int64> entity_indices = entity_index_bag.back();
            entity_index_bag.pop_back();
            index_t entity_dim = entity_dim_bag.back();
            entity_dim_bag.pop_back();

            ShapeType entity_shape = topo_cascade.get_shape(entity_dim);

            // base case: add entity to list of entities
            if(entity_dim == dest_dim)
            {
                // NOTE: This code assumes that all entities can be uniquely
                // identified by the list of coordinate indices of which they
                // are comprised. This is certainly true of all implicit topologies
                // and of 2D polygonal topologies, but it may not be always the
                // case for 3D polygonal topologies.
                std::set<int64> entity;
                if(!entity_shape.is_poly())
                {
                    entity = std::set<int64>(entity_indices.begin(), entity_indices.end());
                }
                else
                {
                    const bool is_3d = entity_shape.dim == 3;
                    index_t elem_outer_count =  is_3d ? entity_indices[0] : 1;
                    for(index_t oi = 0, ooff = is_3d; oi < elem_outer_count; oi++)
                    {
                        index_t elem_inner_count = entity_indices[ooff++];
                        for(index_t ii = 0; ii < elem_inner_count; ii++)
                        {
                            index_t ioff = ooff + ii;
                            entity.insert(entity_indices[ioff]);
                        }
                        ooff += elem_inner_count;
                    }
                }

                bool entity_exists = entity_map.find(entity) != entity_map.end();
                index_t entity_offset = offset_index;

                if(!make_unique || !entity_exists)
                {
                    for(index_t ei = 0; ei < (index_t)entity_indices.size(); ei++)
                    {
                        entity_buffer[buffer_index++] = entity_indices[ei];
                    }
                    ++offset_index;
                }
                if(!entity_exists)
                {
                    entity_map[entity] = entity_offset;
                }

                // NOTE(JRC): The ID for each entity is set to be the index
                // of the entity within an offsets array. This is chosen b/c
                // indexing is a challenge when including non-unique entitys.
                index_t entity_id = entity_map.find(entity)->second;
                topo_to_dest[elem_index].insert(entity_id);
                dest_to_topo[entity_id].insert(elem_index);
            }
            // recursive case: add all entities that comprise the current entity
            // to the total list of entities
            else
            {
                ShapeType embed_shape = topo_cascade.get_shape(entity_dim - 1);

                index_t elem_outer_count = entity_shape.is_poly() ?
                    entity_indices[0] : entity_shape.embed_count;
                for(index_t oi = 0, ooff = entity_shape.is_poly();
                    oi < elem_outer_count; oi++)
                {
                    index_t elem_inner_count = entity_shape.is_polyhedral() ?
                        (entity_indices[ooff] + 1) : embed_shape.indices;

                    std::vector<int64> embed_indices;
                    for(index_t ii = 0; ii < elem_inner_count; ii++)
                    {
                        index_t ioff = ooff + (entity_shape.is_poly() ?
                            ii : entity_shape.embedding[oi * elem_inner_count + ii]);
                        embed_indices.push_back(
                            entity_indices[ioff % entity_indices.size()]);
                    }
                    ooff += (
                        entity_shape.is_polyhedral() ? elem_inner_count : (
                        entity_shape.is_polygonal() ? 1 : 0));

                    // TODO(JRC): This is a hack to ensure that the last edge
                    // value for polygonal edge lists is correct.
                    if(entity_shape.is_polygonal() && oi == elem_outer_count - 1)
                    {
                        embed_indices[1] = entity_indices[1];
                    }

                    entity_index_bag.push_back(embed_indices);
                    entity_dim_bag.push_back(embed_shape.dim);
                }
            }
        }
    }

    // Store Entity Index Data //

    Node &dest_conn = dest["elements/connectivity"];
    Node temp_conn(DataType::int64(buffer_index), entity_buffer, true);
    dest_conn.set(DataType(int_dtype.id(), buffer_index));
    temp_conn.to_data_type(int_dtype.id(), dest_conn);

    delete [] entity_buffer;
}

//-------------------------------------------------------------------------
bool
contains_unstructured_entity(const conduit::Node &topo,
                             const conduit::Node &coordset,
                             index_t elem_dim,
                             index_t entity_dim,
                             const std::vector<int64> &elem_indices,
                             const std::vector<int64> &entity_indices)
{
    // TODO(JRC): This is a hack to get around the fact that recursive association
    // isn't supported by 'extract_unstructured_entities' (e.g. subentities all
    // relate to top-level elements, but relationships among subentities aren't
    // specified/established). This should be removed/refactored if possible.
    const ShapeCascade topo_cascade(topo);
    const ShapeType elem_shape = topo_cascade.get_shape(elem_dim);
    const ShapeType entity_shape = topo_cascade.get_shape(entity_dim);

    Node elem_topo;
    elem_topo["type"].set("unstructured");
    elem_topo["coordset"].set(coordset.name());
    elem_topo["elements/shape"].set(elem_shape.type);
    elem_topo["elements/connectivity"].set(elem_indices);

    std::set<int64> elem_index_set(elem_indices.begin(), elem_indices.end());
    std::set<int64> entity_index_set(entity_indices.begin(), entity_indices.end());

    std::vector< std::set<int64> > elem_entities;
    {
        Node elem_entity_topo;
        std::map< index_t, std::set<index_t> > elem_to_entities, entity_to_elems;
        extract_unstructured_entities(elem_topo, coordset, entity_dim, true,
            elem_entity_topo, elem_to_entities, entity_to_elems);

        get_topology_offsets(elem_entity_topo, elem_entity_topo["elements/offsets"]);
        const index_t elem_num_entities = get_topology_length(
            "unstructured", elem_entity_topo);

        for(index_t ei = 0; ei < elem_num_entities; ei++)
        {
            std::vector<int64> entity_indices;
            get_unstructured_element(elem_entity_topo, ei, entity_indices);
            elem_entities.push_back(std::set<index_t>(
                entity_indices.begin(), entity_indices.end()));
        }
    }

    bool found_entity = false;
    for(std::vector< std::set<int64> >::iterator elem_entities_it = elem_entities.begin();
        elem_entities_it != elem_entities.end(); ++elem_entities_it)
    {
        found_entity |= (entity_index_set == *elem_entities_it);
    }

    return found_entity;
}

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
        log::validation(info,res);
    }

    return res;
}


//-----------------------------------------------------------------------------
bool
mesh::verify(const Node &n,
             Node &info)
{
    bool res = true;
    info.reset();

    if(mesh::verify_multi_domain(n, info))
    {
        res = true;
    }
    else
    {
        info.reset();
        res = mesh::verify_single_domain(n, info);
    }

    return res;
}


//-------------------------------------------------------------------------
bool mesh::is_multi_domain(const conduit::Node &n)
{
    Node info;
    return mesh::verify_multi_domain(n, info);
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
                    // spacing names start with "d"
                    axis_name = axis_name.substr(1);
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
            NodeConstIterator mats_itr = matset["volume_fractions"].children();
            while(mats_itr.has_next())
            {
                mats_itr.next();
                idx_matset["materials"][mats_itr.name()];
            }
            
            std::string ms_ref_path = join_path(ref_path, "matsets");
            ms_ref_path = join_path(ms_ref_path, matset_name);
            idx_matset["path"] = ms_ref_path;
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
                log::error(info, protocol, "value child \"" + chld_name + "\" " +
                                           "is not a number array");
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
        // single shape case
        if(topo_elems.has_child("shape"))
        {
            elems_res &= verify_field_exists(protocol, topo_elems, info_elems, "shape") &&
                   mesh::topology::shape::verify(topo_elems["shape"], info_elems["shape"]);
            elems_res &= verify_integer_field(protocol, topo_elems, info_elems, "connectivity");
            // optional: shape topologies can have an "offsets" array to index
            // individual elements; this list must be an integer array
            if(elems_res && topo_elems.has_child("offsets"))
            {
                elems_res &= verify_integer_field(protocol, topo_elems, info_elems, "offsets");
            }
        }
        // shape stream case
        else if(topo_elems.has_child("element_types"))
        {
            // TODO
        }
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
                // optional: shape topologies can have an "offsets" array to index
                // individual elements; this list must be an integer array
                if(chld_res && chld.has_child("offsets"))
                {
                    chld_res &= verify_integer_field(protocol, chld, chld_info, "offsets");
                }

                log::validation(chld_info,chld_res);
                elems_res &= chld_res;
            }
        }
        else
        {
            log::error(info,protocol,"invalid child \"elements\"");
            res = false;
        }

        log::validation(info_elems,elems_res);
        res &= elems_res;
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

    // TODO(JRC): Write a note here about why the topology shape is the same
    // as the embedding shape in <3D (has to do with self-embedding process
    // for 1D/2D polygonal and full embedding for 3D polyhedral).
    const ShapeCascade topo_cascade(topo);
    ShapeType topo_shape(topo_cascade.get_shape());
    ShapeType embed_shape(topo_shape.dim == 3 ?
        topo_cascade.get_shape(topo_shape.dim - 1) : topo_shape);

    const DataType int_dtype = find_widest_dtype(topo, blueprint::mesh::default_int_dtypes);

    // polygonal topology case
    if(topo_shape.is_poly())
    {
        dest.set(topo);
    }
    // nonpolygonal topology case
    else
    {
        const Node &topo_conn_const = topo["elements/connectivity"];
        Node topo_conn; topo_conn.set_external(topo_conn_const);
        const DataType topo_dtype(topo_conn.dtype().id(), 1);
        const index_t topo_indices = topo_conn.dtype().number_of_elements();
        const index_t topo_elems = topo_indices / topo_shape.indices;
        const bool is_topo_3d = topo_shape.dim == 3;

        // NOTE(JRC): In order to make iterations consistent, <3D topologies
        // are self-embedded (which follows polygonal definition) and thus
        // have only one embedding per element.
        topo_shape.embed_count = is_topo_3d ? topo_shape.embed_count : 1;

        Node topo_templ;
        topo_templ.set_external(topo);
        topo_templ.remove("elements");
        dest.set(topo_templ);
        dest["elements/shape"].set(is_topo_3d ? "polyhedral" : "polygonal");

        Node data_node;
        std::vector<int64> poly_conn_data(topo_elems *
            (is_topo_3d + topo_shape.embed_count * (1 + embed_shape.indices)));
        for(index_t ei = 0; ei < topo_elems; ei++)
        {
            index_t data_off = topo_shape.indices * ei;
            index_t poly_off = (is_topo_3d + topo_shape.embed_count *
                (1 + embed_shape.indices)) * ei;

            poly_conn_data[poly_off] = topo_shape.embed_count;
            for(index_t oi = 0; oi < topo_shape.embed_count; oi++)
            {
                index_t outer_off = poly_off + (is_topo_3d +
                    oi * (1 + embed_shape.indices));

                poly_conn_data[outer_off] = embed_shape.indices;
                for(index_t ii = 0; ii < embed_shape.indices; ii++)
                {
                    index_t inner_off = outer_off + 1 + ii;
                    index_t inner_data_off = data_off + (is_topo_3d ?
                        topo_shape.embedding[oi * embed_shape.indices + ii] : ii);

                    data_node.set_external(topo_dtype, topo_conn.element_ptr(inner_data_off));
                    poly_conn_data[inner_off] = data_node.to_int64();
                }
            }
        }

        Node poly_conn;
        poly_conn.set_external(poly_conn_data);
        poly_conn.to_data_type(int_dtype.id(), dest["elements/connectivity"]);
    }
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_centroids(const Node &topo,
                                                 Node &dest,
                                                 Node &cdest)
{
    // TODO(JRC): Revise this function so that it works on every base topology
    // type and then move it to "mesh::topology::{uniform|...}::generate_centroids".
    Node coordset;
    find_reference_node(topo, "coordset", coordset);

    std::map< index_t, index_t > topo_to_dest, dest_to_topo;
    calculate_unstructured_centroids(topo, coordset, dest, cdest,
        topo_to_dest, dest_to_topo);
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_vertices(const Node &topo,
                                                bool make_unique,
                                                Node &dest)
{
    // TODO(JRC): Revise this function so that it works on every base topology
    // type and then move it to "mesh::topology::{uniform|...}::generate_vertices".
    Node coordset;
    find_reference_node(topo, "coordset", coordset);

    std::map< index_t, std::set<index_t> > topo_to_dest, dest_to_topo;
    dest.reset();
    extract_unstructured_entities(topo, coordset, 0, make_unique, dest,
        topo_to_dest, dest_to_topo);
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_edges(const Node &topo,
                                             bool make_unique,
                                             Node &dest)
{
    // TODO(JRC): Revise this function so that it works on every base topology
    // type and then move it to "mesh::topology::{uniform|...}::generate_edges".
    Node coordset;
    find_reference_node(topo, "coordset", coordset);

    std::map< index_t, std::set<index_t> > topo_to_dest, dest_to_topo;
    extract_unstructured_entities(topo, coordset, 1, make_unique, dest,
        topo_to_dest, dest_to_topo);
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_faces(const Node &topo,
                                             bool make_unique,
                                             Node &dest)
{
    // TODO(JRC): Revise this function so that it works on every base topology
    // type and then move it to "mesh::topology::{uniform|...}::generate_faces".
    Node coordset;
    find_reference_node(topo, "coordset", coordset);

    std::map< index_t, std::set<index_t> > topo_to_dest, dest_to_topo;
    extract_unstructured_entities(topo, coordset, 2, make_unique, dest,
        topo_to_dest, dest_to_topo);
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_sides(const Node &topo,
                                             Node &dest,
                                             Node &cdest,
                                             Node &fdest)
{
    // Retrieve Relevent Coordinate/Topology Metadata //

    Node coordset;
    find_reference_node(topo, "coordset", coordset);
    const std::vector<std::string> csys_axes = identify_coordset_axes(coordset);

    // TODO(JRC): Somehow, this code is reordering the original points array,
    // which is causing the results to come out inverted. The index set for
    // the original mesh needs to be preserved in order to this all to work
    // out properly.

    const ShapeCascade topo_cascade(topo);
    const ShapeType topo_shape = topo_cascade.get_shape();
    const ShapeType edge_shape = topo_cascade.get_shape(1);
    const ShapeType side_shape(topo_shape.dim == 3 ? "tet" : "tri");
    if(topo_shape.dim < 2)
    {
        CONDUIT_ERROR("Failed to generate side mesh for input; " <<
            "input tology must be topologically 2D or 3D.");
    }

    Node topo_offsets;
    get_topology_offsets(topo, topo_offsets);
    const index_t topo_num_elems = topo_offsets.dtype().number_of_elements();

    // Extract Derived Coordinate/Topology Data //

    typedef std::map< index_t, std::set<index_t> > ElemEntityMap;
    std::vector<conduit::Node> dim_entity_topos(topo_shape.dim + 1);
    std::vector<ElemEntityMap> dim_elem_to_entities(topo_shape.dim + 1);
    std::vector<ElemEntityMap> dim_entity_to_elems(topo_shape.dim + 1);
    for(index_t di = 0; di <= topo_shape.dim; di++)
    {
        extract_unstructured_entities(topo, coordset, di, true,
            dim_entity_topos[di], dim_elem_to_entities[di], dim_entity_to_elems[di]);

        Node &dim_topo = dim_entity_topos[di];
        get_topology_offsets(dim_topo, dim_topo["elements/offsets"]);
    }

    // TODO(JRC): May need to hack some code together for the following loop
    // in order to properly associate the coordsets with the topologies in a
    // mesh hierarchy (needed for internal consistency maybe).

    typedef std::map< index_t, index_t > ElemCoordMap;
    std::vector<conduit::Node> dim_cent_topos(topo_shape.dim + 1);
    std::vector<conduit::Node> dim_cent_coords(topo_shape.dim + 1);
    std::vector<ElemCoordMap> dim_elem_to_cent(topo_shape.dim + 1);
    std::vector<ElemCoordMap> dim_cent_to_elem(topo_shape.dim + 1);
    for(index_t di = 0; di <= topo_shape.dim; di++)
    {
        // NOTE: No centroids are generate for the edges of the geometry
        // because they aren't included in the final sides topology.
        if(di == edge_shape.dim) { continue; }

        calculate_unstructured_centroids(
            dim_entity_topos[di], coordset,
            dim_cent_topos[di], dim_cent_coords[di],
            dim_elem_to_cent[di], dim_cent_to_elem[di]);
    }

    // Discover Data Types //

    DataType int_dtype, float_dtype;
    {
        std::vector<const Node*> src_nodes;
        src_nodes.push_back(&topo);
        src_nodes.push_back(&coordset);
        int_dtype = find_widest_dtype(src_nodes, blueprint::mesh::default_int_dtypes);
        float_dtype = find_widest_dtype(src_nodes, blueprint::mesh::default_float_dtype);
    }

    const Node &topo_conn_const = topo["elements/connectivity"];
    Node topo_conn; topo_conn.set_external(topo_conn_const);
    Node &edge_conn = dim_entity_topos[edge_shape.dim]["elements/connectivity"];
    const DataType conn_dtype(topo_conn.dtype().id(), 1);
    const DataType offset_dtype(topo_offsets.dtype().id(), 1);
    const DataType edge_dtype(edge_conn.dtype().id(), 2);

    // Allocate Data Templates for Outputs //

    index_t sides_num_coords_acc = 0, sides_num_elems_acc = 0;
    {
        for(index_t di = 0; di <= topo_shape.dim; di++)
        {
            sides_num_coords_acc += dim_cent_topos[di].dtype().is_empty() ? 0 :
                get_topology_length("unstructured", dim_cent_topos[di]);
        }

        // TODO(JRC): Determine if there is any better way to calculate this
        // value (i.e. a means without quite so much computation).
        Node all_edge_topo; ElemEntityMap elem_to_edge, edge_to_elem;
        extract_unstructured_entities(topo, coordset, edge_shape.dim, false,
            all_edge_topo, elem_to_edge, edge_to_elem);
        sides_num_elems_acc = get_topology_length("unstructured", all_edge_topo);
    }
    const index_t sides_num_coords = sides_num_coords_acc;
    const index_t sides_num_elems = sides_num_elems_acc;
    const index_t sides_dim_span = topo_shape.dim - edge_shape.dim;
    const index_t sides_elem_degree = sides_dim_span + 2;

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

    fdest.reset();
    fdest["association"].set("element");
    fdest["topology"].set(dest.name());
    fdest["volume_dependent"].set("false");
    fdest["values"].set(DataType(int_dtype.id(), sides_num_elems));

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

    std::vector<int64> edge_data_raw(2);
    std::vector<int64> dent_data_raw;
    std::vector<int64> side_data_raw(sides_elem_degree);

    Node misc_data;
    Node elem_data(DataType::int64(1), &elem_index, true);
    Node edge_data(DataType::int64(2), &edge_data_raw[0], true);
    Node side_data(DataType::int64(sides_elem_degree), &side_data_raw[0], true);

    Node &dest_conn = dest["elements/connectivity"];
    Node &dest_field = fdest["values"];
    for(; elem_index < (int64)topo_num_elems; elem_index++)
    {
        const std::set<index_t> &elem_edge_ids =
            dim_elem_to_entities[edge_shape.dim][elem_index];
        for(std::set<index_t>::const_iterator edges_it = elem_edge_ids.begin();
            edges_it != elem_edge_ids.end(); ++edges_it)
        {
            index_t edge_index = *edges_it;
            misc_data.set_external(edge_dtype,
                edge_conn.element_ptr(2 * edge_index));
            misc_data.to_data_type(edge_data.dtype().id(), edge_data);

            std::vector< std::vector<index_t> > edge_dent_lists(sides_dim_span);
            std::vector<index_t> edge_dent_lens(sides_dim_span);
            index_t edge_dent_combos = 1;
            for(index_t di = edge_shape.dim + 1; di <= topo_shape.dim; di++)
            {
                index_t li = di - (edge_shape.dim + 1);
                const Node &dim_topo = dim_entity_topos[di];
                std::vector<index_t> &edge_dent_list = edge_dent_lists[li];

                const std::set<index_t> &elem_dent_ids =
                    dim_elem_to_entities[di][elem_index];
                for(std::set<index_t>::const_iterator dents_it = elem_dent_ids.begin();
                    dents_it != elem_dent_ids.end(); ++dents_it)
                {
                    index_t dent_index = *dents_it;
                    get_unstructured_element(dim_topo, dent_index, dent_data_raw);
                    if(contains_unstructured_entity(dim_topo, coordset,
                        di, edge_shape.dim, dent_data_raw, edge_data_raw))
                    {
                        edge_dent_list.push_back(dent_index);
                    }
                }

                edge_dent_lens[li] = edge_dent_list.size();
                edge_dent_combos *= edge_dent_list.size();
            }

            side_data_raw[0] = edge_data_raw[0];
            side_data_raw[1] = edge_data_raw[1];
            for(index_t ci = 0; ci < (index_t)edge_dent_combos; ci++, side_index++)
            {
                std::vector<index_t> dcoords;
                grid_id_to_coord(ci, edge_dent_lens, dcoords);
                for(index_t di = edge_shape.dim + 1; di <= topo_shape.dim; di++)
                {
                    index_t li = di - (edge_shape.dim + 1);
                    index_t dent_index = edge_dent_lists[li][dcoords[li]];
                    side_data_raw[li + 2] = dim_coord_offsets[di] + dent_index;
                }

                misc_data.set_external(DataType(int_dtype.id(), sides_elem_degree),
                    dest_conn.element_ptr(sides_elem_degree * side_index));
                side_data.to_data_type(int_dtype.id(), misc_data);

                misc_data.set_external(int_dtype,
                    dest_field.element_ptr(side_index));
                elem_data.to_data_type(int_dtype.id(), misc_data);
            }
        }
    }
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_corners(const Node &topo,
                                               Node &dest,
                                               Node &cdest,
                                               Node &fdest)
{
    // Retrieve Relevent Coordinate/Topology Metadata //

    Node coordset;
    find_reference_node(topo, "coordset", coordset);
    const std::vector<std::string> csys_axes = identify_coordset_axes(coordset);
    const index_t topo_num_coords = get_coordset_length("explicit", coordset);

    // TODO(JRC): This function needs to be adapted in order to be able to handle
    // 3D topology inputs when it comes to generating "side" elements.
    if(csys_axes.size() != 2)
    {
        CONDUIT_ERROR("Failed to generate side mesh for input; " <<
            csys_axes.size() << "D meshes are not yet supported.");
    }

    Node topo_offsets;
    get_topology_offsets(topo, topo_offsets);
    const index_t topo_num_elems = topo_offsets.dtype().number_of_elements();

    Node cent_coordset, cent_topo;
    std::map< index_t, index_t > elemid_to_centid, centid_to_elemid;
    calculate_unstructured_centroids(topo, coordset, cent_topo, cent_coordset,
        elemid_to_centid, centid_to_elemid);
    blueprint::mesh::topology::unstructured::generate_offsets(cent_topo,
        cent_topo["elements/offsets"]);

    Node edge_topo;
    std::map< index_t, std::set<index_t> > elemid_to_edgeids, edgeid_to_elemids;
    extract_unstructured_entities(topo, coordset, 1, false, edge_topo,
        elemid_to_edgeids, edgeid_to_elemids);
    const index_t topo_num_total_edges = get_topology_length("unstructured", edge_topo);
    extract_unstructured_entities(topo, coordset, 1, true, edge_topo,
        elemid_to_edgeids, edgeid_to_elemids);
    const index_t topo_num_unique_edges = get_topology_length("unstructured", edge_topo);
    blueprint::mesh::topology::unstructured::generate_offsets(edge_topo,
        edge_topo["elements/offsets"]);

    Node ecent_coordset, ecent_topo;
    std::map< index_t, index_t > edgeid_to_ecentid, ecentid_to_edgeid;
    calculate_unstructured_centroids(edge_topo, coordset, ecent_topo, ecent_coordset,
        edgeid_to_ecentid, ecentid_to_edgeid);
    blueprint::mesh::topology::unstructured::generate_offsets(ecent_topo,
        ecent_topo["elements/offsets"]);

    // Discover Data Types //

    DataType int_dtype, float_dtype;
    {
        std::vector<const Node*> src_nodes;
        src_nodes.push_back(&topo);
        src_nodes.push_back(&coordset);
        int_dtype = find_widest_dtype(src_nodes, blueprint::mesh::default_int_dtypes);
        float_dtype = find_widest_dtype(src_nodes, blueprint::mesh::default_float_dtype);
    }

    const Node &topo_conn_const = topo["elements/connectivity"];
    Node topo_conn; topo_conn.set_external(topo_conn_const);
    Node &edge_conn = edge_topo["elements/connectivity"];
    const DataType conn_dtype(topo_conn.dtype().id(), 1);
    const DataType offset_dtype(topo_offsets.dtype().id(), 1);
    const DataType edge_dtype(edge_conn.dtype().id(), 2);

    // Allocate Data Templates for Outputs //

    const index_t corners_num_coords = topo_num_coords + topo_num_unique_edges + topo_num_elems;
    const index_t corners_num_elems = topo_num_total_edges;

    dest.reset();
    dest["type"].set("unstructured");
    dest["coordset"].set(cdest.name());
    dest["elements/shape"].set("quad");
    dest["elements/connectivity"].set(DataType(int_dtype.id(), 4 * corners_num_elems));

    cdest.reset();
    cdest["type"].set("explicit");
    for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
    {
        cdest["values"][csys_axes[ai]].set(DataType(float_dtype.id(), corners_num_coords));
    }

    fdest.reset();
    fdest["association"].set("element");
    fdest["topology"].set(dest.name());
    fdest["volume_dependent"].set("false");
    fdest["values"].set(DataType(int_dtype.id(), corners_num_elems));

    // Populate Data Arrays w/ Calculated Coordinates //

    const Node *cset_nodes[] = {&coordset, &ecent_coordset, &cent_coordset};
    const index_t cset_lengths[] = {topo_num_coords, topo_num_unique_edges, topo_num_elems};
    index_t cset_offsets[] = {-1, -1, -1};
    const index_t cset_count = sizeof(cset_nodes) / sizeof(cset_nodes[0]);

    for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
    {
        Node dst_data;
        Node &dst_axis = cdest["values"][csys_axes[ai]];

        for(index_t ci = 0, coffset = 0; ci < cset_count; ci++)
        {
            const Node &cset_axis = (*cset_nodes[ci])["values"][csys_axes[ai]];
            const index_t cset_length = cset_lengths[ci];
            cset_offsets[ci] = coffset;

            dst_data.set_external(DataType(float_dtype.id(), cset_length),
                dst_axis.element_ptr(coffset));
            cset_axis.to_data_type(float_dtype.id(), dst_data);

            coffset += cset_length;
        }
    }

    // Compute New Elements/Fields for Corner Topology //

    int64 elem_index = 0, corner_index = 0;
    int64 edge_data_raw[2] = {-1, -1}, quad_data_raw[4] = {-1, -1, -1};

    Node misc_data;
    Node elem_data(DataType::int64(1), &elem_index, true);
    Node edge_data(DataType::int64(2), &edge_data_raw[0], true);
    Node quad_data(DataType::int64(3), &quad_data_raw[0], true);

    Node &dest_conn = dest["elements/connectivity"];
    Node &dest_field = fdest["values"];
    for(; elem_index < (int64)topo_num_elems; elem_index++)
    {
        index_t cent_id = elemid_to_centid[elem_index];

        // TODO(JRC): This solution really won't work for 3D; cascading centroids
        // will be required for each dimension (cell, face, edge).
        std::vector<index_t> elem_edge_ids;
        calculate_ordered_edges(edge_topo, elemid_to_edgeids[elem_index], elem_edge_ids);
        for(index_t curr_edge_index = 0;
            curr_edge_index < (index_t)elem_edge_ids.size();
            curr_edge_index++, corner_index++)
        {
            index_t next_edge_index = (curr_edge_index + 1) % elem_edge_ids.size();

            index_t curr_edge_id = elem_edge_ids[curr_edge_index];
            index_t next_edge_id = elem_edge_ids[next_edge_index];

            index_t curr_ecent_id = edgeid_to_ecentid[curr_edge_id];
            index_t next_ecent_id = edgeid_to_ecentid[next_edge_id];

            index_t corner_id = calculate_shared_coord(edge_topo, curr_edge_id, next_edge_id);

            quad_data_raw[0] = cset_offsets[1] + curr_ecent_id;
            quad_data_raw[1] = cset_offsets[0] + corner_id;
            quad_data_raw[2] = cset_offsets[1] + next_ecent_id;
            quad_data_raw[3] = cset_offsets[2] + cent_id;

            misc_data.set_external(DataType(int_dtype.id(), 4),
                dest_conn.element_ptr(4 * corner_index));
            quad_data.to_data_type(int_dtype.id(), misc_data);

            misc_data.set_external(int_dtype,
                dest_field.element_ptr(corner_index));
            elem_data.to_data_type(int_dtype.id(), misc_data);
        }
    }
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_offsets(const Node &topo,
                                               Node &dest)
{
    dest.reset();

    const ShapeType topo_shape(topo);
    const DataType int_dtype = find_widest_dtype(topo, blueprint::mesh::default_int_dtypes);

    const Node &topo_conn = topo["elements/connectivity"];
    const DataType topo_dtype(topo_conn.dtype().id(), 1, 0, 0,
        topo_conn.dtype().element_bytes(), topo_conn.dtype().endianness());

    if(topo_shape.indices > 0)
    {
        const index_t topo_shapes =
            topo_conn.dtype().number_of_elements() / topo_shape.indices;

        Node shape_node(DataType::int64(topo_shapes));
        int64_array shape_array = shape_node.as_int64_array();
        for(index_t s = 0; s < topo_shapes; s++)
        {
            shape_array[s] = s * topo_shape.indices;
        }
        shape_node.to_data_type(int_dtype.id(), dest);
    }
    else if(topo_shape.type == "polygonal")
    {
        std::vector<int64> shape_array;
        index_t s = 0;
        while(s < topo_conn.dtype().number_of_elements())
        {
            const Node index_node(topo_dtype,
                const_cast<void*>(topo_conn.element_ptr(s)), true);
            shape_array.push_back(s);
            s += index_node.to_int64() + 1;
        }

        Node shape_node;
        shape_node.set_external(shape_array);
        shape_node.to_data_type(int_dtype.id(), dest);
    }
    else if(topo_shape.type == "polyhedral")
    {
        std::vector<int64> shape_array;
        index_t s = 0;
        while(s < topo_conn.dtype().number_of_elements())
        {
            const Node index_node(topo_dtype,
                const_cast<void*>(topo_conn.element_ptr(s)), true);
            shape_array.push_back(s);

            s += 1;
            for(index_t f = 0; f < index_node.to_int64(); f++)
            {
                const Node face_node(topo_dtype,
                    const_cast<void*>(topo_conn.element_ptr(s)), true);
                s += face_node.to_int64() + 1;
            }
        }

        Node shape_node;
        shape_node.set_external(shape_array);
        shape_node.to_data_type(int_dtype.id(), dest);
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
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, matset, info, "topology");
    res &= verify_mcarray_field(protocol, matset, info, "volume_fractions");

    log::validation(info, res);

    return res;
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
    res &= verify_object_field(protocol, matset_idx, info, "materials");
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
        log::error(info, protocol, "missing child \"association\" or \"basis\"");
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
    if(!has_topo && !has_matset)
    {
        log::error(info, protocol, "missing child \"topology\" or \"matset\"");
        res = false;
    }
    if(has_topo)
    {
        res &= verify_string_field(protocol, field, info, "topology");
        res &= verify_mlarray_field(protocol, field, info, "values");
    }
    if(has_matset)
    {
        res &= verify_string_field(protocol, field, info, "matset");
        res &= verify_mlarray_field(protocol, field, info, "matset_values");
    }

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
        log::error(info, protocol, "missing child \"association\" or \"basis\"");
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
        log::error(info, protocol, "missing child \"topology\" or \"matset\"");
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
            group_res &= verify_integer_field(protocol, chld, chld_info, "values");

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

