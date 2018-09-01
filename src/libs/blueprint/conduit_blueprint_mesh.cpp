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

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstring>

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

    static const index_t topo_shape_index_count_list[8] = {1, 2,
        3, 4, 4, 8, -1, -1};
    static const std::vector<index_t> topo_shape_index_counts(
        topo_shape_index_count_list, topo_shape_index_count_list +
        sizeof(topo_shape_index_count_list) / sizeof(topo_shape_index_count_list[0]));

    static const index_t topo_shape_face_count_list[8] = {0, 0,
        1, 1, 4, 6, -1, -1};
    static const std::vector<index_t> topo_shape_face_counts(
        topo_shape_face_count_list, topo_shape_face_count_list +
        sizeof(topo_shape_face_count_list) / sizeof(topo_shape_face_count_list[0]));

    static const index_t topo_shape_face_index_count_list[8] = {1, 2,
        3, 4, 3, 4, -1, -1};
    static const std::vector<index_t> topo_shape_face_index_counts(
        topo_shape_face_index_count_list, topo_shape_face_index_count_list +
        sizeof(topo_shape_face_index_count_list) / sizeof(topo_shape_face_index_count_list[0]));

    static const index_t topo_tet_face_arrangements[4][3] = {
        {0, 2, 1}, {0, 1, 3},
        {0, 3, 2}, {1, 2, 3}};
    static const index_t topo_hex_face_arrangements[6][4] = {
        {0, 2, 1, 3}, {0, 1, 4, 5}, {1, 3, 5, 7},
        {0, 4, 2, 6}, {2, 6, 3, 7}, {4, 5, 6, 7}};

    static const index_t* topo_shape_face_arrangement_list[8] = {NULL, NULL,
        NULL, NULL, &topo_tet_face_arrangements[0][0], &topo_hex_face_arrangements[0][0], NULL, NULL};
    static const std::vector<const index_t*> topo_shape_face_arrangements(
        topo_shape_face_arrangement_list, topo_shape_face_arrangement_list +
        sizeof(topo_shape_face_arrangement_list) / sizeof(topo_shape_face_arrangement_list[0]));

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

    static const DataType default_int_type(DataType::INT32_ID, 1);
    static const DataType default_uint_type(DataType::UINT32_ID, 1);
    static const DataType default_float_type(DataType::FLOAT64_ID, 1);

    static const DataType default_int_type_list[2] = {default_int_type, default_uint_type};
    static const std::vector<DataType> default_int_types(default_int_type_list,
        default_int_type_list + sizeof(default_int_type_list) / sizeof(default_int_type_list[0]));

    static const DataType default_number_type_list[3] = {default_float_type,
        default_int_type, default_uint_type};
    static const std::vector<DataType> default_number_types(default_number_type_list,
        default_number_type_list + sizeof(default_number_type_list) /
        sizeof(default_number_type_list[0]));
} } }

//-----------------------------------------------------------------------------
// -- begin internal helper functions --
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
DataType find_widest_type(const Node &node,
                          const std::vector<std::string> &paths,
                          const std::vector<DataType> &defaults)
{
    std::vector<DataType> widest_types;
    for(index_t t = 0; t < (index_t)defaults.size(); t++)
    {
        widest_types.push_back(DataType(defaults[t].id(), 1));
    }

    Node info;
    for(index_t p = 0; p < (index_t)paths.size(); p++)
    {
        const std::string &curr_path = paths[p];
        if(node.has_path(curr_path))
        {
            const Node &curr_node = node.fetch(curr_path);

            DataType curr_type;
            if(conduit::blueprint::mcarray::verify(curr_node, info))
            {
                curr_type.set(find_widest_type(curr_node, curr_node.child_names(), defaults));
            }
            else
            {
                curr_type.set(DataType(curr_node.dtype().id(), 1));
            }

            for(index_t t = 0; t < (index_t)widest_types.size(); t++)
            {
                DataType &widest_type = widest_types[t];

                bool are_types_equivalent =
                    (curr_type.is_float() && widest_type.is_float()) ||
                    (curr_type.is_signed_integer() && widest_type.is_signed_integer()) ||
                    (curr_type.is_unsigned_integer() && widest_type.is_unsigned_integer()) ||
                    (curr_type.is_string() && widest_type.is_string());
                if(are_types_equivalent &&
                    (widest_type.element_bytes() >= curr_type.element_bytes()))
                {
                    widest_type.set(curr_type);
                }
            }
        }
    }

    DataType widest_type(widest_types[0]);
    for(index_t t = 1; t < (index_t)widest_types.size(); t++)
    {
        if(widest_type.element_bytes() < widest_types[t].element_bytes())
        {
            widest_type.set(widest_types[t]);
        }
    }

    return widest_type;
}

//-----------------------------------------------------------------------------
DataType find_widest_type(const Node &node,
                          const std::vector<std::string> &paths,
                          const DataType &default_type)
{
    std::vector<DataType> defaults(1, default_type);
    return find_widest_type(node, paths, defaults);
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

//-------------------------------------------------------------------------
void
convert_coordset_to_rectilinear(const std::string &/*base_type*/,
                                const conduit::Node &coordset,
                                conduit::Node &dest)
{
    // bool is_base_uniform = true;

    dest.reset();
    dest["type"].set("rectilinear");

    DataType float_type;
    {
        std::vector<std::string> float_paths;
        float_paths.push_back("origin");
        float_paths.push_back("spacing");
        float_type.set(find_widest_type(coordset, float_paths,
            blueprint::mesh::default_float_type));
    }

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
        dst_cvals_node.set(DataType(float_type.id(), dim_len));

        Node src_cval_node, dst_cval_node;
        for(index_t d = 0; d < dim_len; d++)
        {
            src_cval_node.set(dim_origin + d * dim_spacing);
            dst_cval_node.set_external(float_type, dst_cvals_node.element_ptr(d));
            src_cval_node.to_data_type(float_type.id(), dst_cval_node);
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

    DataType float_type;
    if(is_base_rectilinear)
    {
        std::vector<std::string> float_paths(1, "values");
        float_type.set(find_widest_type(coordset, float_paths,
            blueprint::mesh::default_float_type));
    }
    else if(is_base_uniform)
    {
        std::vector<std::string> float_paths;
        float_paths.push_back("origin");
        float_paths.push_back("spacing");
        float_type.set(find_widest_type(coordset, float_paths,
            blueprint::mesh::default_float_type));
    }

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
        dst_cvals_node.set(DataType(float_type.id(), coords_len));

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
                    dst_cval_node.set_external(float_type,
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

                    src_cval_node.to_data_type(float_type.id(), dst_cval_node);
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
    DataType int_type;
    {
        std::vector<std::string> int_paths(1, "origin");
        int_type.set(find_widest_type(topo, int_paths,
            blueprint::mesh::default_int_types));
    }

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
        src_dlen_node.to_data_type(int_type.id(), dst_dlen_node);
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
    DataType int_type;
    if(is_base_structured)
    {
        std::vector<std::string> int_paths;
        int_paths.push_back("elements/dims");
        int_paths.push_back("elements/origin");
        int_type.set(find_widest_type(topo, int_paths,
            blueprint::mesh::default_int_types));
    }
    else
    {
        std::vector<std::string> int_paths(1, "origin");
        int_type.set(find_widest_type(topo, int_paths,
            blueprint::mesh::default_int_types));
    }

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
    conn_node.set(DataType(int_type.id(), num_elems * indices_per_elem));

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
            dst_idx_node.set_external(int_type,
                conn_node.element_ptr(e * indices_per_elem + i));
            src_idx_node.to_data_type(int_type.id(), dst_idx_node);
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
            t1.set(int_type, conn_node.element_ptr(p1));
            t2.set(int_type, conn_node.element_ptr(p2));

            t3.set_external(int_type, conn_node.element_ptr(p1));
            t2.to_data_type(int_type.id(), t3);
            t3.set_external(int_type, conn_node.element_ptr(p2));
            t1.to_data_type(int_type.id(), t3);
        }
    }
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
bool
mesh::topology::unstructured::is_manifold(const Node &/*topo*/,
                                          Node &info)
{
    bool res = true;
    info.reset();

    return res;
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::to_polygonal(const Node &topo,
                                           Node &dest)
{
    dest.reset();

    DataType int_type;
    {
        std::vector<std::string> int_paths(1, "elements/connectivity");
        int_type.set(find_widest_type(topo, int_paths,
            blueprint::mesh::default_int_types));
    }

    const std::string topo_shape_type = topo["elements/shape"].as_string();
    index_t topo_shape_indices, topo_shape_faces, topo_shape_findices;
    index_t* topo_shape_farrange;
    for(index_t i = 0; i < (index_t)topo_shapes.size(); i++)
    {
        if(topo_shapes[i] == topo_shape_type)
        {
            topo_shape_indices = topo_shape_index_counts[i];
            topo_shape_faces = topo_shape_face_counts[i];
            topo_shape_findices = topo_shape_face_index_counts[i];
            topo_shape_farrange =
                const_cast<index_t*>(topo_shape_face_arrangements[i]);
        }
    }

    // polygonal topology case
    if(topo_shape_faces < 0)
    {
        dest.set(topo);
    }
    // nonpolygonal topology case
    else
    {
        const Node &topo_conn = topo["elements/connectivity"];
        const DataType topo_dtype(topo_conn.dtype().id(), 1);
        const index_t topo_indices = topo_conn.dtype().number_of_elements();
        const index_t topo_elems = topo_indices / topo_shape_indices;

        // NOTE(JRC): Elements without faces (e.g. lines) are given 1 degenerate
        // face when creating polygons to make subsequent offset math work properly.
        topo_shape_faces = std::max(topo_shape_faces, (index_t)1);
        const bool is_topo_3d = topo_shape_faces > 1;

        Node topo_templ;
        topo_templ.set_external(topo);
        topo_templ.remove("elements");
        dest.set(topo_templ);

        dest["elements/shape"].set(is_topo_3d ? "polyhedral" : "polygonal");

        std::vector<int64> poly_conn_data(topo_elems *
            (is_topo_3d + topo_shape_faces * (1 + topo_shape_findices)));
        for(index_t e = 0; e < topo_elems; e++)
        {
            const index_t ebase = topo_shape_indices * e;
            const index_t epoly = (is_topo_3d + topo_shape_faces *
                (1 + topo_shape_findices)) * e;

            poly_conn_data[epoly] = topo_shape_faces;
            for(index_t f = 0; f < topo_shape_faces; f++)
            {
                const index_t epoly_foff = epoly + (is_topo_3d +
                    f * (1 + topo_shape_findices));
                poly_conn_data[epoly_foff] = topo_shape_findices;
                for(index_t fi = 0; fi < topo_shape_findices; fi++)
                {
                    const index_t ebase_ioff = ebase + (is_topo_3d ?
                        topo_shape_farrange[f * topo_shape_findices + fi] : fi);
                    const Node index_node(topo_dtype,
                        const_cast<void*>(topo_conn.element_ptr(ebase_ioff)), true);
                    poly_conn_data[epoly_foff + 1 + fi] = index_node.to_int64();
                }
            }
        }

        Node poly_conn;
        poly_conn.set_external(poly_conn_data);
        poly_conn.to_data_type(int_type.id(), dest["elements/connectivity"]);
    }
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_corners(const Node &/*topo*/,
                                               Node &dest,
                                               Node &cdest,
                                               Node &fdest)
{
    // TODO(JRC): Implement this function after a general definition for sides
    // is constructed to handle arbitrary polygons/polyhedra.

    dest.reset();
    cdest.reset();
    fdest.reset();
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_sides(const Node &topo,
                                             Node &dest,
                                             Node &cdest,
                                             Node &fdest)
{
    // TODO(JRC): This function needs to be adapted in order to be able to handle
    // 3D topology inputs when it comes to generating "side" elements.

    dest.reset();
    cdest.reset();
    fdest.reset();

    // Retrieve Relevent Coordinate/Topology Metadata //

    Node coordset;
    find_reference_node(topo, "coordset", coordset);

    Node topo_offsets;
    if(topo.has_path("elements/offsets"))
    {
        topo_offsets.set_external(topo["elements/offsets"]);
    }
    else
    {
        mesh::topology::unstructured::generate_offsets(topo, topo_offsets);
    }

    DataType int_type(find_widest_type(topo,
        std::vector<std::string>(1, "elements/connectivity"),
        blueprint::mesh::default_int_types));
    DataType float_type(find_widest_type(coordset["values"],
        identify_coordset_axes(coordset),
        blueprint::mesh::default_float_type));

    const std::vector<std::string> csys_axes = identify_coordset_axes(coordset);
    const size_t topo_num_coords = coordset["values"][csys_axes[0]].dtype().number_of_elements();
    const size_t topo_num_elems = topo_offsets.dtype().number_of_elements();

    const size_t sides_num_coords = topo_num_coords + topo_num_elems;
    size_t sides_num_elems_acc = 0;
    {
        // TODO(JRC): This is incredibly inefficient and should be optimized
        // so that these expensive transform operations don't need to be performed
        // for the sake of uniformity of input.
        Node poly_topo;
        mesh::topology::unstructured::to_polygonal(topo, poly_topo);

        Node &poly_conn = poly_topo["elements/connectivity"];
        Node &poly_offsets = poly_topo["elements/offsets"];
        mesh::topology::unstructured::generate_offsets(poly_topo, poly_offsets);

        DataType offset_dtype(poly_offsets.dtype().id(), 1);
        DataType conn_dtype(poly_conn.dtype().id(), 1);

        Node data_node;
        for(index_t ei = 0; ei < (index_t)topo_num_elems; ei++)
        {
            data_node.set_external(offset_dtype, poly_offsets.element_ptr(ei));
            data_node.set_external(conn_dtype, poly_conn.element_ptr(data_node.to_int64()));
            sides_num_elems_acc += data_node.to_int64();
        }
    }
    const size_t sides_num_elems = sides_num_elems_acc;

    // Allocate Data Templates for Outputs //

    dest["type"].set("unstructured");
    dest["coordset"].set(cdest.name());
    dest["elements/shape"].set("tris");
    dest["elements/connectivity"].set(DataType(int_type.id(), 3 * sides_num_elems));

    cdest["type"].set("explicit");
    for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
    {
        cdest["values"][csys_axes[ai]].set(DataType(float_type.id(), sides_num_coords));
    }

    fdest["association"].set("element");
    fdest["topology"].set(dest.name());
    fdest["volume_dependent"].set("false");
    fdest["values"].set(DataType(int_type.id(), sides_num_elems));

    // Compute New Coordinates by Finding Elements Centers //

    for(index_t ai = 0; ai < (index_t)csys_axes.size(); ai++)
    {
        Node &src_axis = coordset["values"][csys_axes[ai]];
        Node &dst_axis = cdest["values"][csys_axes[ai]];

        Node dst_data(DataType(dst_axis.dtype().id(), topo_num_coords),
            dst_axis.data_ptr(), true);
        src_axis.to_data_type(dst_axis.dtype().id(), dst_data);
    }

    // Compute New Element Topologies //

    // TODO(JRC): Whatever this iteration looks like, it will probably be set up
    // well to fill in the mapping field.
}

//-----------------------------------------------------------------------------
void
mesh::topology::unstructured::generate_offsets(const Node &topo,
                                               Node &dest)
{
    dest.reset();

    DataType int_type;
    {
        std::vector<std::string> int_paths(1, "elements/connectivity");
        int_type.set(find_widest_type(topo, int_paths,
            blueprint::mesh::default_int_types));
    }

    const Node &topo_conn = topo["elements/connectivity"];
    const DataType topo_dtype(topo_conn.dtype().id(), 1, 0, 0,
        topo_conn.dtype().element_bytes(), topo_conn.dtype().endianness());

    const std::string topo_shape_type = topo["elements/shape"].as_string();
    index_t topo_shape_indices;
    for(index_t i = 0; i < (index_t)topo_shapes.size(); i++)
    {
        if(topo_shapes[i] == topo_shape_type)
        {
            topo_shape_indices = topo_shape_index_counts[i];
        }
    }

    if(topo_shape_indices > 0)
    {
        const index_t topo_shapes =
            topo_conn.dtype().number_of_elements() / topo_shape_indices;

        Node shape_node(DataType::int64(topo_shapes));
        int64_array shape_array = shape_node.as_int64_array();
        for(index_t s = 0; s < topo_shapes; s++)
        {
            shape_array[s] = s * topo_shape_indices;
        }
        shape_node.to_data_type(int_type.id(), dest);
    }
    else if(topo_shape_type == "polygonal")
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
        shape_node.to_data_type(int_type.id(), dest);
    }
    else if(topo_shape_type == "polyhedral")
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
        shape_node.to_data_type(int_type.id(), dest);
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

