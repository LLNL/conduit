//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mcarray.hpp"
#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_utils.hpp"

using namespace conduit;
// access verify logging helpers
using namespace conduit::blueprint::utils;

namespace conduit { namespace blueprint { namespace mesh {
    bool verify_single_domain(const conduit::Node &n, conduit::Node &info);
    bool verify_multi_domain(const conduit::Node &n, conduit::Node &info);

    static const std::string association_list[2] = {"vertex", "element"};
    static const std::vector<std::string> associations(association_list,
        association_list + sizeof(association_list) / sizeof(association_list[0]));
} } }

//-----------------------------------------------------------------------------
// -- begin internal helper functions --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool verify_integer_field(const std::string &protocol,
                          const conduit::Node &node,
                          conduit::Node &info,
                          const std::string &field_name)
{
    bool res = true;

    if(!node.has_child(field_name))
    {
        log_error(info, protocol, "missing child \"" + field_name + "\"");
        res = false;
    }
    else if(!node[field_name].dtype().is_integer())
    {
        log_error(info, protocol, "\"" + field_name + "\" is not an integer (array)");
        res = false;
    }

    log_verify_result(info[field_name], res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_number_field(const std::string &protocol,
                         const conduit::Node &node,
                         conduit::Node &info,
                         const std::string &field_name)
{
    bool res = true;

    if(!node.has_child(field_name))
    {
        log_error(info, protocol, "missing child \"" + field_name + "\"");
        res = false;
    }
    else if(!node[field_name].dtype().is_number())
    {
        log_error(info, protocol, "\"" + field_name + "\" is not a number");
        res = false;
    }

    log_verify_result(info[field_name], res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_string_field(const std::string &protocol,
                         const conduit::Node &node,
                         conduit::Node &info,
                         const std::string &field_name)
{
    bool res = true;

    if(!node.has_child(field_name))
    {
        log_error(info, protocol, "missing child \"" + field_name + "\"");
        res = false;
    }
    else if(!node[field_name].dtype().is_string())
    {
        log_error(info, protocol, "\"" + field_name + "\" is not a string");
        res = false;
    }

    log_verify_result(info[field_name], res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_object_field(const std::string &protocol,
                         const conduit::Node &node,
                         conduit::Node &info,
                         const std::string &field_name,
                         const bool allow_empty = false)
{
    bool res = true;

    if(!node.has_child(field_name))
    {
        log_error(info, protocol, "missing child \"" + field_name + "\"");
        res = false;
    }
    else if(!node[field_name].dtype().is_object())
    {
        log_error(info, protocol, "\"" + field_name + "\" is not an object");
        res = false;
    }
    else if(!allow_empty && node[field_name].number_of_children() == 0)
    {
        log_error(info,protocol,"\"" + field_name + "\" has no children");
        res = false;
    }

    log_verify_result(info[field_name], res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_mcarray_field(const std::string &protocol,
                          const conduit::Node &node,
                          conduit::Node &info,
                          const std::string &field_name,
                          const bool allow_single = true)
{
    bool res = true;

    if(!node.has_child(field_name))
    {
        log_error(info, protocol, "missing child \"" + field_name + "\"");
        res = false;
    }
    else if(node[field_name].dtype().is_object())
    {
        if(!blueprint::mcarray::verify(node[field_name],info[field_name]))
        {
             res = false;
        }
        else
        {
            log_info(info, protocol, "\"" + field_name + "\" is a mcarray.");
        }
    }
    else if(allow_single && node[field_name].dtype().is_number())
    {
        log_info(info, protocol, "\"" + field_name + "\" " +
                                   "is a single component numeric array.");
    }
    else
    {
        log_error(info, protocol, "\"" + field_name + "\" is not a " +
                                    (allow_single ? "numeric array or " : "") +
                                    "mcarray.");
        res = false;
    }

    log_verify_result(info[field_name], res);

    return res;
}


//-----------------------------------------------------------------------------
bool verify_enum_field(const std::string &protocol,
                       const conduit::Node &node,
                       conduit::Node &info,
                       const std::string &field_name,
                       const std::vector<std::string> &enum_values)
{
    bool res = verify_string_field(protocol, node, info, field_name);

    if(res)
    {
        const std::string field_value = node[field_name].as_string();

        bool is_field_enum = false;
        for(size_t i=0; i < enum_values.size(); i++)
        {
            is_field_enum |= (field_value == enum_values[i]);
        }

        if(is_field_enum)
        {
            log_info(info, protocol, "\"" + field_name + "\" " +
                                       "has valid value " + field_value);
        }
        else
        {
            log_error(info, protocol, "\"" + field_name + "\" " +
                                        "has invalid value " + field_value);
            res = false;
        }
    }

    log_verify_result(info[field_name], res);

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
            log_error(info, protocol, "reference to non-existent " + ref_path +
                                        " \"" + field_name + "\"");
            res = false;
        }
        else if(info_tree[ref_path][ref_name]["valid"].as_string() != "true")
        {
            log_error(info, protocol, "reference to invalid " + ref_path +
                                        " \"" + field_name + "\"");
            res = false;
        }
    }

    log_verify_result(info[field_name], res);

    return res;
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
    else if(protocol == "domain_adjacency")
    {
        res = domain_adjacency::verify(n,info);
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
    else if(protocol == "domain_adjacency/index")
    {
        res = domain_adjacency::index::verify(n,info);
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

    // Given that not conforming result is likely to trigger an error 
    // state in client code it seems like we should give as much info as
    // possible about what is wrong with the mesh, so we don't early
    // return when an error is found.
    if(!verify_object_field(protocol, n, info, "coordsets"))
    {
        res = false;
    }
    else
    {
        NodeConstIterator itr = n["coordsets"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();

            res &= coordset::verify(chld, info["coordsets"][chld_name]);
        }
    }

    if(!verify_object_field(protocol, n, info, "topologies"))
    {
        res = false;
    }
    else
    {
        NodeConstIterator itr = n["topologies"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            Node &chld_info = info["topologies"][chld_name];

            res &= topology::verify(chld, chld_info);
            res &= verify_reference_field(protocol, n, info,
                chld, chld_info, "coordset", "coordsets");
        }
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
            NodeConstIterator itr = n["matsets"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["matsets"][chld_name];

                res &= matset::verify(chld, chld_info);
                res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "topology", "topologies");
            }
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
            NodeConstIterator itr = n["fields"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["fields"][chld_name];

                res &= field::verify(chld, chld_info);
                if(chld.has_child("topology"))
                {
                    res &= verify_reference_field(protocol, n, info,
                        chld, chld_info, "topology", "topologies");
                }
                if(chld.has_child("matset"))
                {
                    res &= verify_reference_field(protocol, n, info,
                        chld, chld_info, "matset", "matsets");
                }
            }
        }
    }

    // optional: "domain_adjacencies", each child must conform to "mesh::matset"
    if(n.has_path("domain_adjacencies"))
    {
        if(!verify_object_field(protocol, n, info, "domain_adjacencies"))
        {
            res = false;
        }
        else
        {
            NodeConstIterator itr = n["domain_adjacencies"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["domain_adjacencies"][chld_name];

                res &= domain_adjacency::verify(chld, chld_info);
                res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "topology", "topologies");
            }
        }
    }

    // one last pass to make sure if a grid_function was specified by a topo,
    // it is valid
    if (n.has_child("topologies"))
    {
        NodeConstIterator itr = n["topologies"].children();

        while (itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();

            if (chld.has_child("grid_function") &&
                chld["grid_function"].dtype().is_string())
            {
                std::string gf_name = chld["grid_function"].as_string();

                if(!n.has_child("fields") || !n["fields"].has_child(gf_name))
                {
                    std::ostringstream oss;
                    oss << "topology "
                        << "\"" << chld_name << "\" "
                        << " grid_function references a non-existent field "
                        << "\"" << gf_name << "\" ";
                    log_error(info, protocol, oss.str());
                    res = false;
                }
                else if (info["fields"][gf_name]["valid"].as_string() != "true")
                {
                    std::ostringstream oss;
                    oss << "topology "
                        << "\"" << chld_name << "\" "
                        << " grid_function references an invalid field "
                        << "\"" << gf_name << "\" ";
                    log_error(info, protocol, oss.str());
                    res = false;
                }
            }
        }
    }

    log_verify_result(info,res);

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
        log_error(info, protocol, "not an object or a list");
        res = false;
    }
    else
    {
        NodeConstIterator itr = n.children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            std::string chld_name;
            {
                std::ostringstream oss;
                if(n.dtype().is_object())
                {
                    oss << itr.name();
                }
                else
                {
                    oss << itr.index();
                }
                chld_name = oss.str();
            }

            if(!mesh::verify_single_domain(chld, info[chld_name]))
            {
                log_error(info,protocol,
                          "child " + chld_name + " is not a valid mesh");
                res = false;
            }
        }

        log_info(info, protocol, "is a multi domain mesh");
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
bool mesh::to_multi_domain(const conduit::Node &n,
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

    return true;
}


//-----------------------------------------------------------------------------
std::string 
identify_coord_sys_type(const Node &coords)
{
    if(coords.has_child("theta") || coords.has_child("phi"))
    {
        return std::string("spherical");
    }
    else if(coords.has_child("r")) // rz, or r w/o theta, phi
    {
        return std::string("cylindrical");
    }
    else if(coords.has_child("x") ||
            coords.has_child("y") ||
            coords.has_child("z"))
    {
        return std::string("cartesian");
    }
    else
    {
        return std::string("unknown");
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

        idx_coordset["coord_system/type"] = identify_coord_sys_type(idx_coordset["coord_system/axes"]);

        idx_coordset["path"] = ref_path + "/coordsets/" + coordset_name;
    }

    itr = mesh["topologies"].children();
    while(itr.has_next())
    {
        const Node &topo = itr.next();
        std::string topo_name = itr.name();
        Node &idx_topo = index_out["topologies"][topo_name];
        idx_topo["type"] = topo["type"].as_string();
        idx_topo["coordset"] = topo["coordset"].as_string();
        idx_topo["path"] = ref_path + "/topologies/" + topo_name;
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
            idx_matset["number_of_components"] =
                matset["volume_fractions"].number_of_children();
            idx_matset["path"] = ref_path + "/matsets/" + matset_name;
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

            idx_fld["path"] = ref_path + "/fields/" + fld_name;
        }
    }

    if(mesh.has_child("domain_adjacencies"))
    {
        itr = mesh["domain_adjacencies"].children();
        while(itr.has_next())
        {
            const Node &adjacency = itr.next();
            const std::string adj_name = itr.name();
            Node &idx_adjacency = index_out["domain_adjacencies"][adj_name];

            // TODO(JRC): Determine whether or not any information from the
            // "neighbors" and "values" sections need to be included in the index.
            idx_adjacency["association"] = adjacency["association"].as_string();
            idx_adjacency["topology"] = adjacency["topology"].as_string();
            idx_adjacency["path"] = ref_path + "/domain_adjacencies/" + adj_name;
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

    log_verify_result(info, res);

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

    if(origin.has_child("x") && !origin["x"].dtype().is_number())
    {
        log_error(info,protocol,"origin/x is not a number");
        res = false;
    }

    if(origin.has_child("y") && !origin["y"].dtype().is_number())
    {
        log_error(info,protocol,"origin/y is not a number");
        res = false;
    }

    if(origin.has_child("z") && !origin["z"].dtype().is_number())
    {
        log_error(info,protocol,"origin/z is not a number");
        res = false;
    }
    
    if(origin.has_child("r") && !origin["r"].dtype().is_number())
    {
        log_error(info,protocol,"origin/r is not a number");
        res = false;
    }

    if(origin.has_child("theta") && !origin["theta"].dtype().is_number())
    {
        log_error(info,protocol,"origin/theta is not a number");
        res = false;
    }

    if(origin.has_child("phi") && !origin["phi"].dtype().is_number())
    {
        log_error(info,protocol,"origin/phi is not a number");
        res = false;
    }

    log_verify_result(info,res);

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

    if(spacing.has_child("dx") && !spacing["dx"].dtype().is_number())
    {
        log_error(info,protocol,"spacing/dx is not a number");
        res = false;
    }

    if(spacing.has_child("dy") && !spacing["dy"].dtype().is_number())
    {
        log_error(info,protocol,"spacing/dy is not a number");
        res = false;
    }

    if(spacing.has_child("dz") && !spacing["dz"].dtype().is_number())
    {
        log_error(info,protocol,"spacing/dz is not a number");
        res = false;
    }
    
    if(spacing.has_child("dr") && !spacing["dr"].dtype().is_number())
    {
        log_error(info,protocol,"spacing/dr is not a number");
        res = false;
    }

    if(spacing.has_child("dtheta") && !spacing["dtheta"].dtype().is_number())
    {
        log_error(info,protocol,"spacing/dtheta is not a number");
        res = false;
    }

    if(spacing.has_child("dphi") && !spacing["dphi"].dtype().is_number())
    {
        log_error(info,protocol,"spacing/phi is not a number");
        res = false;
    }

    log_verify_result(info,res);

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

    if(!coordset.has_child("dims"))
    {
        log_error(info,protocol, "missing child \"dims\"");
        res = false;
    }
    else
    {
        if(!mesh::logical_dims::verify(coordset["dims"],info["dims"]))
        {
            res= false;
        }
    }

    if(coordset.has_child("origin"))
    {
        log_optional(info,protocol, "has origin");

        if(!mesh::coordset::uniform::origin::verify(coordset["origin"],
                                                    info["origin"]))
        {
            res= false;
        }
    }

    if(coordset.has_child("spacing"))
    {
        log_optional(info,protocol, "has spacing");

        if(!mesh::coordset::uniform::spacing::verify(coordset["spacing"],
                                                     info["spacing"]))
        {
            res= false;
        }
    }

    log_verify_result(info,res);

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

    if(!coordset.has_child("values"))
    {
        log_error(info,protocol, "missing child \"values\"");
        res = false;
    }
    else
    {
        const Node &n_vals = coordset["values"];

        if( ! (n_vals.dtype().is_object() || n_vals.dtype().is_list()) )
        {
            log_error(info,protocol,"Node has no children");
            res = false;
        }
        else
        {
            // each child should be a numeric array
            NodeConstIterator itr = n_vals.children();

            while(itr.has_next())
            {
                const Node &chld = itr.next();
                // make sure every child is a numeric array
                if(!chld.dtype().is_number())
                {
                    std::ostringstream oss;
                    std::string chld_name = itr.name();

                    if(chld_name.size() == 0)
                    {
                        oss << "child [" << itr.index() <<  "]";
                    }
                    else
                    {
                        oss << "child \"" << chld_name << "\"";
                    }

                    oss << " is not a numeric type.";

                    log_error(info,protocol,oss.str());
                    res = false;
                }
            }
        }
    }

    log_verify_result(info,res);

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

    res &= verify_mcarray_field(protocol, coordset, info, "values", false);

    log_verify_result(info,res);

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

    if(!coordset.has_child("type"))
    {
        log_error(info,protocol, "missing child \"type\"");
        res = false;
    }
    else if(!coordset::type::verify(coordset["type"],info["type"]))
    {
        res = false;
    }
    else
    {
        std::string type_name = coordset["type"].as_string();

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

    log_verify_result(info,res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::coordset::type protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::coordset::type::verify(const Node &coordset_type,
                             Node &info)
{
    const std::string protocol = "mesh::coordset::type";
    bool res = true;
    info.reset();

    if(!coordset_type.dtype().is_string())
    {
        log_error(info,protocol,"is not a string");
        res = false;
    }
    else
    {
        std::string coordset_type_str = coordset_type.as_string();

        if(coordset_type_str == "uniform"     ||
           coordset_type_str == "rectilinear" ||
           coordset_type_str == "explicit")
        {
            log_info(info,protocol,"valid type: " + coordset_type_str);
        }
        else
        {
            log_error(info,
                      protocol,
                      "unsupported value:" + coordset_type_str);
            res = false;
        }
    }

    log_verify_result(info,res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::coordset::coord_system protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
check_cart_coord_sys_axis_name(std::string &name)
{
    return ( name == "x" || name == "y" || name == "z");
}

//-----------------------------------------------------------------------------
bool
check_sph_coord_sys_axis_name(std::string &name)
{
    return ( name == "r" || name == "theta" || name == "phi");
}

//-----------------------------------------------------------------------------
bool
check_cyln_coord_sys_axis_name(std::string &name)
{
    return ( name == "r" || name == "z");
}


//-----------------------------------------------------------------------------
bool
mesh::coordset::coord_system::verify(const Node &coord_sys,
                                     Node &info)
{
    const std::string protocol = "mesh::coordset::coord_system";
    bool res = true;
    info.reset();

    std::string coord_sys_str = "unknown";
    if(!coord_sys.has_child("type"))
    {
        log_error(info,protocol,"missing child \"type\"");
        res = false;
    }
    else if(!coord_sys["type"].dtype().is_string())
    {
        log_error(info,protocol,"is not a string");
        res = false;
    }
    else
    {
        coord_sys_str = coord_sys["type"].as_string();

        if( coord_sys_str == "cartesian" ||
            coord_sys_str == "cylindrical" ||
            coord_sys_str == "spherical")
        {
            log_info(info,protocol, "valid type: " + coord_sys_str);
        }
        else
        {
            log_error(info,protocol,"unsupported value:"
                                       + coord_sys_str);
            res = false;
        }
    }

    if(!coord_sys.has_child("axes"))
    {
        log_error(info,protocol,"missing child \"axes\"");
        res = false;
    }
    else
    {
        const Node &axes = coord_sys["axes"];

        if(axes.number_of_children() == 0)
        {
            log_error(info,protocol,"axes has no children");
            res = false;
        }
        else
        {
            NodeConstIterator itr  = axes.children();

            while(itr.has_next())
            {
                bool axis_name_ok = true;

                itr.next();
                std::string axis_name = itr.name();
                if(coord_sys_str == "cartesian")
                {
                    axis_name_ok = check_cart_coord_sys_axis_name(axis_name);
                }
                else if(coord_sys_str == "cylindrical")
                {
                    axis_name_ok = check_cyln_coord_sys_axis_name(axis_name);
                }
                else if(coord_sys_str == "spherical")
                {
                    axis_name_ok = check_sph_coord_sys_axis_name(axis_name);
                }
                else
                {
                    log_error(info,
                              protocol,
                              "cannot verify axis name (" + axis_name + ") "
                              "for coord_sys type " + coord_sys_str);
                    res = false;
                }

                if(!axis_name_ok)
                {
                    log_error(info,
                              protocol,
                              "unsupported " + coord_sys_str  + 
                              " axis name: " + axis_name);
                    res = false;
                }
            }
        }
    }

    log_verify_result(info,res);

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

    // we need the mesh type
    if(!coordset_idx.has_child("type"))
    {
        log_error(info,protocol, "missing child \"type\"");
        res = false;
    }
    else if(!coordset::type::verify(coordset_idx["type"],info["type"]))
    {
        res = false;
    }

    // we need a coord_system
    if(!coordset_idx.has_child("coord_system"))
    {
        log_error(info,protocol,"missing mesh::coordset::index child \"coord_system\"");
        res = false;
    }
    else if(!coordset::coord_system::verify(coordset_idx["coord_system"],
                                            info["coord_system"]))
    {
        res = false;
    }

    // we need a path
    if(!coordset_idx.has_child("path"))
    {
        log_error(info,protocol,"missing child \"path\"");
        res = false;
    }
    else if(!coordset_idx["path"].dtype().is_string())
    {
        log_error(info,protocol,"\"path\" is not a string");
        res = false;
    }
    
    log_verify_result(info,res);

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

    // we need the topo type
    if(!topo.has_child("type"))
    {
        log_error(info,protocol, "missing child \"type\"");
        res = false;
    }
    else if(!mesh::topology::type::verify(topo["type"],info["type"]))
    {
        res = false;
    }
    else
    {
        std::string type_name = topo["type"].as_string();

        if(type_name == "uniform")
        {
            res = mesh::topology::uniform::verify(topo,info);
        }
        else if(type_name == "rectilinear")
        {
            res = mesh::topology::rectilinear::verify(topo,info);
        }
        else if(type_name == "structured")
        {
            res = mesh::topology::structured::verify(topo,info);
        }
        else if(type_name == "unstructured")
        {
            res = mesh::topology::unstructured::verify(topo,info);
        }
    }

    // we need a coordset ref
    if (!topo.has_child("coordset"))
    {
        log_error(info, protocol, "missing child \"coordset\"");
        res = false;
    }
    else if (!topo["coordset"].dtype().is_string())
    {
        log_error(info, protocol, "\"coordset\" is not a string");
        res = false;
    }

    // optional grid_function ref
    if (topo.has_child("grid_function"))
    {
        log_optional(info, protocol, "includes grid_function");
        if (!topo["grid_function"].dtype().is_string())
        {
            log_error(info, protocol, "\"grid_function\" is not a string");
            res = false;
        }
    }


    log_verify_result(info,res);
    
    return res;

}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::uniform protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::uniform::verify(const Node & /*topo*/,
                                Node &info)
{
    info.reset();
    // future: will be used to verify optional info from "elements"
    // child of a uniform topology
    bool res = true;
    log_verify_result(info,res);
    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::rectilinear protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::rectilinear::verify(const Node &/*topo*/,
                                    Node &info)
{
    info.reset();
    // future: will be used to verify optional info from "elements"
    // child of a rectilinear topology
    bool res = true;
    log_verify_result(info,res);
    return res;
}

//-----------------------------------------------------------------------------
bool
mesh::topology::structured::verify(const Node &topo,
                                   Node &info)
{
    const std::string protocol = "mesh::topology::structured";
    bool res = true;
    info.reset();

    if(!topo.has_child("elements"))
    {
        log_error(info,protocol, "missing child \"elements\"");
        res = false;
    }
    else
    {
        const Node &topo_elements = topo["elements"];

        if(!topo_elements.has_child("dims"))
        {
            log_error(info["elements"],protocol, "missing child \"dims\"");
            res = false;
        }
        else
        {
            if(!mesh::logical_dims::verify(topo_elements["dims"],
                                           info["elements/dims"]))
            {
                res = false;
            }
        }
    }

    // FIXME: Add some verification code here for the optional origin in the
    // structured topology.

    log_verify_result(info,res);

    return res;
}



//-----------------------------------------------------------------------------
bool
mesh::topology::unstructured::verify(const Node &topo,
                                   Node &info)
{
    const std::string protocol = "mesh::topology::unstructured";
    bool res = true;
    info.reset();

    if(!topo.has_child("elements"))
    {
        log_error(info,protocol, "missing child \"elements\"");
        res = false;
    }
    else
    {
        const Node &topo_elements = topo["elements"];

        if(topo_elements.has_child("shape"))
        {
            // single shape case
            if(!mesh::topology::shape::verify(topo_elements["shape"],
                                             info["elements/shape"]))
            {
                res = false;
            }

            if(!topo_elements.has_child("connectivity"))
            {
                log_error(info["elements"],
                          protocol, "missing child \"connectivity\"");
                res = false;
            }
            else if(!topo_elements["connectivity"].dtype().is_integer())
            {
                log_error(info["elements"],
                         protocol,
                         "\"connectivity\" is not an integer array");
                res = false;
            }
        }
        else if(topo_elements.has_child("element_types"))
        {
            // TODO stream cases
        }
        else if(topo_elements.number_of_children() != 0)
        {
            bool has_names = topo_elements.dtype().is_object();

            NodeConstIterator itr = topo_elements.children();
            while(itr.has_next())
            {
                const Node &cld  = itr.next();
                std::string name = itr.name();

                if(has_names)
                {
                    info["elements"][name];
                }
                else
                {
                    info["elements"].append();
                }

                Node &cld_info = info["elements"][itr.index()];

                if(!cld.has_child("shape"))
                {
                    log_error(cld_info,protocol, "missing child \"shape\"");
                    res = false;
                }
                else
                {
                    // verify shape
                    if(!mesh::topology::shape::verify(cld["shape"],
                                                      cld_info["shape"]))
                    {
                        res = false;
                    }
                }

                if(!cld.has_child("connectivity"))
                {
                    log_error(cld_info,
                              protocol,
                              "missing child \"connectivity\"");
                    res = false;
                }
                else if(!cld["connectivity"].dtype().is_integer())
                {
                    log_error(cld_info,protocol,"\"connectivity\" "
                              "is not an integer array");
                    res = false;
                }
            }
        }
        else
        {
            log_error(info,protocol,"invalid child \"elements\"");
            res = false;
        }
    }

    log_verify_result(info,res);

    return res;
}



//-----------------------------------------------------------------------------
// blueprint::mesh::shape::index protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::shape::verify(const Node &shape,
                              Node &info)
{
    const std::string protocol = "mesh::topology::shape";
    bool res = true;
    info.reset();

    if(!shape.dtype().is_string())
    {
        log_error(info,protocol,"is not a string");
        res = false;
    }
    else
    {
        std::string shape_str = shape.as_string();

        if(shape_str == "point" ||
           shape_str == "line"  ||
           shape_str == "tri"   ||
           shape_str == "quad"  ||
           shape_str == "tet"   ||
           shape_str == "hex" )
        {
            log_info(info,protocol,"valid type: " + shape_str);
        }
        else
        {
            log_error(info,protocol, "unsupported value:"
                           + shape_str);
            res = false;
        }
    }

    log_verify_result(info,res);

    return res;
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

    // we need the mesh type
    if(!topo_idx.has_child("type"))
    {
        log_error(info,protocol,"missing child \"type\"");
        res = false;
    }
    else if(!topology::type::verify(topo_idx["type"],info["type"]))
    {
        res = false;
    }

    // we need a coordset ref
    if(!topo_idx.has_child("coordset"))
    {
        log_error(info,protocol,"missing child \"coordset\"");
        res = false;
    }
    else if(!topo_idx["coordset"].dtype().is_string())
    {
        log_error(info,protocol,"\"coordset\" is not a string");
        res = false;
    }

    // we need a path
    if(!topo_idx.has_child("path"))
    {
        log_error(info,protocol, "missing child \"path\"");
        res = false;
    }
    else if(!topo_idx["path"].dtype().is_string())
    {
        log_error(info,protocol, "\"path\" is not a string");
        res = false;
    }

    // optional grid_function ref
    if (topo_idx.has_child("grid_function"))
    {
        log_optional(info, protocol, "includes grid_function");
        if (!topo_idx["grid_function"].dtype().is_string())
        {
            log_error(info, protocol, "\"grid_function\" is not a string");
            res = false;
        }
    }

    
    log_verify_result(info,res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::type protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::topology::type::verify(const Node &topo_type,
                             Node &info)
{
    const std::string protocol = "mesh::topology::type";
    bool res = true;
    info.reset();

    if(!topo_type.dtype().is_string())
    {
        log_error(info,protocol,"is not a string");
        res = false;
    }
    else
    {
        std::string topo_type_str = topo_type.as_string();

        if(topo_type_str == "uniform"     ||
           topo_type_str == "rectilinear" ||
           topo_type_str == "structured"  ||
           topo_type_str == "unstructured" )
        {
            log_info(info,protocol, "valid type: " + topo_type_str);
        }
        else
        {
            log_error(info,protocol, "unsupported value:"
                           + topo_type_str);
            res = false;
        }
    }

    log_verify_result(info,res);

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
    res &= verify_mcarray_field(protocol, matset, info, "volume_fractions", false);

    log_verify_result(info, res);

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

    res &= verify_string_field(protocol, matset_idx, info, "topology");
    res &= verify_integer_field(protocol, matset_idx, info, "number_of_components");
    res &= verify_string_field(protocol, matset_idx, info, "path");

    log_verify_result(info, res);

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
        log_error(info, protocol, "missing child \"association\" or \"basis\"");
        res = false;
    }
    if(has_assoc)
    {
        res &= verify_enum_field(protocol, field, info, "association",
                                 mesh::associations);
    }
    if(has_basis)
    {
        res &= verify_string_field(protocol, field, info, "basis");
    }

    bool has_topo = field.has_child("topology");
    bool has_matset = field.has_child("matset");
    if(!has_topo && !has_matset)
    {
        log_error(info, protocol, "missing child \"topology\" or \"matset\"");
        res = false;
    }
    if(has_topo)
    {
        res &= verify_string_field(protocol, field, info, "topology");
        res &= verify_mcarray_field(protocol, field, info, "values", true);
    }
    if(has_matset)
    {
        res &= verify_string_field(protocol, field, info, "matset");
        res &= verify_mcarray_field(protocol, field, info, "matset_values", false);
    }

    log_verify_result(info, res);

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
        log_error(info, protocol, "missing child \"association\" or \"basis\"");
        res = false;
    }
    if(has_assoc)
    {
        res &= verify_enum_field(protocol, field_idx, info, "association",
                                 mesh::associations);
    }
    if(has_basis)
    {
        res &= verify_string_field(protocol, field_idx, info, "basis");
    }

    bool has_topo = field_idx.has_child("topology");
    bool has_matset = field_idx.has_child("matset");
    if(!has_topo && !has_matset)
    {
        log_error(info, protocol, "missing child \"topology\" or \"matset\"");
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

    log_verify_result(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::domain_adjacency protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::domain_adjacency::verify(const Node &adjacency,
                               Node &info)
{
    const std::string protocol = "mesh::domain_adjacency";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, adjacency, info, "topology");
    res &= verify_enum_field(protocol, adjacency, info, "association",
                             mesh::associations);
    res &= verify_integer_field(protocol, adjacency, info, "neighbors");
    res &= verify_integer_field(protocol, adjacency, info, "values");

    log_verify_result(info, res);

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::domain_adjacency::index protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::domain_adjacency::index::verify(const Node &adj_idx,
                                      Node &info)
{
    const std::string protocol = "mesh::domain_adjacency::index";
    bool res = true;
    info.reset();

    res &= verify_string_field(protocol, adj_idx, info, "topology");
    res &= verify_enum_field(protocol, adj_idx, info, "association", 
                             mesh::associations);
    res &= verify_string_field(protocol, adj_idx, info, "path");

    log_verify_result(info, res);

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

    // note "errors" is a list, this allows us to log multiple errors.
    // Given that not conforming result is likely to trigger an error 
    // state in client code it seems like we should give as much info as
    // possible about what is wrong with the mesh, so we don't early
    // return when an error is found.
    if(!verify_object_field(protocol, n, info, "coordsets"))
    {
        res = false;
    }
    else
    {
        NodeConstIterator itr = n["coordsets"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            res &= coordset::index::verify(chld, info["coordsets"][chld_name]);
        }
    }

    if(!verify_object_field(protocol, n, info, "topologies"))
    {
        res = false;
    }
    else
    {
        NodeConstIterator itr = n["topologies"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            Node &chld_info = info["topologies"][chld_name];

            res &= topology::index::verify(chld, chld_info);
            res &= verify_reference_field(protocol, n, info,
                chld, chld_info, "coordset", "coordsets");
        }
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
            NodeConstIterator itr = n["matsets"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["matsets"][chld_name];

                res &= matset::index::verify(chld, chld_info);
                res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "topology", "topologies");
            }
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
            NodeConstIterator itr = n["fields"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["fields"][chld_name];

                res &= field::index::verify(chld, chld_info);
                if(chld.has_child("topology"))
                {
                    res &= verify_reference_field(protocol, n, info,
                        chld, chld_info, "topology", "topologies");
                }
                if(chld.has_child("matset"))
                {
                    res &= verify_reference_field(protocol, n, info,
                        chld, chld_info, "matset", "matsets");
                }
            }
        }
    }

    // optional: "domain_adjacencies", each child must conform to
    // "mesh::index::domain_adjacencies"
    if(n.has_path("domain_adjacencies"))
    {
        if(!verify_object_field(protocol, n, info, "domain_adjacencies"))
        {
            res = false;
        }
        else
        {
            NodeConstIterator itr = n["domain_adjacencies"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                Node &chld_info = info["domain_adjacencies"][chld_name];

                res &= domain_adjacency::index::verify(chld, chld_info);
                res &= verify_reference_field(protocol, n, info,
                    chld, chld_info, "topology", "topologies");
            }
        }
    }

    log_verify_result(info, res);

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

