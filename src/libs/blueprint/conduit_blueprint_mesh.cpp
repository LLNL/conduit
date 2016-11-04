//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
    else if(protocol == "field")
    {
        res = field::verify(n,info);
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
    else if(protocol == "field/index")
    {
        res = field::index::verify(n,info);
    }

    return res;
}

//-----------------------------------------------------------------------------
bool
mesh::verify(const Node &n,
             Node &info)
{
    info.reset();
    bool res = true;

    const std::string proto_name = "mesh";

    // Given that not conforming result is likely to trigger an error 
    // state in client code it seems like we should give as much info as
    // possible about what is wrong with the mesh, so we don't early
    // return when an error is found.
    
    // required: "coordsets", with at least one child
    //  each child must conform to "mesh::coordset"
    if(!n.has_child("coordsets"))
    {
        log_error(info,proto_name,"missing child \"coordsets\"");
        res = false;
    }
    else if(n["coordsets"].number_of_children() == 0)
    {
        log_error(info,proto_name,"\"coordsets\" has no children");
        res = false;
    }
    else
    {
        NodeConstIterator itr = n["coordsets"].children();
    
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();

            if(!coordset::verify(chld,info["coordsets"][chld_name]))
            {
                log_error(info,proto_name,chld_name 
                            + " is not a valid mesh::coordset");
                res = false;
            }
        }
    }
    
    // required: "topologies",  with at least one child
    // each child must conform to "mesh::topology"
    if(!n.has_child("topologies"))
    {
        log_error(info,proto_name,"missing child \"topologies\"");
        res = false;
    }
    else if(n["topologies"].number_of_children() == 0)
    {
        log_error(info,proto_name,"\"topologies\" has no children");
        res = false;
    }
    else
    {
        NodeConstIterator itr = n["topologies"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            const std::string coords_name = chld.has_child("coordset") ?
                chld["coordset"].as_string() : "";

            if(!topology::verify(chld,info["topologies"][chld_name]))
            {
                log_error(info,proto_name,chld_name 
                            + " is not a valid mesh::topology");
                res = false;
            }
            else if(!n.has_child("coordsets") || !n["coordsets"].has_child(coords_name))
            {
                std::ostringstream oss;
                oss << "topology "
                    << "\"" << chld_name  << "\" "
                    << "references a non-existent coordset "
                    << "\"" << coords_name  << "\" ";
                log_error(info,proto_name,oss.str());
                res = false;
            }
            else if(info["coordsets"][coords_name]["valid"].as_string() != "true")
            {
                std::ostringstream oss;
                oss << "topology "
                    << "\"" << chld_name  << "\" "
                    << "references an invalid coordset "
                    << "\"" << coords_name  << "\" ";
                log_error(info,proto_name,oss.str());
                res = false;
            }
        }
    }
    
    // optional: "fields", each child must conform to "mesh::field"
    if(n.has_path("fields"))
    {
        if(n["fields"].number_of_children() == 0)
        {
            log_error(info,proto_name,"\"fields\" has no children");
            res = false;
        }
        else
        {
            NodeConstIterator itr = n["fields"].children();
            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                const std::string topo_name = chld.has_child("topology") ?
                    chld["topology"].as_string() : "";

                if(!field::verify(chld,info["fields"][chld_name]))
                {
                    log_error(info,proto_name,chld_name 
                                + " is not a valid mesh::field");
                    res = false;
                }
                else if(!n.has_child("topologies") || !n["topologies"].has_child(topo_name))
                {
                    std::ostringstream oss;
                    oss << "field "
                        << "\"" << chld_name  << "\" "
                        << "references a non-existent topology "
                        << "\"" << topo_name  << "\" ";
                    log_error(info,proto_name,oss.str());
                    res = false;
                }
                else if(info["topologies"][topo_name]["valid"].as_string() != "true")
                {
                    std::ostringstream oss;
                    oss << "field "
                        << "\"" << chld_name  << "\" "
                        << "references an invalid topology "
                        << "\"" << topo_name << "\" ";
                    log_error(info,proto_name,oss.str());
                    res = false;
                }
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
                    log_error(info, proto_name, oss.str());
                    res = false;
                }
                else if (info["fields"][gf_name]["valid"].as_string() != "true")
                {
                    std::ostringstream oss;
                    oss << "topology "
                        << "\"" << chld_name << "\" "
                        << " grid_function references an invalid field "
                        << "\"" << gf_name << "\" ";
                    log_error(info, proto_name, oss.str());
                    res = false;
                }
            }
        }
    }

    log_verify_result(info,res);

    return res;
}


//-----------------------------------------------------------------------------
std::string 
identify_coord_sys_type(const Node &coords)
{
    if( coords.has_child("theta") || coords.has_child("phi"))
    {
        return std::string("spherical");
    }
    else if( coords.has_child("r") ) // rz, or r w/o theta, phi
    {
        return std::string("cylindrical");
    }
    else if( coords.has_child("x") || 
             coords.has_child("y") || 
             coords.has_child("z") )
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
    
    if(mesh.has_child("fields"))
    {
    
        itr = mesh["fields"].children();
        
        while(itr.has_next())
        {
            const Node &fld = itr.next();
            std::string fld_name = itr.name();
            Node &idx_fld = index_out["fields"][fld_name];
            
            index_t ncomps = 1;
            if(fld["values"].dtype().is_object())
            {
                ncomps = fld["values"].number_of_children();
            }
            
            idx_fld["number_of_components"] = ncomps;
            
            idx_fld["topology"] = fld["topology"].as_string();
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
}

//-----------------------------------------------------------------------------
// blueprint::mesh::logical_dims protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::logical_dims::verify(const Node &dims,
                           Node &info)
{
    info.reset();
    bool res = true;
    const std::string proto_name = "mesh::logical_dims";
    
    if(!dims.has_child("i"))
    {
        log_error(info,proto_name, "missing child \"dims\"");
        res = false;
    }
    else if(!dims["i"].dtype().is_number())
    {
        log_error(info,proto_name,"dims/i is not a number");
        res = false;
    }
    
    if(dims.has_child("j"))
    {
        log_optional(info,proto_name, "dims includes j");
        if(!dims["j"].dtype().is_number())
        {
            log_error(info,proto_name,"dims/j is not a number");
            res = false;
        }
    }
    
    if(dims.has_child("k"))
    {
        log_optional(info,proto_name, "dims includes k");
        if(!dims["k"].dtype().is_number())
        {
            log_error(info,proto_name,"dims/k is not a number");
            res = false;
        }
    }
    
    log_verify_result(info,res);

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
    info.reset();
    bool res = true;

    const std::string proto_name = "mesh::coordset::uniform::origin";

    if(origin.has_child("x") && !origin["x"].dtype().is_number())
    {
        log_error(info,proto_name,"origin/x is not a number");
        res = false;
    }

    if(origin.has_child("y") && !origin["y"].dtype().is_number())
    {
        log_error(info,proto_name,"origin/y is not a number");
        res = false;
    }

    if(origin.has_child("z") && !origin["z"].dtype().is_number())
    {
        log_error(info,proto_name,"origin/z is not a number");
        res = false;
    }
    
    if(origin.has_child("r") && !origin["r"].dtype().is_number())
    {
        log_error(info,proto_name,"origin/r is not a number");
        res = false;
    }

    if(origin.has_child("theta") && !origin["theta"].dtype().is_number())
    {
        log_error(info,proto_name,"origin/theta is not a number");
        res = false;
    }

    if(origin.has_child("phi") && !origin["phi"].dtype().is_number())
    {
        log_error(info,proto_name,"origin/phi is not a number");
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
    info.reset();
    bool res = true;

    const std::string proto_name = "mesh::coordset::uniform::spacing";

    if(spacing.has_child("dx") && !spacing["dx"].dtype().is_number())
    {
        log_error(info,proto_name,"spacing/dx is not a number");
        res = false;
    }

    if(spacing.has_child("dy") && !spacing["dy"].dtype().is_number())
    {
        log_error(info,proto_name,"spacing/dy is not a number");
        res = false;
    }

    if(spacing.has_child("dz") && !spacing["dz"].dtype().is_number())
    {
        log_error(info,proto_name,"spacing/dz is not a number");
        res = false;
    }
    
    if(spacing.has_child("dr") && !spacing["dr"].dtype().is_number())
    {
        log_error(info,proto_name,"spacing/dr is not a number");
        res = false;
    }

    if(spacing.has_child("dtheta") && !spacing["dtheta"].dtype().is_number())
    {
        log_error(info,proto_name,"spacing/dtheta is not a number");
        res = false;
    }

    if(spacing.has_child("dphi") && !spacing["dphi"].dtype().is_number())
    {
        log_error(info,proto_name,"spacing/phi is not a number");
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
    info.reset();
    bool res = true;

    const std::string proto_name = "mesh::coordset::uniform";

    if(!coordset.has_child("dims"))
    {
        log_error(info,proto_name, "missing child \"dims\"");
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
        log_optional(info,proto_name, "has origin");

        if(!mesh::coordset::uniform::origin::verify(coordset["origin"],
                                                    info["origin"]))
        {
            res= false;
        }
    }

    if(coordset.has_child("spacing"))
    {
        log_optional(info,proto_name, "has spacing");

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
    info.reset();
    bool res = true;

    std::string proto_name = "mesh::coordset::rectilinear";


    if(!coordset.has_child("values"))
    {
        log_error(info,proto_name, "missing child \"values\"");
        res = false;
    }
    else
    {
        // values should be a mcarray
        res = blueprint::mcarray::verify(coordset["values"],
                                         info["values"]);
    }

    log_verify_result(info,res);

    return res;
}


//-----------------------------------------------------------------------------
bool
mesh::coordset::_explicit::verify(const Node &coordset,
                                 Node &info)
{
    info.reset();
    bool res = true;
    
    std::string proto_name = "mesh::coordset::explicit";

    if(!coordset.has_child("values"))
    {
        log_error(info,proto_name, "missing child \"values\"");
        res = false;
    }
    else
    {
        // values should be a mcarray
        res = blueprint::mcarray::verify(coordset["values"],
                                         info["values"]);
    }

    log_verify_result(info,res);

    return res;
}

//-----------------------------------------------------------------------------
bool
mesh::coordset::verify(const Node &coordset,
                       Node &info)
{
    info.reset();
    bool res = true;

    std::string proto_name = "mesh::coordset";
    
    if(!coordset.has_child("type"))
    {
        log_error(info,proto_name, "missing child \"type\"");
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
    info.reset();
    bool res = true;
    std::string proto_name = "mesh::coordset::type";


    if(!coordset_type.dtype().is_string())
    {
        log_error(info,proto_name,"is not a string");
        res = false;
    }
    else
    {
        std::string coordset_type_str = coordset_type.as_string();

        if(coordset_type_str == "uniform"     ||
           coordset_type_str == "rectilinear" ||
           coordset_type_str == "explicit")
        {
            log_info(info,proto_name,"valid type: " + coordset_type_str);
        }
        else
        {
            log_error(info,
                      proto_name,
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
    info.reset();
    bool res = true;

    std::string proto_name = "mesh::coordset::coord_system";
    std::string coord_sys_str = "unknown";

    if(!coord_sys.has_child("type"))
    {
        log_error(info,proto_name,"missing child \"type\"");
        res = false;
    }
    else if(!coord_sys["type"].dtype().is_string())
    {
        log_error(info,proto_name,"is not a string");
        res = false;
    }
    else
    {
        coord_sys_str = coord_sys["type"].as_string();

        if( coord_sys_str == "cartesian" ||
            coord_sys_str == "cylindrical" ||
            coord_sys_str == "spherical")
        {
            log_info(info,proto_name, "valid type: " + coord_sys_str);
        }
        else
        {
            log_error(info,proto_name,"unsupported value:"
                                       + coord_sys_str);
            res = false;
        }
    }

    if(!coord_sys.has_child("axes"))
    {
        log_error(info,proto_name,"missing child \"axes\"");
        res = false;
    }
    else
    {
        const Node &axes = coord_sys["axes"];

        if(axes.number_of_children() == 0)
        {
            log_error(info,proto_name,"axes has no children");
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
                              proto_name,
                              "cannot verify axis name (" + axis_name + ") "
                              "for coord_sys type " + coord_sys_str);
                    res = false;
                }

                if(!axis_name_ok)
                {
                    log_error(info,
                              proto_name,
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
    info.reset();
    bool res = true;
    std::string proto_name = "mesh::coordset::index";
    // we need the mesh type
    if(!coordset_idx.has_child("type"))
    {
        log_error(info,proto_name, "missing child \"type\"");
        res = false;
    }
    else if(!coordset::type::verify(coordset_idx["type"],info["type"]))
    {
        res = false;
    }

    // we need a coord_system
    if(!coordset_idx.has_child("coord_system"))
    {
        log_error(info,proto_name,"missing mesh::coordset::index child \"coord_system\"");
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
        log_error(info,proto_name,"missing child \"path\"");
        res = false;
    }
    else if(!coordset_idx["path"].dtype().is_string())
    {
        log_error(info,proto_name,"\"path\" is not a string");
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
    info.reset();
    bool res = true;
    std::string proto_name = "mesh::topology";

    // we need the topo type
    if(!topo.has_child("type"))
    {
        log_error(info,proto_name, "missing child \"type\"");
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
        log_error(info, proto_name, "missing child \"coordset\"");
        res = false;
    }
    else if (!topo["coordset"].dtype().is_string())
    {
        log_error(info, proto_name, "\"coordset\" is not a string");
        res = false;
    }

    // optional grid_function ref
    if (topo.has_child("grid_function"))
    {
        log_optional(info, proto_name, "includes grid_function");
        if (!topo["grid_function"].dtype().is_string())
        {
            log_error(info, proto_name, "\"grid_function\" is not a string");
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
    info.reset();
    bool res = true;
    std::string proto_name = "mesh::topology::structured";

    if(!topo.has_child("elements"))
    {
        log_error(info,proto_name, "missing child \"elements\"");
        res = false;
    }
    else
    {
        const Node &topo_elements = topo["elements"];

        if(!topo_elements.has_child("dims"))
        {
            log_error(info["elements"],proto_name, "missing child \"dims\"");
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
    info.reset();
    bool res = true;
    std::string proto_name = "mesh::topology::unstructured";

    if(!topo.has_child("elements"))
    {
        log_error(info,proto_name, "missing child \"elements\"");
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
                          proto_name, "missing child \"connectivity\"");
                res = false;
            }
            else if(!topo_elements["connectivity"].dtype().is_integer())
            {
                log_error(info["elements"],
                         proto_name,
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
                    log_error(cld_info,proto_name, "missing child \"shape\"");
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
                              proto_name,
                              "missing child \"connectivity\"");
                    res = false;
                }
                else if(!cld["connectivity"].dtype().is_integer())
                {
                    log_error(cld_info,proto_name,"\"connectivity\" "
                              "is not an integer array");
                    res = false;
                }
            }
        }
        else
        {
            log_error(info,proto_name,"invalid child \"elements\"");
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
    info.reset();
    bool res = true;
    std::string proto_name = "mesh::topology::shape";

    if(!shape.dtype().is_string())
    {
        log_error(info,proto_name,"is not a string");
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
            log_info(info,proto_name,"valid type: " + shape_str);
        }
        else
        {
            log_error(info,proto_name, "unsupported value:"
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
    info.reset();
    bool res = true;
    std::string proto_name = "mesh::topology::index";

    // we need the mesh type
    if(!topo_idx.has_child("type"))
    {
        log_error(info,proto_name,"missing child \"type\"");
        res = false;
    }
    else if(!topology::type::verify(topo_idx["type"],info["type"]))
    {
        res = false;
    }

    // we need a coordset ref
    if(!topo_idx.has_child("coordset"))
    {
        log_error(info,proto_name,"missing child \"coordset\"");
        res = false;
    }
    else if(!topo_idx["coordset"].dtype().is_string())
    {
        log_error(info,proto_name,"\"coordset\" is not a string");
        res = false;
    }

    // we need a path
    if(!topo_idx.has_child("path"))
    {
        log_error(info,proto_name, "missing child \"path\"");
        res = false;
    }
    else if(!topo_idx["path"].dtype().is_string())
    {
        log_error(info,proto_name, "\"path\" is not a string");
        res = false;
    }

    // optional grid_function ref
    if (topo_idx.has_child("grid_function"))
    {
        log_optional(info, proto_name, "includes grid_function");
        if (!topo_idx["grid_function"].dtype().is_string())
        {
            log_error(info, proto_name, "\"grid_function\" is not a string");
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
    info.reset();
    bool res = true;
    
    std::string proto_name = "mesh::topology::type";

    if(!topo_type.dtype().is_string())
    {
        log_error(info,proto_name,"is not a string");
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
            log_info(info,proto_name, "valid type: " + topo_type_str);
        }
        else
        {
            log_error(info,proto_name, "unsupported value:"
                           + topo_type_str);
            res = false;
        }
    }

    log_verify_result(info,res);

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
    info.reset();
    bool res = true;

    std::string proto_name = "mesh::field";
        
    // we need a topology
    if(!field.has_child("topology"))
    {
        log_error(info, proto_name,"missing child \"topology\"");
        res = false;
    }
    else if(!field["topology"].dtype().is_string())
    {
        log_error(info, proto_name, "\"topology\" is not a string");
        res = false;
    }


    if(!field.has_child("values"))
    {
        log_error(info, proto_name, "missing child \"values\"");
        res = false;
    }
    else if(field["values"].dtype().is_object())
    {
        if(!blueprint::mcarray::verify(field["values"],info["values"]))
        {
             res = false;
        }
        else
        {
            log_info(info,proto_name,"is a mcarray.");
        }
    }
    else if(field["values"].dtype().is_number())
    {
        log_info(info,proto_name,"is a single component numeric array.");
    }
    else
    {
        log_error(info, proto_name, "\"values\" is not a "
                        " numeric array or mcarray.");
        res = false;
    }


    // make sure we have either a basis or assoc entry
    bool has_assoc = field.has_child("association");
    bool has_basis = field.has_child("basis");
    
    if( ! (has_assoc || has_basis))
    {
        log_error(info,proto_name, "missing child "
                       "\"association\" or \"basis\"");
        res = false;
    }
    else if(has_assoc)
    {
        if(!field::association::verify(field["association"],
                                       info["association"]))
        {
            res = false;
        }
    }
    else if(has_basis)
    {
        if(!field::basis::verify(field["basis"],info["basis"]))
        {
            res = false;
        }
    }

    log_verify_result(info,res);

    return res;
}


//-----------------------------------------------------------------------------
// blueprint::mesh::field::association protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
mesh::field::association::verify(const Node &assoc,
                                 Node &info)
{
    info.reset();
    bool res = true;

    std::string proto_name = "mesh::field::association";

    if(!assoc.dtype().is_string())
    {
        log_error(info,proto_name,"is not a string");
        res = false;
    }
    else
    {
        std::string assoc_str = assoc.as_string();

        if(assoc_str == "vertex" ||
           assoc_str == "element")
        {
            log_info(info,proto_name, "association: " + assoc_str );
        }
        else
        {
            log_error(info,proto_name, "unsupported value:"
                           + assoc_str);
            res = false;
        }
    }

    log_verify_result(info,res);

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
    info.reset();
    bool res = true;

    std::string proto_name = "mesh::field::basis";

    if(!basis.dtype().is_string())
    {
        log_error(info,proto_name,"is not a string");
        res = false;
    }
    else
    {
        log_info(info,proto_name,"basis: " + basis.as_string());
    }

    log_verify_result(info,res);

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
    info.reset();
    bool res = true;
    std::string proto_name = "mesh::field::index";

    // we need a topology
    if(!field_idx.has_child("topology"))
    {
        log_error(info,proto_name,"missing child \"topology\"");
        res = false;
    }
    else if(!field_idx["topology"].dtype().is_string())
    {
        log_error(info,proto_name,"\"topology\" is not a string");
        res = false;
    }

    // we need number_of_components
    if(!field_idx.has_child("number_of_components"))
    {
        log_error(info,proto_name,"missing child "
                       "\"number_of_components\"");
        res = false;
    }
    else if(!field_idx["number_of_components"].dtype().is_integer())
    {
        log_error(info,proto_name,"\"number_of_components\" "
                       "is not an integer");
        res = false;
    }

    // we need a path
    if(!field_idx.has_child("path"))
    {
        log_error(info,proto_name,"missing child \"path\"");
        res = false;
    }
    else if(!field_idx["path"].dtype().is_string())
    {
        log_error(info,proto_name,"\"path\" is not a string");
        res = false;
    }
    
    bool has_assoc = field_idx.has_child("association");
    bool has_basis = field_idx.has_child("basis");
    
    // make sure we have either a basis or assoc entry
    if( ! (has_assoc || has_basis))
    {
        log_error(info,proto_name,"missing child "
                       "\"association\" or \"basis\"");
        res = false;
    }
    else if(has_assoc)
    {
        if(!field::association::verify(field_idx["association"],
                                       info["association"]))
        {
            res = false;
        }
    }
    else if(has_basis)
    {
        if(!field::basis::verify(field_idx["basis"],info["basis"]))
        {
            res = false;
        }
    }

    log_verify_result(info,res);

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
    info.reset();
    // the mesh blueprint index provides metadata about a valid 
    // mesh blueprint conf
    //
    
    bool res = true;
    std::string proto_name = "mesh::index";

    // note "errors" is a list, this allows us to log multiple errors.
    // Given that not conforming result is likely to trigger an error 
    // state in client code it seems like we should give as much info as
    // possible about what is wrong with the mesh, so we don't early
    // return when an error is found.
    
    // required: "coordsets", with at least one child
    //  each child must conform to "mesh::coordset"
    if(!n.has_child("coordsets"))
    {
        log_error(info,proto_name,"missing child \"coordsets\"");
        res = false;
    }
    else if(n["coordsets"].number_of_children() == 0)
    {
        log_error(info,proto_name,"\"coordsets\" has no children");
        res = false;
    }
    else
    {
        NodeConstIterator itr = n["coordsets"].children();
    
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();

            if(!coordset::index::verify(chld,info["coordsets"][chld_name]))
            {
                log_error(info,proto_name,chld_name 
                                + " is not a valid mesh::coordset::index");
                res = false;
            }
        }
    }
    
    // required: "topologies",  with at least one child
    // each child must conform to "mesh::topology"
    if(!n.has_child("topologies"))
    {
        log_error(info,proto_name,"missing child \"topologies\"");
        res = false;
    }
    else if(n["topologies"].number_of_children() == 0)
    {
        log_error(info,proto_name,"\"topologies\" has no children");
        res = false;
    }
    else
    {
        NodeConstIterator itr = n["topologies"].children();
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.name();
            const std::string coords_name = chld.has_child("coordset") ?
                chld["coordset"].as_string() : "";

            if(!topology::index::verify(chld,info["topologies"][chld_name]))
            {
                log_error(info,proto_name,chld_name 
                                + " is not a valid mesh::topology::index");
                res = false;
            }
            else if(!n.has_child("coordsets") || !n["coordsets"].has_child(coords_name))
            {
                std::ostringstream oss;
                oss << "topology index entry "
                    << "\"" << chld_name  << "\" "
                    << "references a non-existent coordset index entry"
                    << "\"" << coords_name  << "\" ";
                log_error(info,proto_name,oss.str());
                res = false;
            }
            else if(info["coordsets"][coords_name]["valid"].as_string() != "true")
            {
                std::ostringstream oss;
                oss << "topology index entry "
                    << "\"" << chld_name  << "\" "
                    << "references an invalid coordset index entry "
                    << "\"" << coords_name  << "\" ";
                log_error(info,proto_name,oss.str());
                res = false;
            }
        }
    }
    
    // optional: "fields", each child must conform to "mesh::field"
    if(n.has_path("fields"))
    {
        if(n["fields"].number_of_children() == 0)
        {
            log_error(info,proto_name,"\"fields\" has no children");
            res = false;
        }
        else
        {
            NodeConstIterator itr = n["fields"].children();

            while(itr.has_next())
            {
                const Node &chld = itr.next();
                const std::string chld_name = itr.name();
                const std::string topo_name = chld.has_child("topology") ?
                    chld["topology"].as_string() : "";

                if(!field::index::verify(chld,info["fields"][chld_name]))
                {
                    log_error(info,proto_name,chld_name 
                                    + " is not a valid mesh::field::index");
                    res = false;
                }
                else if(!n.has_child("topologies") || !n["topologies"].has_child(topo_name))
                {
                    std::ostringstream oss;
                    oss << "field index entry "
                        << "\"" << chld_name  << "\" "
                        << "references a non-existent topology index entry"
                        << "\"" << topo_name  << "\" ";
                    log_error(info,proto_name,oss.str());
                    res = false;
                }
                else if(info["topologies"][topo_name]["valid"].as_string() != "true")
                {
                    std::ostringstream oss;
                    oss << "field index entry "
                        << "\"" << chld_name  << "\" "
                        << "references an invalid topology index entry"
                        << "\"" << topo_name << "\" ";
                    log_error(info,proto_name,oss.str());
                    res = false;
                }
            }
        }
    }
    
    log_verify_result(info,res);

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

