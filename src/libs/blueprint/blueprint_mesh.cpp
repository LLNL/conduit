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
/// file: blueprint_mesh.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <string.h>
#include <math.h>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "blueprint_mcarray.hpp"
#include "blueprint_mesh.hpp"

using namespace conduit;


//-----------------------------------------------------------------------------
void
log_info(Node &info, const std::string &msg)
{
    info["info"].append().set(msg);
}

//-----------------------------------------------------------------------------
void
log_optional(Node &info, const std::string &msg)
{
    info["optional"].append().set(msg);
}

//-----------------------------------------------------------------------------
void
log_error(Node &info, const std::string &msg)
{
    info["errors"].append().set(msg);
}


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
// -- begin conduit::blueprint::mesh --
//-----------------------------------------------------------------------------

namespace mesh
{

//-----------------------------------------------------------------------------
bool
verify(const std::string &protocol,
       const Node &n)
{
    Node info;
    return verify(protocol,n,info);
}

//-----------------------------------------------------------------------------
bool
verify(const std::string &protocol,
       const Node &n,
       Node &info)
{
    bool res = false;

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
verify(const Node &n)
{
    Node info;
    return verify(n,info);
}

//-----------------------------------------------------------------------------
bool
verify(const Node &n,
       Node &info)
{
    bool res = true;

    // note "errors" is a list, this allows us to log multiple errors.
    // Given that not conforming result is likely to trigger an error 
    // state in client code it seems like we should give as much info as
    // possible about what is wrong with the mesh, so we don't early
    // return when an error is found.
    
    // required: "coordsets", with at least one child
    //  each child must conform to "mesh::coordset"
    if(!n.has_child("coordsets"))
    {
        info["errors"].append().set("missing child node \"coordsets\"");
        res = false;
    }
    else if(n["coordsets"].number_of_children() == 0)
    {
        info["errors"].append().set("\"coordsets\" has no children");
        res = false;
    }
    else
    {
        NodeConstIterator itr = n["coordsets"].children();
    
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.path();

            if(!coordset::verify(chld,info["coordsets/" + chld_name]))
            {
                info["errors"].append().set(chld_name 
                                + " is not a valid blueprint::mesh::coordset");
                res = false;
            }
        }
    }
    
    // required: "topologies",  with at least one child
    // each child must conform to "mesh::topology"
    if(!n.has_child("topologies"))
    {
        info["errors"].append().set("missing child node \"topologies\"");
        res = false;
    }
    else if(n["topologies"].number_of_children() == 0)
    {
        info["errors"].append().set("\"topologies\" has no children");
        res = false;
    }
    else
    {
    
        NodeConstIterator itr = n["topologies"].children();
        
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.path();

            if(!topology::verify(chld,info["topologies"][chld_name]))
            {
                info["errors"].append().set(chld_name 
                                + " is not a valid blueprint::mesh::topology");
                res = false;
            }

            // make sure the topology references a valid coordset
            std::string coords_name = chld["coordset"].as_string();
            
            if(!n["coordsets"].has_child(coords_name))
            {
                std::ostringstream oss;
                oss << "topology "
                    << "\"" << chld_name  << "\" "
                    << "references a non-existent coordset "
                    << "\"" << coords_name  << "\" ";
                info["errors"].append().set(oss.str());
                res = false;
            }
            else if(info["coordsets"][coords_name]["valid"].as_string() != "true")
            {
                std::ostringstream oss;
                oss << "topology "
                    << "\"" << chld_name  << "\" "
                    << "references a invalid coordset "
                    << "\"" << coords_name  << "\" ";
                info["errors"].append().set(oss.str());
                res = false;
            }
        }
    }
    
    // optional: "fields", each child must conform to "mesh::field"
    if(n.has_path("fields"))
    {
        NodeConstIterator itr = n["fields"].children();
    
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.path();

            if(!field::verify(chld,info["fields/" + chld_name]))
            {
                info["errors"].append().set(chld_name 
                                    + " is not a valid blueprint::mesh::field");
                res = false;
            }

            // make sure the field references a valid topology
            std::string topo_name = chld["topology"].as_string();
        
            if(!n["topologies"].has_child(topo_name))
            {
                std::ostringstream oss;
                oss << "field "
                    << "\"" << chld_name  << "\" "
                    << "references a non-existent topology "
                    << "\"" << topo_name  << "\" ";
                info["errors"].append().set(oss.str());
                res = false;
            }
            else if(info["topologies"][topo_name]["valid"].as_string() != "true")
            {
                std::ostringstream oss;
                oss << "field "
                    << "\"" << chld_name  << "\" "
                    << "references a invalid topology "
                    << "\"" << topo_name << "\" ";
                info["errors"].append().set(oss.str());
                res = false;
            }
        }
    }
    
    if(res)
    {
        info["valid"] = "true";
    }
    else
    {
        info["valid"] = "false";
    }

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::coordset protocol interface
//-----------------------------------------------------------------------------
namespace coordset
{
//-----------------------------------------------------------------------------
bool
verify(const Node &coordset)
{
    Node info;
    return verify(coordset,info);
}

//-----------------------------------------------------------------------------
bool
verify_coordset_uniform(const Node &coordset,
                        Node &info)
{
    bool res = true;

    if(!coordset.has_child("dims"))
    {
        log_error(info,"missing uniform coordset child \"value\"");
        res = false;
    }
    else // dims needs at least one child ("i") 
    {
        const Node &dims = coordset["dims"];
        
        if(!dims.has_child("i"))
        {
            log_error(info,"missing coordset child \"dims\"");
            res = false;
        }
        else if(!dims["i"].dtype().is_number())
        {
            log_error(info,"uniform dims/i is not a number");
            res = false;
        }
        
        if(dims.has_child("j"))
        {
            log_optional(info,"uniform dims includes j");
            if(!dims["j"].dtype().is_number())
            {
                log_error(info,"uniform dims/j is not a number");
                res = false;
            }
        }
        
        if(dims.has_child("k"))
        {
            log_optional(info,"uniform dims includes k");
            if(!dims["k"].dtype().is_number())
            {
                log_error(info,"uniform dims/k is not a number");
                res = false;
            }
        }
    }
    
    if(!coordset.has_child("origin"))
    {
        log_optional(info, "uniform coordset has origin");

        const Node &origin = coordset["origin"];

        if(origin.has_child("x"))
        {
            if(!origin["x"].dtype().is_number())
            {
                log_error(info,"uniform origin/x is not a number");
                res = false;
            }
        }

        if(origin.has_child("y"))
        {
            if(!origin["y"].dtype().is_number())
            {
                log_error(info,"uniform origin/y is not a number");
                res = false;
            }
        }

        if(origin.has_child("z"))
        {
            if(!origin["z"].dtype().is_number())
            {
                log_error(info, "uniform origin/z is not a number");
                res = false;
            }
        }
    }

    if(!coordset.has_child("spacing"))
    {
        log_optional(info, "uniform coordset has spacing");
        
        const Node &spacing = coordset["spacing"];

        if(spacing.has_child("dx"))
        {

            if(!spacing["dx"].dtype().is_number())
            {
                log_error(info, "uniform spacing/dx is not a number");
                res = false;
            }
        }

        if(spacing.has_child("dy"))
        {

            if(!spacing["dy"].dtype().is_number())
            {
                log_error(info, "uniform spacing/dy is not a number");
                res = false;
            }
        }

        if(spacing.has_child("dz"))
        {

            if(!spacing["dz"].dtype().is_number())
            {
                log_error(info,"uniform spacing/dz is not a number");
                res = false;
            }
        }
    }

    return res;
}

//-----------------------------------------------------------------------------
bool
verify_coordset_rectilinear(const Node &coordset,
                            Node &info)
{
    bool res = true;

    if(!coordset.has_child("values"))
    {
        log_error(info, "missing rectilinear coordset child \"value\"");
        res = false;
    }
    else
    {
        // values should be a mcarray
        res = blueprint::mcarray::verify(coordset["values"],
                                         info["values"]);
    }

    return res;
}


//-----------------------------------------------------------------------------
bool
verify_coordset_explicit(const Node &coordset,
                         Node &info)
{
    bool res = true;

    if(!coordset.has_child("values"))
    {
        log_error(info,"missing explicit coordset child \"value\"");
        res = false;
    }
    else
    {
        // values should be a mcarray
        res = blueprint::mcarray::verify(coordset["values"],
                                         info["values"]);
    }

    return res;
}

//-----------------------------------------------------------------------------
bool
verify(const Node &coordset,
       Node &info)
{
    bool res = true;

    if(!coordset.has_child("type"))
    {
        log_error(info,"missing coordset child \"type\"");
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
            res = verify_coordset_uniform(coordset,info);
        }
        else if(type_name == "rectilinear")
        {
            res = verify_coordset_rectilinear(coordset,info);
        }
        else if(type_name == "explicit")
        {
            res = verify_coordset_explicit(coordset,info);
        }
    }

    if(res)
    {
        info["valid"] = "true";
    }
    else
    {
        info["valid"] = "false";
    }

    return res;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::coordset::type protocol interface
//-----------------------------------------------------------------------------
namespace type
{
//-----------------------------------------------------------------------------
bool
verify(const Node &n)
{
    Node info;
    return verify(n,info);
}

//-----------------------------------------------------------------------------
bool
verify(const Node &coordset_type,
       Node &info)
{    
    bool res = true;

    if(!coordset_type.dtype().is_string())
    {
        log_error(info,"mesh::coordset::type expected string");
        res = false;
    }
    else
    {
        std::string coordset_type_str = coordset_type.as_string();

        if(coordset_type_str == "uniform"     ||
           coordset_type_str == "rectilinear" ||
           coordset_type_str == "explicit")
        {
            log_info(info,"type: " + coordset_type_str);
        }
        else
        {
            log_error(info,"mesh::coordset::type unsupported value:"
                           + coordset_type_str);
            res = false;
        }
    }

    if(res)
    {
        info["valid"] = "true";
    }
    else
    {
        info["valid"] = "false";
    }

    return res;
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::coordset::type --
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// blueprint::mesh::coordset::coord_system protocol interface
//-----------------------------------------------------------------------------
namespace coord_system
{
//-----------------------------------------------------------------------------
bool
verify(const Node &n)
{
    Node info;
    return verify(n,info);
}

//-----------------------------------------------------------------------------
bool
verify(const Node &coord_sys,
       Node &info)
{    
    bool res = true;

    if(!coord_sys.dtype().is_string())
    {
        log_error(info,"mesh::coordset::coord_system expected string");
        res = false;
    }
    else
    {
        std::string coord_sys_str = coord_sys.as_string();

        if(coord_sys_str == "xy" ||
           coord_sys_str == "xyz")
        {
            log_info(info,"type: " + coord_sys_str);
        }
        else
        {
            log_error(info,"mesh::coordset::coord_system unsupported value:"
                           + coord_sys_str);
            res = false;
        }
    }

    if(res)
    {
        info["valid"] = "true";
    }
    else
    {
        info["valid"] = "false";
    }

    return res;
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::coordset::coord_system --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// blueprint::mesh::coordset::index protocol interface
//-----------------------------------------------------------------------------
namespace index
{

//-----------------------------------------------------------------------------
bool
verify(const Node &n)
{
    Node info;
    return verify(n,info);
}


//-----------------------------------------------------------------------------
bool
verify(const Node &coordset_idx,
       Node &info)
{    
    bool res = true;

    // we need the mesh type
    if(!coordset_idx.has_child("type"))
    {
        log_error(info,"missing mesh::coordset::index child \"type\"");
        res = false;
    }
    else if(!coordset::type::verify(coordset_idx["type"],info["type"]))
    {
        res = false;
    }

    // we need a coord_system
    if(!coordset_idx.has_child("coord_system"))
    {
        log_error(info,"missing mesh::coordset::index child \"coord_system\"");
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
        log_error(info,"missing mesh::topology::index child \"path\"");
        res = false;
    }
    else if(!coordset_idx["path"].dtype().is_string())
    {
        log_error(info,"mesh::topology::index \"path\" is not a string");
        res = false;
    }
    
    if(res)
    {
        info["valid"] = "true";
    }
    else
    {
        info["valid"] = "false";
    }

    return res;
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::coordset::index --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::coordset --
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// blueprint::mesh::topology protocol interface
//-----------------------------------------------------------------------------
namespace topology
{

//-----------------------------------------------------------------------------
bool
verify(const Node &topology)
{
    Node info;
    return verify(topology,info);
}

//-----------------------------------------------------------------------------
bool
verify(const Node &coordset,
       Node &info)
{
    // TODO: IMPLEMENT!
    info["valid"] = "true";
    return true;
}

//-----------------------------------------------------------------------------
// blueprint::mesh::topology::index protocol interface
//-----------------------------------------------------------------------------
namespace index
{
//-----------------------------------------------------------------------------
bool
verify(const Node &n)
{
    Node info;
    return verify(n,info);
}

//-----------------------------------------------------------------------------
bool
verify(const Node &topo_idx,
       Node &info)
{    
    bool res = true;

    // we need the mesh type
    if(!topo_idx.has_child("type"))
    {
        log_error(info,"missing mesh::topology::index child \"type\"");
        res = false;
    }
    else if(!topology::type::verify(topo_idx["type"],info["type"]))
    {
        res = false;
    }

    // we need a coordset ref
    if(!topo_idx.has_child("coordset"))
    {
        log_error(info,"missing mesh::topology::index child \"coordset\"");
        res = false;
    }
    else if(!topo_idx["coordset"].dtype().is_string())
    {
        log_error(info,"mesh::topology::index \"coordset\" is not a string");
        res = false;
    }

    // we need a path
    if(!topo_idx.has_child("path"))
    {
        log_error(info,"missing mesh::topology::index child \"path\"");
        res = false;
    }
    else if(!topo_idx["path"].dtype().is_string())
    {
        log_error(info,"mesh::topology::index \"path\" is not a string");
        res = false;
    }
    
    if(res)
    {
        info["valid"] = "true";
    }
    else
    {
        info["valid"] = "false";
    }

    return res;
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::topology::index --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// blueprint::mesh::topology::type protocol interface
//-----------------------------------------------------------------------------
namespace type
{
//-----------------------------------------------------------------------------
bool
verify(const Node &n)
{
    Node info;
    return verify(n,info);
}

//-----------------------------------------------------------------------------
bool
verify(const Node &topo_type,
       Node &info)
{    
    bool res = true;

    if(!topo_type.dtype().is_string())
    {
        log_error(info,"mesh::topology::type expected string");
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
            log_info(info,"type: " + topo_type_str);
        }
        else
        {
            log_error(info,"mesh::topology::type unsupported value:"
                           + topo_type_str);
            res = false;
        }
    }

    if(res)
    {
        info["valid"] = "true";
    }
    else
    {
        info["valid"] = "false";
    }

    return res;
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::topology::type --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::topology --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// blueprint::mesh::field protocol interface
//-----------------------------------------------------------------------------
namespace field
{
//-----------------------------------------------------------------------------
bool
verify(const Node &n)
{
    Node info;
    return verify(n,info);
}

//-----------------------------------------------------------------------------
bool
verify(const Node &n,
       Node &info)
{
    // TODO: IMPLEMENT!
    info["valid"] = "true";
    return true;
}


//-----------------------------------------------------------------------------
// blueprint::mesh::field::association protocol interface
//-----------------------------------------------------------------------------
namespace association
{
    
//-----------------------------------------------------------------------------
bool
verify(const Node &n)
{
    Node info;
    return verify(n,info);
}

//-----------------------------------------------------------------------------
bool
verify(const Node &assoc,
       Node &info)
{    
    bool res = true;

    if(!assoc.dtype().is_string())
    {
        log_error(info,"mesh::field::association expected string");
        res = false;
    }
    else
    {
        std::string assoc_str = assoc.as_string();

        if(assoc_str == "point" ||
           assoc_str == "element")
        {
            log_info(info,"association: " + assoc_str );
        }
        else
        {
            log_error(info,"mesh::field::association unsupported value:"
                           + assoc_str);
            res = false;
        }
    }

    if(res)
    {
        info["valid"] = "true";
    }
    else
    {
        info["valid"] = "false";
    }

    return res;
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// end blueprint::mesh::field::association protocol interface
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// blueprint::mesh::field::basis protocol interface
//-----------------------------------------------------------------------------
namespace basis
{
    
//-----------------------------------------------------------------------------
bool
verify(const Node &n)
{
    Node info;
    return verify(n,info);
}

//-----------------------------------------------------------------------------
bool
verify(const Node &basis,
       Node &info)
{    
    bool res = true;

    if(!basis.dtype().is_string())
    {
        log_error(info,"mesh::field::basis expected string");
        res = false;
    }
    else
    {
        log_info(info,"basis: " + basis.as_string());
    }

    if(res)
    {
        info["valid"] = "true";
    }
    else
    {
        info["valid"] = "false";
    }

    return res;
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// end blueprint::mesh::field::basis protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// blueprint::mesh::field::index protocol interface
//-----------------------------------------------------------------------------
namespace index
{
    
//-----------------------------------------------------------------------------
bool
verify(const Node &n)
{
    Node info;
    return verify(n,info);
}

//-----------------------------------------------------------------------------
bool
verify(const Node &field_idx,
       Node &info)
{    
    bool res = true;

    // we need a topology
    if(!field_idx.has_child("topology"))
    {
        log_error(info,"missing mesh::field::index child \"topology\"");
        res = false;
    }
    else if(!field_idx["topology"].dtype().is_string())
    {
        log_error(info,"mesh::field::index \"topology\" is not a string");
        res = false;
    }

    // we need number_of_components
    if(!field_idx.has_child("number_of_components"))
    {
        log_error(info,"missing  mesh::field::index child "
                       "\"number_of_components\"");
        res = false;
    }
    else if(!field_idx["number_of_components"].dtype().is_integer())
    {
        log_error(info,"mesh::field::index \"number_of_components\" "
                       "is not an integer");
        res = false;
    }

    // we need a path
    if(!field_idx.has_child("path"))
    {
        log_error(info,"missing mesh::field::index child \"path\"");
        res = false;
    }
    else if(!field_idx["path"].dtype().is_string())
    {
        log_error(info,"mesh::field::index \"path\" is not a string");
        res = false;
    }
    
    bool has_assoc = field_idx.has_child("association");
    bool has_basis = field_idx.has_child("basis");
    
    // make sure we have either a basis or assoc entry
    if( ! (has_assoc || has_basis))
    {
        log_error(info,"missing mesh::field::index child "
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

    if(res)
    {
        info["valid"] = "true";
    }
    else
    {
        info["valid"] = "false";
    }

    return res;
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::field::index --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::field --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// blueprint::mesh::index protocol interface
//-----------------------------------------------------------------------------
namespace index
{
//-----------------------------------------------------------------------------
bool
verify(const Node &n)
{
    Node info;
    return verify(n,info);
}

//-----------------------------------------------------------------------------
bool
verify(const Node &n,
       Node &info)
{    
    // the mesh blueprint index provides metadata about a valid 
    // mesh blueprint conf
    //
    
    bool res = true;

    // note "errors" is a list, this allows us to log multiple errors.
    // Given that not conforming result is likely to trigger an error 
    // state in client code it seems like we should give as much info as
    // possible about what is wrong with the mesh, so we don't early
    // return when an error is found.
    
    // required: "coordsets", with at least one child
    //  each child must conform to "mesh::coordset"
    if(!n.has_child("coordsets"))
    {
        info["errors"].append().set("missing child node \"coordsets\"");
        res = false;
    }
    else if(n["coordsets"].number_of_children() == 0)
    {
        info["errors"].append().set("\"coordsets\" has no children");
        res = false;
    }
    else
    {
        NodeConstIterator itr = n["coordsets"].children();
    
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.path();

            if(!coordset::index::verify(chld,info["coordsets/" + chld_name]))
            {
                info["errors"].append().set(chld_name 
                                + " is not a valid blueprint::mesh::coordset::index");
                res = false;
            }
        }
    }
    
    // required: "topologies",  with at least one child
    // each child must conform to "mesh::topology"
    if(!n.has_child("topologies"))
    {
        info["errors"].append().set("missing child node \"topologies\"");
        res = false;
    }
    else if(n["topologies"].number_of_children() == 0)
    {
        info["errors"].append().set("\"topologies\" has no children");
        res = false;
    }
    else
    {
    
        NodeConstIterator itr = n["topologies"].children();
        
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.path();

            if(!topology::index::verify(chld,info["topologies"][chld_name]))
            {
                info["errors"].append().set(chld_name 
                                + " is not a valid blueprint::mesh::topology::index");
                res = false;
            }

            // make sure the topology references a valid coordset
            std::string coords_name = chld["coordset"].as_string();
            
            if(!n["coordsets"].has_child(coords_name))
            {
                std::ostringstream oss;
                oss << "topology index entry "
                    << "\"" << chld_name  << "\" "
                    << "references a non-existent coordset index entry"
                    << "\"" << coords_name  << "\" ";
                info["errors"].append().set(oss.str());
                res = false;
            }
            else if(info["coordsets"][coords_name]["valid"].as_string() != "true")
            {
                std::ostringstream oss;
                oss << "topology index entry "
                    << "\"" << chld_name  << "\" "
                    << "references a invalid coordset index entry "
                    << "\"" << coords_name  << "\" ";
                info["errors"].append().set(oss.str());
                res = false;
            }
        }
    }
    
    // optional: "fields", each child must conform to "mesh::field"
    if(n.has_path("fields"))
    {
        NodeConstIterator itr = n["fields"].children();
    
        while(itr.has_next())
        {
            const Node &chld = itr.next();
            const std::string chld_name = itr.path();

            if(!field::index::verify(chld,info["fields/" + chld_name]))
            {
                info["errors"].append().set(chld_name 
                                    + " is not a valid blueprint::mesh::field::index");
                res = false;
            }

            // make sure the field references a valid topology
            std::string topo_name = chld["topology"].as_string();
        
            if(!n["topologies"].has_child(topo_name))
            {
                std::ostringstream oss;
                oss << "field index entry"
                    << "\"" << chld_name  << "\" "
                    << "references a non-existent topology index entry"
                    << "\"" << topo_name  << "\" ";
                info["errors"].append().set(oss.str());
                res = false;
            }
            else if(info["topologies"][topo_name]["valid"].as_string() != "true")
            {
                std::ostringstream oss;
                oss << "field index entry"
                    << "\"" << chld_name  << "\" "
                    << "references a invalid topology index entry"
                    << "\"" << topo_name << "\" ";
                info["errors"].append().set(oss.str());
                res = false;
            }
        }
    }
    
    if(res)
    {
        info["valid"] = "true";
    }
    else
    {
        info["valid"] = "false";
    }

    return res;
}


//-----------------------------------------------------------------------------
void
generate(const Node &n,
         Node &index_out)
{
    index_out.reset();
    // TODO: IMPLEMENT!
}


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::index --
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

