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
        std::string msg = "missing uniform coordset child \"value\"";
        info["errors"].append().set(msg);
        res = false;
    }
    else // dims needs at least one child ("i") 
    {
        const Node &dims = coordset["dims"];
        
        if(!dims.has_child("i"))
        {
            std::string msg ="missing coordset child \"dims\"";
            info["errors"].append().set(msg);
            res = false;
        }
        else if(!dims["i"].dtype().is_number())
        {
            std::string msg ="uniform dims/i is not a number";
            info["errors"].append().set(msg); 
            res = false;
        }
        
        if(dims.has_child("j"))
        {
            info["optional"].append().set("uniform dims includes j");
            if(!dims["j"].dtype().is_number())
            {
                std::string msg = "uniform dims/j is not a number";
                info["errors"].append().set(msg);
                res = false;
            }
        }
        
        if(dims.has_child("k"))
        {
            info["optional"].append().set("uniform dims includes k");
            if(!dims["k"].dtype().is_number())
            {
                std::string msg = "uniform dims/k is not a number";
                info["errors"].append().set(msg);
                res = false;
            }
        }
    }
    
    if(!coordset.has_child("origin"))
    {
        const Node &origin = coordset["origin"];

        info["optional"].append().set("uniform coordset has origin");

        if(origin.has_child("x"))
        {
            if(!origin["x"].dtype().is_number())
            {
                std::string msg = "uniform origin/x is not a number";
                info["errors"].append().set(msg);
                res = false;
            }
        }

        if(origin.has_child("y"))
        {
            if(!origin["y"].dtype().is_number())
            {
                std::string msg = "uniform origin/y is not a number";
                info["errors"].append().set(msg);
                res = false;
            }
        }

        if(origin.has_child("z"))
        {
            if(!origin["z"].dtype().is_number())
            {
                std::string msg = "uniform origin/z is not a number";
                info["errors"].append().set(msg);
                res = false;
            }
        }
    }

    if(!coordset.has_child("spacing"))
    {
        const Node &spacing = coordset["spacing"];

        info["optional"].append().set("uniform coordset has spacing");

        if(spacing.has_child("dx"))
        {

            if(!spacing["dx"].dtype().is_number())
            {
                std::string msg = "uniform spacing/dx is not a number";
                info["errors"].append().set(msg);
                res = false;
            }
        }

        if(spacing.has_child("dy"))
        {

            if(!spacing["dy"].dtype().is_number())
            {
                std::string msg = "uniform spacing/dy is not a number";
                info["errors"].append().set(msg);
                res = false;
            }
        }

        if(spacing.has_child("dz"))
        {

            if(!spacing["dz"].dtype().is_number())
            {
                std::string msg = "uniform spacing/dz is not a number";
                info["errors"].append().set(msg);
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
        std::string msg = "missing rectilinear coordset child \"value\"";
        info["errors"].append().set(msg);
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
        std::string msg = "missing explicit coordset child \"value\"";
        info["errors"].append().set(msg);
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
        info["errors"].append().set("missing coordset child \"type\"");
        res = false;
    }
    else if(!coordset["type"].dtype().is_string())
    {
        info["errors"].append().set("coordset \"type\" is not a string");
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
        else
        {
            info["errors"].append().set("unknown coordset type: \"" 
                                        + type_name + "\"");
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
    // TODO: IMPLEMENT!
    info["valid"] = "true";
    return true;
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

