// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint.cpp
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
#include "conduit_blueprint.hpp"


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


//---------------------------------------------------------------------------//
std::string
about()
{
    Node n;
    blueprint::about(n);
    return n.to_yaml();
}

//---------------------------------------------------------------------------//
void
about(Node &n)
{
    n.reset();
    n["protocols/mesh/coordset"] = "enabled";
    n["protocols/mesh/topology"] = "enabled";
    n["protocols/mesh/field"]    = "enabled";
    n["protocols/mesh/index"]    = "enabled";
    
    n["protocols/mcarray"]  = "enabled";
    n["protocols/zfparray"] = "enabled";
}

//---------------------------------------------------------------------------//
bool
verify(const std::string &protocol,
       const Node &n,
       Node &info)
{
    bool res = false;
    info.reset();
    
    std::string p_curr;
    std::string p_next;
    conduit::utils::split_path(protocol,p_curr,p_next);

    if(!p_next.empty())
    {
        if(p_curr == "mesh")
        {
            res = mesh::verify(p_next,n,info);
        }
        else if(p_curr == "mcarray")
        {
            res = mcarray::verify(p_next,n,info);
        }
    }
    else
    {
        if(p_curr == "mesh")
        {
            res = mesh::verify(n,info);
        }
        else if(p_curr == "mcarray")
        {
            res = mcarray::verify(n,info);
        }
    }

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

