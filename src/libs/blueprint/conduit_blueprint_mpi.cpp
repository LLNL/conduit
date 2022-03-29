// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi.cpp
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
#include "conduit_blueprint_mpi.hpp"


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
// -- begin conduit::blueprint::mpi --
//-----------------------------------------------------------------------------
namespace mpi
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
    n["protocols/mesh"] = "enabled";

// advertise features
    n["features/mesh/partition"] = "enabled";
    n["features/mesh/flatten"] = "enabled";

#ifdef CONDUIT_BLUEPRINT_MPI_PARMETIS_ENABLED
    n["features/mesh/parmetis"] = "enabled";
#else
    n["features/mesh/parmetis"] = "disabled";
#endif

}

//---------------------------------------------------------------------------//
bool
verify(const std::string &protocol,
       const Node &n,
       Node &info,
       MPI_Comm comm)
{
    bool res = false;
    info.reset();

    std::string p_curr;
    std::string p_next;
    conduit::utils::split_path(protocol,p_curr,p_next);

    if(p_curr == "mesh")
    {
        res = conduit::blueprint::mpi::mesh::verify(n,info,comm);
    }

    return res;
}


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

