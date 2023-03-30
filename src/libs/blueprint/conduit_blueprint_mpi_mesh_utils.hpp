// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh_utils.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MPI_MESH_UTILS_HPP
#define CONDUIT_BLUEPRINT_MPI_MESH_UTILS_HPP

//-----------------------------------------------------------------------------
// std includes
//-----------------------------------------------------------------------------
#include <string>

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_exports.h"
#include "conduit_blueprint_mesh_utils.hpp"

#include <mpi.h>

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
// -- begin conduit::blueprint::mpi --
//-----------------------------------------------------------------------------
namespace mpi
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi::mesh::utils --
//-----------------------------------------------------------------------------
namespace utils
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi::mesh::utils::query --
//-----------------------------------------------------------------------------
namespace query
{

/**
 @brief Execute a set of point queries in parallel where the mesh domains are
        distributed among MPI ranks.
 */
class CONDUIT_BLUEPRINT_API PointQuery :
    public conduit::blueprint::mesh::utils::query::PointQuery
{
public:
    /**
     @brief Constructor

     @param mesh A node that holds one or more domains.
     @param comm The MPI communicator that will be used for communication.
     */
    PointQuery(const conduit::Node &mesh, MPI_Comm comm);

    /**
     @brief Destructor.
     */
    virtual ~PointQuery() = default;

    /**
     @brief Execute all of the point queries. If a rank queries a point that
            exists in a remote domain, the remote domain's owning rank will
            execute the query and return the values.

     @param coordsetName The name of the coordset we're searching in the domains.

     @note This method must be called on all ranks in the communicator.
     */
    virtual void Execute(const std::string &coordsetName) override;

protected:
    MPI_Comm m_comm;  
};

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh::utils::query --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh::utils --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
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


#endif
