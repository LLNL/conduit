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

     @param mesh A node that holds one or more domains. Each domain must have
                 state/domain_id that uniquely identifies the domain.
     @param comm The MPI communicator that will be used for communication.
     */
    PointQuery(const conduit::Node &mesh, MPI_Comm comm);

    /**
     @brief Destructor.
     */
    virtual ~PointQuery() override = default;

    /**
     @brief Execute all of the point queries. If a rank queries a point that
            exists in a remote domain, the remote domain's owning rank will
            execute the query and return the values.

     @param coordsetName The name of the coordset we're searching in the domains.

     @note This method must be called on all ranks in the communicator.
     */
    virtual void execute(const std::string &coordsetName) override;

protected:
    MPI_Comm m_comm;  
};

//---------------------------------------------------------------------------
/**
 @brief Execute a set of membership queries in parallel where queries were
        issued by multiple MPI ranks.
 */
class CONDUIT_BLUEPRINT_API MatchQuery :
    public conduit::blueprint::mesh::utils::query::MatchQuery
{
public:
    /**
     @brief Constructor

     @param mesh A node that holds one or more domains.
     @param comm The MPI communicator that will be used for communication.
     */
    MatchQuery(const conduit::Node &mesh, MPI_Comm comm);

    /**
     @brief Destructor.
     */
    virtual ~MatchQuery() override = default;

    /**
     @brief Execute all of the queries.

     @param shape The type of shapes being queried.

     @note This method must be called on all ranks in the communicator.
     */
    virtual void execute() override;

protected:
    MPI_Comm m_comm;
};

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh::utils::query --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi::mesh::utils::adjset --
//-----------------------------------------------------------------------------
namespace adjset
{
    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API canonicalize(Node &adjset);

    //-------------------------------------------------------------------------
    /**
     @brief Given a set of domains, make sure that the specified adjset in them
            is valid and flag any errors in the info node. This function will
            make sure that each domain's adjset references valid entities in
            neighboring domains.

     @param doms A node containing the domains. There must be multiple domains.
     @param adjsetName The name of the adjset in all domains. It must exist.
     @param[out] info A node that contains any errors.
     @param comm The MPI communicator to use.

     @return True if the adjsets in all domains contained no errors; False if
             there were errors.
     */
    bool CONDUIT_BLUEPRINT_API validate(const Node &doms,
                                        const std::string &adjsetName,
                                        Node &info,
                                        MPI_Comm comm);

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh::utils::adjset --
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
