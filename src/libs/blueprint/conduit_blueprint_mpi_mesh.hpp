// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MPI_MESH_HPP
#define CONDUIT_BLUEPRINT_MPI_MESH_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_exports.h"
#include "conduit_blueprint_mesh.hpp"

#include <set>

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
// -- begin conduit::blueprint::mesh --
//-----------------------------------------------------------------------------

namespace mesh 
{

//-----------------------------------------------------------------------------
/// blueprint protocol verify interface
//-----------------------------------------------------------------------------

// mesh verify
//-----------------------------------------------------------------------------
bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info,
                                  MPI_Comm comm);

//-----------------------------------------------------------------------------
/// blueprint mesh property and transform methods
/// 
/// These methods can be called on any verified blueprint mesh.
//-----------------------------------------------------------------------------

//-------------------------------------------------------------------------
struct SharedFace
    {
        int64 m_face_id;
        int64 m_crse_pt = -1;
        int64 m_fine_pt = -1;
    };


struct PolyBndry
{
    int side; //which 3D side 0-5
    int64 m_nbr_rank;
    int64 m_nbr_id;
    std::vector<int64> m_elems; //elems of nbr domain that touch side
    std::map<int64, int64> m_bface; //map from nbr elem to face of nbr elem
    std::map<int64, std::vector<int64> > m_nbr_elems; //map from local
                                                          //elem to all
                                                          //nbr elems that
                                                          //touch it 
    //outer map: local elem, inner map: nbr elem to face
    std::map<int64, std::map<int64, SharedFace> > m_nbr_faces;
    std::set<int64> m_shared_fine;
};

void CONDUIT_BLUEPRINT_API generate_index(const conduit::Node &mesh,
                                          const std::string &ref_path,
                                          Node &index_out,
                                          MPI_Comm comm);


void CONDUIT_BLUEPRINT_API to_polygonal(const conduit::Node &n,
                                   conduit::Node &dest,
                                   const std::string& name);
void CONDUIT_BLUEPRINT_API to_polyhedral(const conduit::Node &n,
                                   conduit::Node &dest,
                                   const std::string& name);

void CONDUIT_BLUEPRINT_API match_nbr_elems(PolyBndry& pbnd,
    std::map<int, blueprint::mesh::connectivity::ElemType> nbr_elems,
    const Node& ref_topo,
    const Node& ref_win,
    const Node& nbr_win,
    int64 nbr_iwidth, int64 nbr_jwidth,
    int64 ni_lo, int64 nj_lo, int64 nk_lo,
    int64 ratio_i, int64 ratio_j, int64 ratio_k);


//-------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API generate_domain_to_rank_map(
                                              const conduit::Node &mesh,
                                              Node &domain_to_rank_map,
                                              MPI_Comm comm);

//-------------------------------------------------------------------------
index_t CONDUIT_BLUEPRINT_API number_of_domains(const conduit::Node &mesh,
                                                MPI_Comm comm);

void CONDUIT_BLUEPRINT_API generate_partition(const conduit::Node &mesh,
                                              Node &partition,
                                              MPI_Comm comm);

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



