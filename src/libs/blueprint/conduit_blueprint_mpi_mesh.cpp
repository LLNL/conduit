// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_mpi_mesh.hpp"
#include "conduit_relay_mpi.hpp"

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

//-----------------------------------------------------------------------------
// blueprint::mesh::index protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool
verify(const conduit::Node &n,
       conduit::Node &info,
       MPI_Comm comm)
{
    // some MPI tasks may not have data, that is fine
    // but blueprint verify will fail, so if the
    // input node is empty skip verify
    int local_verify_ok = 0;
    if(!n.dtype().is_empty())
    {
        if(conduit::blueprint::mesh::verify(n,
                                            info))
        {
            local_verify_ok = 1;
        }
    }

    int global_verify_ok = 0;

    Node n_snd, n_reduce;
    // make sure some MPI task actually had bp data
    n_snd.set_external(&local_verify_ok,1);
    n_reduce.set_external(&global_verify_ok,1);

    relay::mpi::sum_all_reduce(n_snd, n_reduce, comm);
    return global_verify_ok > 0;
}

//-------------------------------------------------------------------------
void
generate_index(const conduit::Node &mesh,
               const std::string &ref_path,
               Node &index_out,
               MPI_Comm comm)
{
    int par_rank = relay::mpi::rank(comm);
    int par_size = relay::mpi::size(comm);
    // we need to know the mesh structure and the number of domains
    // we can't assume rank zero has any domains (could be empty)
    // so we look for the lowest rank with 1 or more domains

    index_t local_num_domains = ::conduit::blueprint::mesh::number_of_domains(mesh);
    index_t global_num_domains = number_of_domains(mesh,comm);

    index_t rank_send = par_size;
    index_t selected_rank = par_size;
    if(local_num_domains > 0)
        rank_send = par_rank;

    Node n_snd, n_reduce;
    // make sure some MPI task actually had bp data
    n_snd.set_external(&rank_send,1);
    n_reduce.set_external(&selected_rank,1);

    relay::mpi::min_all_reduce(n_snd, n_reduce, comm);

    if(par_rank == selected_rank )
    {
        if(::conduit::blueprint::mesh::is_multi_domain(mesh))
        {
            ::conduit::blueprint::mesh::generate_index(mesh.child(0),
                                                       ref_path,
                                                       global_num_domains,
                                                       index_out);
        }
        else
        {
            ::conduit::blueprint::mesh::generate_index(mesh,
                                                       ref_path,
                                                       global_num_domains,
                                                       index_out);
        }
    }

    // determine if a partition map is needed:  must be multi-domain
    // and have state/domain_id in the right location on selected_rank.
    Node do_par_map;
    do_par_map.set_uint8(0);
    if(par_rank == selected_rank &&
       ::conduit::blueprint::mesh::is_multi_domain(mesh))
    {
        if(mesh.child(0).has_child("state") &&
           mesh.child(0)["state"].has_child("domain_id"))
        {   
            do_par_map.set_uint8(1);
        }
    }

    relay::mpi::broadcast(do_par_map, selected_rank, comm);

    // generate the partition map if needed.    
    Node partition;
    if(do_par_map.as_uint8())
    {
        generate_partition(mesh, partition, comm);
    }

    if(par_rank == selected_rank && !partition.dtype().is_empty())
    {
        index_out["state/partition_size"].set(par_size);
        index_out["state/domain_to_partition_map"].set(partition);
    }

    // broadcast the resulting index to all other ranks
    relay::mpi::broadcast_using_schema(index_out,
                                       selected_rank,
                                       comm);
}


//-----------------------------------------------------------------------------
void generate_partition(const conduit::Node &mesh,
                        Node &partition,
                        MPI_Comm comm)
{
    int par_rank = relay::mpi::rank(comm);

    std::vector<conduit::int64> local_domains;

    conduit::NodeConstIterator itr = mesh.children();
    while (itr.has_next())
    {
        const conduit::Node &chld = itr.next();
        if (chld.has_child("state"))
        {
            const conduit::Node &state = chld["state"];
            if (state.has_child("domain_id"))
            {
                 conduit::int64 dom_id = state["domain_id"].as_int64();
                 local_domains.push_back(dom_id);
            }
        }
    }

    Node num_local, num_global;
    num_local.set_int64(local_domains.size());
    num_global.set_int64(0);
    relay::mpi::sum_all_reduce(num_local, num_global, comm);

    std::vector<int64> local_partition(num_global.as_int64(), 0);
    for (auto m_itr = local_domains.begin(); m_itr != local_domains.end();
         ++m_itr)
    {
         local_partition[*m_itr] = par_rank;
    }

    Node local_par;
    local_par.set_external(&local_partition[0], local_partition.size());

    relay::mpi::max_all_reduce(local_par, partition, comm);
}

//-----------------------------------------------------------------------------
index_t
number_of_domains(const conduit::Node &n,
                  MPI_Comm comm)
{
    // called only when mesh bp very is true, simplifies logic here
    index_t local_num_domains = 0;
    if(!n.dtype().is_empty())
    {
        local_num_domains = ::conduit::blueprint::mesh::number_of_domains(n);
    }

    index_t global_num_domains = 0;

    Node n_snd, n_reduce;
    // count all domains with mpi
    n_snd.set_external(&local_num_domains,1);
    n_reduce.set_external(&global_num_domains,1);

    relay::mpi::all_reduce(n_snd, n_reduce, MPI_SUM, comm);
    return global_num_domains;
}


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh --
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

