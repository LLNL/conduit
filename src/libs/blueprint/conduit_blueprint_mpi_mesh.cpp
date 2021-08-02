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
#include "conduit_blueprint_mesh_partition.hpp"

#include "conduit_relay_mpi.hpp"
#include <mpi.h>
using partitioner = conduit::blueprint::mesh::partitioner;

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
    int par_size = relay::mpi::size(comm);

    // NOTE(JRC): MPI tasks without any domains should use a multi-domain
    // format with empty contents (i.e. an empty object or list node).
    int local_verify_ok = conduit::blueprint::mesh::verify(n, info) ? 1 : 0;
    int global_verify_ok = 0;

    Node n_snd, n_reduce;
    // make sure some MPI task actually had bp data
    n_snd.set_external(&local_verify_ok,1);
    n_reduce.set_external(&global_verify_ok,1);

    relay::mpi::sum_all_reduce(n_snd, n_reduce, comm);
    return global_verify_ok == par_size;
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

    // broadcast the resulting index to all other ranks
    relay::mpi::broadcast_using_schema(index_out,
                                       selected_rank,
                                       comm);
}


//-----------------------------------------------------------------------------
void generate_domain_to_rank_map(const conduit::Node &mesh,
                                 Node &domain_to_rank_map,
                                 MPI_Comm comm)
{
    int64 par_rank = relay::mpi::rank(comm);
    int64 max_local_id = -1;

    std::vector<const Node *> domains = ::conduit::blueprint::mesh::domains(mesh);
    std::vector<int64> local_domains;
    for(index_t di = 0; di < (index_t)domains.size(); di++)
    {
        const conduit::Node &domain = *domains[di];

        int64 domain_id = par_rank;
        if(domain.has_child("state") && domain["state"].has_child("domain_id"))
        {
            domain_id = domain["state/domain_id"].as_int64();
        }
        local_domains.push_back(domain_id);

        max_local_id = (domain_id > max_local_id) ? domain_id : max_local_id;
    }

    Node max_local, max_global;
    max_local.set_int64(max_local_id);
    max_global.set_int64(-1);
    relay::mpi::max_all_reduce(max_local, max_global, comm);

    std::vector<int64> local_map(max_global.as_int64() + 1, -1);
    for(auto m_itr = local_domains.begin(); m_itr != local_domains.end(); ++m_itr)
    {
        local_map[*m_itr] = par_rank;
    }

    Node local_par;
    local_par.set_external(&local_map[0], local_map.size());

    relay::mpi::max_all_reduce(local_par, domain_to_rank_map, comm);
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

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
/**
 @brief This class accepts a set of input meshes and repartitions them
        according to input options. This class subclasses the partitioner
        class to add some parallel functionality.
 @note This class overrides a small amount of code from the partitioner class
       so this class is in here so we do not have to make it public via a
       hpp file.
 */
class parallel_partitioner : public partitioner
{
public:
    parallel_partitioner(MPI_Comm c);
    virtual ~parallel_partitioner();

    virtual long get_total_selections() const override;

    virtual void get_largest_selection(int &sel_rank, int &sel_index) const override;

protected:
    virtual void map_chunks(const std::vector<partitioner::chunk> &chunks,
                            std::vector<int> &dest_ranks,
                            std::vector<int> &dest_domain) override;

    virtual void communicate_chunks(const std::vector<partitioner::chunk> &chunks,
                                    const std::vector<int> &dest_rank,
                                    const std::vector<int> &dest_domain,
                                    std::vector<chunk> &chunks_to_assemble,
                                    std::vector<int> &chunks_to_assemble_domains) override;

private:
    MPI_Comm comm;
};

//---------------------------------------------------------------------------
parallel_partitioner::parallel_partitioner(MPI_Comm c) : partitioner()
{
    comm = c;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
}

//---------------------------------------------------------------------------
parallel_partitioner::~parallel_partitioner()
{
}

//---------------------------------------------------------------------------
long
parallel_partitioner::get_total_selections() const
{
    // Gather the number of selections on each rank.
    long nselections = static_cast<long>(selections.size());
    long ntotal_selections = nselections;
    MPI_Allreduce(&nselections, &ntotal_selections, 1, MPI_LONG, MPI_SUM, comm);

    return ntotal_selections;
}

//---------------------------------------------------------------------------
/**
 @note This method is called iteratively until we have the number of target
       selections that we want to make. We could do better by identifying
       more selections to split in each pass.
 */
void
parallel_partitioner::get_largest_selection(int &sel_rank, int &sel_index) const
{
    // Find largest selection locally.
    long largest_selection_size = 0;
    int  largest_selection_index = 0;
    for(size_t i = 0; i < selections.size(); i++)
    {
        long ssize = static_cast<long>(selections[i]->length());
        if(ssize > largest_selection_size)
        {
            largest_selection_size = ssize;
            largest_selection_index = static_cast<int>(i);
        }
    }

    // What's the largest selection across ranks?
    long global_largest_selection_size = 0;
    MPI_Allreduce(&largest_selection_size, 
                  &global_largest_selection_size, 1, MPI_LONG,
                  MPI_MAX, comm);

    // See if this rank has the largest selection.
    int rank_that_matches = -1, largest_rank_that_matches = -1;
    int local_index = -1;
    for(size_t i = 0; i < selections.size(); i++)
    {
        long ssize = static_cast<long>(selections[i]->length());
        if(ssize == global_largest_selection_size)
        {
            rank_that_matches = rank;
            local_index = -1;
        }
    }
    MPI_Allreduce(&rank_that_matches,
                  &largest_rank_that_matches, 1, MPI_INT,
                  MPI_MAX, comm);

    sel_rank = largest_rank_that_matches;
    if(sel_rank == rank)
        sel_index = local_index;
}

//-------------------------------------------------------------------------
void
parallel_partitioner::map_chunks(const std::vector<partitioner::chunk> &chunks,
    std::vector<int> &dest_ranks,
    std::vector<int> &dest_domain)
{
    // Determine local chunk sizes.
    std::vector<uint64> local_chunk_sizes;
    for(size_t i =0 ; i < chunks.size(); i++)
    {
        const conduit::Node &n_topos = chunks[i].mesh->operator[]("topologies");
        uint64 len = 0;
        for(index_t j = 0; j < n_topos.number_of_children(); j++)
            len += conduit::blueprint::mesh::topology::length(n_topos[j]);
        local_chunk_sizes.push_back(len);
    }

    // Gather sizes of all local chunks.
    auto nlocal_chunks = static_cast<int>(local_chunk_sizes.size());
    std::vector<int> nglobal_chunks(size, 0);
    MPI_Allgather(&nlocal_chunks, 1, MPI_INT,
                  &nglobal_chunks[0], 1, MPI_INT,
                  comm);
    // Compute total chunks
    uint64 ntotal_chunks = 0;
    for(size_t i = 0; i < nglobal_chunks.size(); i++)
        ntotal_chunks += static_cast<uint64>(nglobal_chunks[i]);
#if 1
if(rank == 0)
{
cout << "ntotal_chunks = " << ntotal_chunks << endl;
}
#endif
    // Compute offsets
    std::vector<int> offsets(size, 0);
    for(size_t i = 1; i < nglobal_chunks.size(); i++)
        offsets = offsets[i-1] + nglobal_chunks[i-1];   

    // Get all chunk sizes across all ranks.
    std::vector<uint64> global_chunk_sizes(ntotal_chunks, 0);
    MPI_Allgatherv(&local_chunk_sizes[0],
                   static_cast<int>(local_chunk_sizes.size()),
                   MPI_UNSIGNED_LONG_LONG,
                   &global_chunk_sizes[0],
                   &nglobal_chunks[0],
                   &offsets[0],
                   MPI_UNSIGNED_LONG_LONG,
                   comm);

    // What we have at this point is a list of chunk sizes for all chunks
    // across all ranks. Let's get a global list of chunk destinations
    // and domains. A chunk may already know where it wants to go. If it
    // doesn't then we can assign it to move around. A chunk is free to 
    // move around if its destination rank is -1.
    std::vector<int> local_chunk_dest_rank(chunks.size(), rank);
    std::vector<int> local_chunk_dest_domain(chunks.size(), rank);
    for(size_t i =0 ; i < chunks.size(); i++)
    {
        local_chunk_dest_rank[i] = chunks[i]->destination_rank();
        local_chunk_dest_domain[i] = chunks[i]->destination_domain();
    }
    std::vector<int> global_chunk_dest_rank(ntotal_chunks, 0);
    std::vector<int> global_chunk_dest_domain(ntotal_chunks, 0);
    MPI_Allgatherv(&local_chunk_dest_rank[0],
                   static_cast<int>(local_chunk_dest_rank.size()),
                   MPI_INT,
                   &global_chunk_dest_rank[0],
                   &nglobal_chunks[0],
                   &offsets[0],
                   MPI_INT,
                   comm);
    MPI_Allgatherv(&local_chunk_dest_domain[0],
                   static_cast<int>(local_chunk_dest_domain.size()),
                   MPI_INT,
                   &global_chunk_dest_domain[0],
                   &nglobal_chunks[0],
                   &offsets[0],
                   MPI_INT,
                   comm);

    // Make sure that the global_chunk_dest_rank obeys the target number of
    // domains we're looking for.
    std::set<int> allowed_ranks;
    std::map<int,int> domain_sizes;
    uint64 free_to_move = 0;
    for(size_t i = 0; i < ntotal_chunks; i++)
    {
        int dest = global_chunk_dest_domain[i];
        if(dest >= 0)
        {
            std::map<int,int>::iterator it = domain_sizes.find(dest);
            if(it == domain_sizes.end())
                domain_sizes[dest] = 1;
            else
                it->second++;
        }
        else
            free_to_move++;
    }

    if(domain_sizes.size() == target && free_to_move == 0)
    {
        // The number of domains is equal to the number of targets
        // and we know the ranks where they are supposed to go so
        // we can pass back what the chunks told us to do.
        for(size_t i = 0; i < nlocal_chunks; i++)
        {
            dest_ranks.push_back(local_chunk_dest_rank[i]);
            dest_domain.push_back(local_chunk_dest_domain[i]);
        }
    }    
    else if(domain_sizes.empty() && free_to_move == ntotal_chunks)
    {
        // No chunks told us where they go so all chunks are free to move.

        // We want to make target chunks.
        std::vector<int> target_cell_counts(target, 0);       
        std::vector<int> global_dest_rank(ntotal_domains, 0);
        std::vector<int> global_dest_domain(ntotal_domains, 0);
        for(size_t i = 0; i < ntotal_chunks; i++)
        {
            // Add the size of this chunk to the sizes of the target domains.
            std::vector<int> next_target_cell_counts(target);
            for(size_t r = 0; r < target_cell_counts.size(); r++)
                next_target_cell_counts[r] = target_cell_counts[r] + global_chunk_sizes[i];

            // Find the index of the min value in next_target_cell_counts.
            size_t idx = 0;
            for(size_t r = 1; r < target_cell_counts.size(); r++)
            {
                if(next_target_cell_counts[r] < next_target_cell_counts[idx])
                    idx = r;
            }

            // Add the current chunk to the specified target domain.
            rank_cell_counts[idx] += global_chunk_sizes[i];
            global_dest_domain[i] = idx;
        }

        // We now have a global map indicating the final domain to which
        // each chunk will contribute. Now we need to map the target
        // domains to mpi ranks.


    }
    else if(dest_rank_counts.size() != target)
    {
        // The destination ranks that were specified by the chunks.
        // The set of ranks does not equal the target chunks though
        for(const auto &it : dest_rank_counts)
        {
            if(allowed_ranks.size() < target)
                allowed_ranks.push_back(it->first);
        }

        if(allowed_ranks.size() < target && free_to_move > 0)
        {
        }

        // Take the first #target ranks that we found and consider those
        // the ranks that will receive data. Any ranks beyond those
        // that were destination ranks will get their chunks reassigned
        // among those ranks.
        for(size_t i = 0; i < ntotal_chunks; i++)
        {
            for(size_t t = target; t < ranks.size(); t++)
            {
                if(global_chunk_dest_rank[i] == ranks[t])
                {
                    global_chunk_dest_rank[i] = -1;
                    break;
                }
            }
        }
    }
    else
    {
        //
    }

    // Count the element counts that we know of now so we get an idea of
    // the counts on all the ranks.
    std::vector<uint64> element_counts(size, 0);
    for(size_t i = 0; i < ntotal_chunks; i++)
    {
        if(global_chunk_dest_rank[i] >= 0)
            element_counts[global_chunk_dest_rank[i]] += global_chunk_sizes[i];
    }

    // Now, try to assign the free domains to the ranks that have the
    // smaller element counts.
    
    for(size_t i = 0; i < ntotal_chunks; i++)
    {
        if(global_chunk_dest_rank[i] < 0)
        {
            // global chunk i is free to move. Find the first, lowest
            element_counts[global_chunk_dest_rank[i]] += global_chunk_sizes[i];
    }
}

//-------------------------------------------------------------------------
void
parallel_partitioner::communicate_chunks(const std::vector<partitioner::chunk> &chunks,
    const std::vector<int> &dest_rank,
    const std::vector<int> &dest_domain,
    std::vector<partitioner::chunk> &chunks_to_assemble,
    std::vector<int> &chunks_to_assemble_domains)
{
    // TODO: send chunks to dest_rank if dest_rank[i] != rank.
    //       If dest_rank[i] == rank then the chunk stays on the rank.
    //
    //       Do sends/recvs to send the chunks as blobs among ranks.
    //
    //       Populate chunks_to_assemble, chunks_to_assemble_domains
}

//-------------------------------------------------------------------------
void
partition(const conduit::Node &n_mesh, const conduit::Node &options,
    conduit::Node &output, MPI_Comm comm)
{
    parallel_partitioner P(comm);
    if(P.initialize(n_mesh, options))
    {
        P.split_selections();
        output.reset();
        P.execute(output);
    }
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

