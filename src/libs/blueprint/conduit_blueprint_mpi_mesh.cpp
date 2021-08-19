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

//tmp
using std::cout;
using std::endl;

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
    struct long_int
    {
        long value;
        int  rank;
    };

    struct chunk_info
    {
        uint64 num_elements;
        int destination_rank;
        int destination_domain;
    };

    virtual void map_chunks(const std::vector<partitioner::chunk> &chunks,
                            std::vector<int> &dest_ranks,
                            std::vector<int> &dest_domain,
                            std::vector<int> &offsets) override;

    virtual void communicate_chunks(const std::vector<partitioner::chunk> &chunks,
                                    const std::vector<int> &dest_rank,
                                    const std::vector<int> &dest_domain,
                                    const std::vector<int> &offsets,
                                    std::vector<chunk> &chunks_to_assemble,
                                    std::vector<int> &chunks_to_assemble_domains) override;

private:
    /**
     @brief Creates an MPI structure datatype so we can Allgatherv 3 things
            in 1 call. We initialize the chunk_info_dt member.
     */
    void create_chunk_info_dt();
    /**
     @brief Frees the chunk_info_dt data type.
     */
    void free_chunk_info_dt();

    MPI_Comm     comm;
    MPI_Datatype chunk_info_dt;
};

//---------------------------------------------------------------------------
parallel_partitioner::parallel_partitioner(MPI_Comm c) : partitioner()
{
    comm = c;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    create_chunk_info_dt();
}

//---------------------------------------------------------------------------
parallel_partitioner::~parallel_partitioner()
{
    free_chunk_info_dt();
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
       more selections to split in each pass. We use MPI_LONG_INT because it
       can be used with MPI_MAXLOC.
 */
void
parallel_partitioner::get_largest_selection(int &sel_rank, int &sel_index) const
{
    // Find largest selection locally.
    long_int largest_selection;
    largest_selection.value = 0;
    largest_selection.rank = rank;
    std::vector<uint64> local_sizes(selections.size());
    for(size_t i = 0; i < selections.size(); i++)
    {
        local_sizes[i] = static_cast<uint64>(selections[i]->length());
        if(local_sizes[i] > static_cast<uint64>(largest_selection.value))
        {
            largest_selection.value = static_cast<long>(local_sizes[i]);
        }
    }

    // What's the largest selection across ranks? We use MPI_MAXLOC to 
    // get the max size and the rank where it occurs.
    long_int global_largest_selection;
    MPI_Allreduce(&largest_selection, &global_largest_selection,
                  1, MPI_LONG_INT, MPI_MAXLOC, comm);

    // MPI found us the rank that has the largest selection.
    sel_rank = global_largest_selection.rank;
    sel_index = -1;

    // If we're on that rank, determine the local selection index.
    if(sel_rank == rank)
    {
        uint64 ssize = static_cast<uint64>(global_largest_selection.value);
        for(size_t i = 0; i < selections.size(); i++)
        {
            if(ssize == local_sizes[i])
            {
                sel_index = static_cast<int>(i);
                break;
            }
        }
    }
}

//-------------------------------------------------------------------------
void
parallel_partitioner::create_chunk_info_dt()
{
    chunk_info obj;
    int slen = 3;
    int lengths[3] = {1,1,1};
    MPI_Datatype types[3];
    MPI_Aint offsets[3];

    types[0] = MPI_UNSIGNED_LONG_LONG;
    types[1] = MPI_INT;
    types[2] = MPI_INT;

    size_t base = ((size_t)(&obj));
    offsets[0] = ((size_t)(&obj.num_elements)) - base;
    offsets[1] = ((size_t)(&obj.destination_rank)) - base;
    offsets[2] = ((size_t)(&obj.destination_domain)) - base;

    MPI_Type_create_struct(slen, lengths, offsets, types, &chunk_info_dt);
    MPI_Type_commit(&chunk_info_dt);
}

//-------------------------------------------------------------------------
void
parallel_partitioner::free_chunk_info_dt()
{
    MPI_Type_free(&chunk_info_dt);
}

//-------------------------------------------------------------------------
/**
The purpose of this function is to decide for a set of chunks on a rank, how
they will be assigned to final domains and where in the set of MPI ranks those
domains will live.

Some chunks will not care which domain they belong to nor where they
might end up. Such chunks will indicate a -1 for their domain number so we
have some freedom in how we assemble chunks into domains, according to the
target number of domains.

Some chunks may be the result of a field-based selection that
says explicitly where the cells will end up in a domain/rank. We can only have
a domain going to a single rank though.

@note We pass out the global dest_rank, dest_domain, offsets in this method
      since we consume it immediately in communicate_chunks where we need the
      global information to do matching sends/recvs.
*/
void
parallel_partitioner::map_chunks(const std::vector<partitioner::chunk> &chunks,
    std::vector<int> &dest_rank,
    std::vector<int> &dest_domain,
    std::vector<int> &_offsets)
{
    // Gather number of chunks on each rank.
    auto nlocal_chunks = static_cast<int>(chunks.size());
    std::vector<int> nglobal_chunks(size, 0);
    MPI_Allgather(&nlocal_chunks, 1, MPI_INT,
                  &nglobal_chunks[0], 1, MPI_INT,
                  comm);
    // Compute total chunks
    int ntotal_chunks = 0;
    for(size_t i = 0; i < nglobal_chunks.size(); i++)
        ntotal_chunks += nglobal_chunks[i];
#if 1
MPI_Barrier(comm);
if(rank == 0)
{
cout << "------------------------ map_chunks ------------------------" << endl;
cout << "ntotal_chunks = " << ntotal_chunks << endl;
}
MPI_Barrier(comm);
#endif
    // Compute offsets. We use int because of MPI_Allgatherv
    std::vector<int> offsets(size, 0);
    for(size_t i = 1; i < nglobal_chunks.size(); i++)
        offsets[i] = offsets[i-1] + nglobal_chunks[i-1];

    // What we have at this point is a list of chunk sizes for all chunks
    // across all ranks. Let's get a global list of chunk domains (where
    // they want to go). A chunk may already know where it wants to go.
    // If it doesn't then we can assign it to move around. A chunk is
    // free to move around if its destination domain is -1.
    std::vector<chunk_info> local_chunk_info(chunks.size());
    for(size_t i = 0; i < chunks.size(); i++)
    {
        const conduit::Node &n_topos = chunks[i].mesh->operator[]("topologies");
        uint64 len = 0;
        for(index_t j = 0; j < n_topos.number_of_children(); j++)
            len += conduit::blueprint::mesh::topology::length(n_topos[j]);
        local_chunk_info[i].num_elements = len;
        local_chunk_info[i].destination_rank   = chunks[i].destination_rank;
        local_chunk_info[i].destination_domain = chunks[i].destination_domain;
    }
    std::vector<chunk_info> global_chunk_info(ntotal_chunks);
    MPI_Allgatherv(&local_chunk_info[0],
                   static_cast<int>(local_chunk_info.size()),
                   chunk_info_dt,
                   &global_chunk_info[0],
                   &nglobal_chunks[0],
                   &offsets[0],
                   chunk_info_dt,
                   comm);
#if 1
    if(rank == 0)
    {
        for(int i = 0; i < ntotal_chunks; i++)
        {
            cout << "global_chunk_info[" << i << "]={"
                 << "num_elements=" << global_chunk_info[i].num_elements
                 << ", dest_rank=" << global_chunk_info[i].destination_rank
                 << ", dest_domain=" << global_chunk_info[i].destination_domain
                 << "}" << endl;
        }
    }
#endif

    // Determine how many ranks are free to move to various domains.
    // Also determine the domain ids in use and how many chunks
    // comprise each of them.
    std::map<int,int> domain_sizes;
    int free_to_move = 0;
    for(int i = 0; i < ntotal_chunks; i++)
    {
        int domid = global_chunk_info[i].destination_domain;
        if(domid >= 0)
        {
            std::map<int,int>::iterator it = domain_sizes.find(domid);
            if(it == domain_sizes.end())
                domain_sizes[domid] = 1;
            else
                it->second++;
        }
        else
            free_to_move++;
    }

#if 1
    if(rank == 0)
    {
        cout << "domain_sizes = {";
        for(std::map<int,int>::const_iterator it = domain_sizes.begin();
            it != domain_sizes.end(); it++)
        {
            cout << it->first << ": " << it->second << ", ";
        }
        cout << "}" << endl;
        cout << "free_to_move = " << free_to_move << endl;
    }
#endif

    if(free_to_move == 0)
    {
        // No chunks are free to move around. This means we the domains
        // that we want them all to belong to.

        // NOTE: This may mean that we do not get #target domains though.
        if(!domain_sizes.empty() && domain_sizes.size() != target)
        {
            CONDUIT_WARN("The unique number of domain ids "
                << domain_sizes.size()
                << " was not equal to the desired target number of domains: "
                << target  << ".");
        }

        // Pass out global information
        dest_rank.reserve(ntotal_chunks);
        dest_domain.reserve(ntotal_chunks);
        for(int i = 0; i < ntotal_chunks; i++)
        {
            dest_rank.push_back(global_chunk_info[i].destination_rank);
            dest_domain.push_back(global_chunk_info[i].destination_domain);
        }
        _offsets.swap(offsets);
    }
    else if(free_to_move == ntotal_chunks)
    {
        // No chunks told us where they go so ALL are free to move.
#if 1
        if(rank == 0)
        {
            cout << "** We decide where chunks go." << endl;
        }
#endif
        // We must make #target domains from the chunks we have. Since
        // no chunks told us a domain id they want to be inside, we can
        // number domains 0..target. This scheme ignores the chunk's
        // destination_rank.

        std::vector<uint64> target_element_counts(target, 0);
        std::vector<uint64> next_target_element_counts(target);
        std::vector<int> global_dest_rank(ntotal_chunks, -1);
        std::vector<int> global_dest_domain(ntotal_chunks, 0);
        for(int i = 0; i < ntotal_chunks; i++)
        {
            // Add the size of this chunk to all targets.
            for(unsigned int r = 0; r < target; r++)
            {
                next_target_element_counts[r] = target_element_counts[r] +
                                                global_chunk_info[i].num_elements;
            }

            // Find the index of the min value in next_target_element_counts.
            // This way, we'll let that target domain have the chunk as
            // we are trying to balance the sizes of the output domains.
            //
            // NOTE: We could consider other metrics too such as making the
            //       smallest bounding box so we keep things close together.
            //
            // NOTE: This method has the potential to move chunks far away.
            //       It is sprinkling chunks into targets 0,1,2,... and 
            //       and then repeating when the number of elements is ascending.
            int idx = 0;
            for(unsigned int r = 1; r < target; r++)
            {
                if(next_target_element_counts[r] < next_target_element_counts[idx])
                    idx = r;
            }

            // Add the current chunk to the specified target domain.
            target_element_counts[idx] += global_chunk_info[i].num_elements;
            global_dest_domain[i] = idx;
#if 1
            if(rank == 0)
            {
                cout << "Add chunk " << i << " to domain " << idx << " (nelem=" << target_element_counts[idx] << ")" << endl;
            }
#endif
        }

        // We now have a global map indicating the final domain to which
        // each chunk will contribute. Spread target domains across size ranks.
        std::vector<int> rank_domain_count(size, 0);
        int divsize = std::min(size, static_cast<int>(target));
        for(unsigned int i = 0; i < target; i++)
            rank_domain_count[i % divsize]++;

        // Figure out which source chunks join which ranks.
        int target_id = 0;
        for(int r = 0; r < size; r++)
        {
if(rank == 0) cout << "r=" << r << ", rank_domain_count[r]=" << rank_domain_count[r] << endl;
            if(rank_domain_count[r] == 0)
                break;
            // For each domain on this rank r.
            for(int j = 0; j < rank_domain_count[r]; j++)
            {
                for(int i = 0; i < ntotal_chunks; i++)
                {
                    if(global_dest_domain[i] == target_id)
                    {
if(rank == 0)
cout << "global domain " << target_id << " goes to " << r << endl;
                        global_dest_rank[i] = r;
                    }
                }
                target_id++;
            }
        }
#if 1
        if(rank == 0)
        {
            cout << "target=" << target << endl;
            cout << "target_element_counts = {";
            for(size_t i = 0; i < target_element_counts.size(); i++)
                cout << target_element_counts[i] << ", ";
            cout << "}" << endl;
            cout << "global_dest_rank = {";
            for(size_t i = 0; i < global_dest_rank.size(); i++)
                cout << global_dest_rank[i] << ", ";
            cout << "}" << endl;
            cout << "global_dest_domain = {";
            for(size_t i = 0; i < global_dest_domain.size(); i++)
                cout << global_dest_domain[i] << ", ";
            cout << "}" << endl;
            cout << "rank_domain_count = {";
            for(size_t i = 0; i < rank_domain_count.size(); i++)
                cout << rank_domain_count[i] << ", ";
            cout << "}" << endl;
        }
#endif
        // Pass out the global information.
        dest_rank.swap(global_dest_rank);
        dest_domain.swap(global_dest_domain);
        _offsets.swap(offsets);
    }
    else
    {
        // There must have been a combination of chunks that told us where
        // they want to go and some that did not. Determine whether we want
        // to handle this.
        CONDUIT_ERROR("Invalid mixture of destination rank/domain specifications.");
    }
#if 1
    // We're passing out global info now so all ranks should be the same.

    if(rank == 0)
    {
        std::cout << rank << ": dest_ranks={";
        for(size_t i = 0; i < dest_rank.size(); i++)
            std::cout << dest_rank[i] << ", ";
        std::cout << "}" << std::endl;
        std::cout << rank << ": dest_domain={";
        for(size_t i = 0; i < dest_domain.size(); i++)
            std::cout << dest_domain[i] << ", ";
        std::cout << "}" << std::endl;
        std::cout.flush();
    }
#endif
}

//-------------------------------------------------------------------------
/**
 @note In the parallel version of this function, we pass in global information
       for dest_rank, dest_domain, offsets. This helps us know not only the
       domains to which we have to send but also those who are sending to
       this rank.
 */
void
parallel_partitioner::communicate_chunks(const std::vector<partitioner::chunk> &chunks,
    const std::vector<int> &dest_rank,
    const std::vector<int> &dest_domain,
    const std::vector<int> &offsets,
    std::vector<partitioner::chunk> &chunks_to_assemble,
    std::vector<int> &chunks_to_assemble_domains)
{
    const int PARTITION_TAG_BASE = 12000;

    // Use the offsets to determine the sender for each chunk.
    std::vector<int> src_rank(dest_rank.size(),size-1);
    size_t idx = 0;
    for(size_t r = 1; r < offsets.size(); r++)
    {
        int n = offsets[r] - offsets[r-1];
        for(int i = 0; i < n; i++)
            src_rank[idx++] = r-1;
    }

#if 1
    MPI_Barrier(comm);
    if(rank == 0)
    {
        cout << "offsets = {";
        for(size_t i = 0; i < offsets.size(); i++)
            cout << offsets[i] << ", ";
        cout << "}" << endl;
        cout << "src_rank = {";
        for(size_t i = 0; i < src_rank.size(); i++)
            cout << src_rank[i] << ", ";
        cout << "}" << endl;
    }
    MPI_Barrier(comm);
#endif

    // Do sends for the chunks we own on this processor that must migrate.
    for(size_t i = 0; i < chunks.size(); i++)
    {
        int gidx = offsets[rank] + i;
        int tag = PARTITION_TAG_BASE + gidx;
        int dest = dest_rank[gidx];
        // If we're not sending to self, send the chunk.
        if(dest != rank)
        {
cout << rank << ": send_using_schema(dest=" << dest << ", tag=" << tag << ")" << endl;
            conduit::relay::mpi::send_using_schema(*chunks[i].mesh, dest, tag, comm);
        }
    }

    // Do recvs.
    for(size_t i = 0; i < dest_rank.size(); i++)
    {
        if(dest_rank[i] == rank)
        {
            int gidx = i;
            int tag = PARTITION_TAG_BASE + gidx;
            int start = offsets[rank];
            int end = start + chunks.size();
            bool this_rank_already_owns_it = (gidx >= start && gidx < end);
            if(this_rank_already_owns_it)
            {
                int local_i = i - offsets[rank];
                // Pass the chunk through since we already own it on this rank.
                chunks_to_assemble.push_back(chunk(chunks[local_i].mesh, false));
                chunks_to_assemble_domains.push_back(dest_domain[i]);
            }
            else
            {
cout << rank << ": recv_using_schema(src=" << src_rank[i] << ", tag=" << tag << ")" << endl;

                conduit::Node *n_recv = new conduit::Node;              
                conduit::relay::mpi::recv_using_schema(*n_recv, src_rank[i], tag, comm);
                // Save the received chunk and indicate we own it for later.
                chunks_to_assemble.push_back(chunk(n_recv, true));
                chunks_to_assemble_domains.push_back(dest_domain[i]);
            }
        }
    }


#if 0
    const int PARTITION_TAG_BASE = 12345;

    // If we just have the destination rank/domain then we can send but we do
    // not know who is sending to us. It would be helpful to have the offsets
    // too.

    // Serialize the chunks into binary char buffers.
    std::vector<std::shared_ptr<std::vector<char>>> serialized;
    std::vector<uint64> serialized_lengths;
    for(size_t i = 0; i < chunks.size(); i++)
    {
        auto sbuf = std::make_shared<std::vector<char>>();
        serialized.push_back(sbuf);

        // TODO: serialize chunks[i] into sbuf

        serialized_lengths.push_back(static_cast<uint64>(sbuf->size());
    }

    // How many ranks are sending here.
    int nsend_here = 0;
    for(size_t i = 0; i < dest_rank.size(); i++)
    {
        if(dest_rank[i] == rank)
            nsend_here++;
    }

    // Iterate over this rank's chunks and send lengths of the buffers to
    // the ranks that need to know, provided we are not sending to ourselves.
    std::vector<MPI_Request> reqs(chunks.size() + nsend_here);
    size_t nreq = 0;
    for(size_t i = 0; i < chunks.size(); i++)
    {
        int gchunkid = offsets[rank] + i;
        int tag = PARTITION_TAG_BASE + gchunkid;
        MPI_Isend(&serialized_lengths[i], 1, MPI_UNSIGNED_LONG_LONG,
                  dest_rank[gchunkid], tag, comm, &reqs[nreq]);
        nreq++;
    }

    // Use dest_rank and offsets to determine which ranks are sending
    // each chunk.
    std::vector<int> sender;
    sender.reserve(dest_rank.size());
    for(int r = 0; r < size; r++)
    {
        int n = 0;
        if(r < size-1)
            n = offsets[r+1] - offsets[r];
        else
            n = dest_rank.size() - offsets[r];
        for(int i = 0; i < n; i++)
            sender.push_back(r);
    }

    // Recv on the lengths.
    std::vector<uint64> recv_serialized_lengths(nsend_here, 0);
    int nrli = 0;
    for(size_t i = 0; i < dest_rank.size(); i++)
    {
        int gchunkid = i;
        int tag = PARTITION_TAG_BASE + gchunkid;
        if(dest_rank[gchunkid] == rank)
        {
            MPI_Irecv(&recv_serialized_lengths[nrli], 1, MPI_UNSIGNED_LONG_LONG,
                      sender[gchunkid], tag, comm, &reqs[nreq]);
            nreq++;
            nrli++;
        }
    }

    // Wait for all of the sends/recvs to land
    std::vector<MPI_Status> s(nreq+1);
    MPI_Waitall(nreq, &reqs[0], &s[0]);


    // Post some recvs.

    // Post some sends.
    for(
    // MPI_Waitall


    // TODO: send chunks to dest_rank if dest_rank[i] != rank.
    //       If dest_rank[i] == rank then the chunk stays on the rank.
    //
    //       Do sends/recvs to send the chunks as blobs among ranks.
    //
    //       Populate chunks_to_assemble, chunks_to_assemble_domains
#endif
}

//-------------------------------------------------------------------------
void
partition(const conduit::Node &n_mesh, const conduit::Node &options,
    conduit::Node &output, MPI_Comm comm)
{
    parallel_partitioner P(comm);

    // Partitioners on different ranks ought to return the same value but
    // perhaps some did not when they examined their own domains against
    // selection.
    int iinit, ginit;
    iinit = P.initialize(n_mesh, options) ? 1 : 0;
    MPI_Allreduce(&iinit, &ginit, 1, MPI_INT, MPI_MAX, comm);
    if(ginit > 0)
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

