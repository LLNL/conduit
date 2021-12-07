// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh_partition.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MPI_MESH_PARTITION_HPP
#define CONDUIT_BLUEPRINT_MPI_MESH_PARTITION_HPP

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh_partition.hpp"
#include "conduit_blueprint_exports.h"

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

/**
 @brief This class accepts a set of input meshes and repartitions them
        according to input options. This class subclasses the partitioner
        class to add some parallel functionality.
 @note This class overrides a small amount of code from the partitioner class
       so this class is in here so we do not have to make it public via a
       hpp file.
 */
class CONDUIT_BLUEPRINT_API ParallelPartitioner
: public conduit::blueprint::mesh::Partitioner
{
public:
    ParallelPartitioner(MPI_Comm c);
    virtual ~ParallelPartitioner();

protected:

    virtual void init_dom_to_rank_map(const conduit::Node& n_mesh);

    virtual int get_rank_offset(const std::vector<int>& chunk_offsets);

    virtual long get_total_selections() const override;

    /**
     @note This method is overridden so we can check options across all ranks
           for a target value and provide a consistent result, even if the
           ranks were passed different options. We return the max of any
           provided target.
     */
    virtual bool options_get_target(const conduit::Node &options,
                                    unsigned int &value) const override;

    /**
     @brief Examines the selections and counts them to determine a number of
            targets that would be produced. Most selections result in a domain
            but some selections may combine into a single domain if their
            destination domain is set to the same value.
     @return The number of targets we expect to create.
     */
    virtual unsigned int count_targets() const override;

    virtual void get_largest_selection(int &sel_rank, int &sel_index) const override;

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

    virtual void map_chunks(const std::vector<Chunk> &chunks,
                            std::vector<int> &dest_ranks,
                            std::vector<int> &dest_domain,
                            std::vector<int> &offsets) override;

    virtual void communicate_chunks(const std::vector<Chunk> &chunks,
                                    const std::vector<int> &dest_rank,
                                    const std::vector<int> &dest_domain,
                                    const std::vector<int> &offsets,
                                    std::vector<Chunk> &chunks_to_assemble,
                                    std::vector<int> &chunks_to_assemble_domains,
                                    std::vector<int> &chunks_to_assemble_gids) override;

    virtual void get_prelb_adjset_maps(const std::vector<int>& chunk_offsets,
                                       const DomainToChunkMap& chunks,
                                       const std::map<index_t, const Node*>& domain_map,
                                       std::vector<Node>& adjset_chunk_maps);

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
    std::vector<int64> domain_to_rank_map;
};

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
#endif
