// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh_distribute.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_mpi_mesh.hpp"
#include "conduit_blueprint.hpp"

#include "conduit_relay_mpi.hpp"
#include <mpi.h>

#include "conduit_fmt/conduit_fmt.h"

#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <unordered_map>

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
void 
distribute(const conduit::Node &mesh,
           const conduit::Node &options,
           conduit::Node &output,
           MPI_Comm comm)
{
    ///
    /// Make sure options["domain_map"] exists
    ///

    if(!options.has_child("domain_map"))
    {
            CONDUIT_ERROR("Options[\"domain_map\"] one-to-many relation "
                          "is missing");
    }

    /// TODO? Make sure domain_map is an o2mrelation?
    
    Node domain_map;
    domain_map.set_external(options["domain_map"]);

    // if we have sizes but not offsets, generate them b/c the o2m iterator
    // won't function w/o them
    if (domain_map.has_child("sizes") && !domain_map.has_child("offsets"))
    {
        Node info;
        if(!conduit::blueprint::o2mrelation::generate_offsets(domain_map,
                                                              info))
        {
            CONDUIT_ERROR("Failed to generate_offsets for "
                          "options[\"domain_map\"] one-to-many relation "
                          << info.to_yaml());
        }
    }

    // get par_rank
    int par_rank = conduit::relay::mpi::rank(comm);

    // gen domain to rank map
    // note: this only handles non domain overloaded inputs cases
    Node d2r_map;
    conduit::blueprint::mpi::mesh::generate_domain_to_rank_map(mesh,
                                                               d2r_map,
                                                               comm);

    index_t_accessor d2r_vals = d2r_map.value();

    // local domain pointers
    std::vector<const Node *> local_domains = ::conduit::blueprint::mesh::domains(mesh);
    // map from domain id to local domain index
    std::map<index_t,index_t> local_domain_ids;

    // populate domain id to local domain index map
    for(index_t local_domain_idx = 0;
        local_domain_idx < (index_t)local_domains.size();
        local_domain_idx++)
    {
        index_t domain_id = par_rank;
        const conduit::Node *domain = local_domains[local_domain_idx];
        // see if we have read domain  id
        if(domain->has_child("state") && domain->fetch("state").has_child("domain_id"))
        {
            domain_id = domain->fetch("state/domain_id").to_index_t();
        }

        local_domain_ids[domain_id] = local_domain_idx;
    }

    // clear output mesh
    output.reset();

    // domain map is an o2m
    // full walk the map to queue sends and recvs
    blueprint::o2mrelation::O2MIterator o2m_iter(domain_map);
    // future o2m interface?
    // O2MMap o2m_rel(domain_map);
    index_t_accessor dmap_values = domain_map["values"].value();

    conduit::relay::mpi::communicate_using_schema comm_using_schema(comm);
    // comm_using_schema.set_logging(true);
    int tag = 422000; // unique tag start for distribute

    // full walk
    while(o2m_iter.has_next(conduit::blueprint::o2mrelation::DATA))
    // for( int i = 0; i < o2m_rel.size(); i++)
    {
        int i = o2m_iter.next(conduit::blueprint::o2mrelation::ONE);
        // i is domain id
        // check if we have domain w/ domain id == i
        // if so we will send
        bool have_domain = local_domain_ids.find(i) != local_domain_ids.end();
        // loop over all dests for domain i
        o2m_iter.to_front(conduit::blueprint::o2mrelation::MANY);
        while(o2m_iter.has_next(conduit::blueprint::o2mrelation::MANY))
        // for (int j = 0; j < o2m_rel.size(i); j++)
        {
            o2m_iter.next(conduit::blueprint::o2mrelation::MANY);
            // index_t o2m_idx = o2m_rel.map(i,j);
            index_t o2m_idx  = o2m_iter.index(conduit::blueprint::o2mrelation::DATA);
            index_t des_rank = dmap_values[o2m_idx];

            if(have_domain)
            {
                const Node &send_dom = *local_domains[local_domain_ids[i]];
                if(par_rank == des_rank)
                {
                    // debug info
                    // std::cout << "rank " << par_rank << " self send domain "
                    //           << i <<std::endl;
                    // self send ... simply copy out
                    output.append().set(send_dom);
                }
                else
                {
                    // debug info
                    // std::cout << "rank " << par_rank << " qsend domain " << i
                    //           << " to " << des_rank
                    //           << " tag = " << tag << std::endl;
                    // queue send of domain
                    comm_using_schema.add_isend(send_dom,des_rank,tag);
                }
            }
            else if(par_rank == des_rank)
            {
                // look up who is sending
                index_t send_rank = d2r_vals[i];
                // queue recv of domain
                // debug info
                // std::cout << "rank " << par_rank << " qrecv domain "
                //           << i << " from " << send_rank
                //           << " tag = " << tag << std::endl;
                Node &res_domain = output.append();
                comm_using_schema.add_irecv(res_domain,send_rank,tag);
            }
            //this count allows each pair to have a unique tag
            tag++;
        }
    }
    comm_using_schema.execute();

}



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

