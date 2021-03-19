// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <algorithm>
#include <tuple>
#include <vector>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_o2mrelation_iterator.hpp"
#include "conduit_relay_mpi.hpp"

#include "conduit_blueprint_mpi_mesh.hpp"

// access conduit blueprint mesh utilities
namespace bputils = conduit::blueprint::mesh::utils;
// access one-to-many index types
namespace O2MIndex = conduit::blueprint::o2mrelation;

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


//-----------------------------------------------------------------------------
void
generate_points(conduit::Node &mesh,
                const std::string& src_adjset_name,
                const std::string& dst_adjset_name,
                const std::string& dst_topo_name,
                conduit::Node& s2dmap,
                conduit::Node& d2smap)
{
    s2dmap.reset();
    d2smap.reset();

    // NOTE(JRC): We want to be sure we're modifying the structure of the
    // given node, so we need to be careful to use pointers and references.
    std::vector<conduit::Node *> domains, domain_s2dmaps, domain_d2smaps;
    if(!conduit::blueprint::mesh::is_multi_domain(mesh))
    {
        domains.push_back(&mesh);
        domain_s2dmaps.push_back(&s2dmap);
        domain_d2smaps.push_back(&d2smap);
    }
    else
    {
        NodeIterator domains_it = mesh.children();
        while(domains_it.has_next())
        {
            conduit::Node& curr_domain = domains_it.next();
            domains.push_back(&curr_domain);
            // TODO(JRC): Support multi-domain via list also.
            domain_s2dmaps.push_back(&s2dmap[curr_domain.name()]);
            domain_d2smaps.push_back(&d2smap[curr_domain.name()]);
        }
    }

    const bool is_src_assoc_vertex = (*domains.front())["adjsets"][src_adjset_name]["association"].as_string() == "vertex";

    Node src_data, dst_data;
    std::vector<std::map<index_t, index_t>> domain_c2tmaps(domains.size());
    for(index_t di = 0; di < (index_t)domains.size(); di++)
    {
        conduit::Node &domain = *domains[di];
        conduit::Node &domain_s2dmap = *domain_s2dmaps[di];
        conduit::Node &domain_d2smap = *domain_d2smaps[di];

        // TODO(JRC): Add in error handling for:
        // - paths to be sure important paths exist (e.g. adjset isn't empty).
        // - ensure that the topology is unstructured
        const conduit::Node &src_adjset = domain["adjsets"][src_adjset_name];
        const Node *src_topo_ptr = bputils::find_reference_node(src_adjset, "topology");
        const Node &src_topo = *src_topo_ptr;

        conduit::Node &dst_topo = domain["topologies"][dst_topo_name];
        // TODO(JRC): This assumes that the 'src_topo' is an unstructured topology.
        conduit::blueprint::mesh::topology::unstructured::generate_points(
            src_topo, dst_topo, domain_s2dmap, domain_d2smap);

        conduit::Node &dst_adjset = domain["adjsets"][dst_adjset_name];
        dst_adjset.reset();
        dst_adjset["association"].set("element");
        dst_adjset["topology"].set(dst_topo_name);

        if(is_src_assoc_vertex)
        { // generate mapping from coordset to point topology //
            std::map<index_t, index_t> &domain_c2tmap = domain_c2tmaps[di];

            Node &dst_topo_conn = dst_topo["elements/connectivity"];
            for(index_t ti = 0; ti < dst_topo_conn.dtype().number_of_elements(); ti++)
            {
                src_data.set_external(DataType(dst_topo_conn.dtype().id(), 1),
                    dst_topo_conn.element_ptr(ti));
                domain_c2tmap[src_data.to_index_t()] = ti;
            }
        }
    }

    src_data.reset();
    dst_data.reset();
    for(index_t di = 0; di < (index_t)domains.size(); di++)
    {
        conduit::Node &domain = *domains[di];
        std::map<index_t, index_t> &domain_c2tmap = domain_c2tmaps[di];

        const Node *src_topo_ptr = bputils::find_reference_node(domain["adjsets"][src_adjset_name], "topology");
        const Node &src_topo = *src_topo_ptr;
        const Node *src_cset_ptr = bputils::find_reference_node(src_topo, "coordset");
        const Node &src_cset = *src_cset_ptr;

        const conduit::Node &src_adjset_groups = domain["adjsets"][src_adjset_name]["groups"];
        conduit::Node &dst_adjset_groups = domain["adjsets"][dst_adjset_name]["groups"];

        for(const std::string &group_name : src_adjset_groups.child_names())
        {
            const conduit::Node &src_group = src_adjset_groups[group_name];
            conduit::Node &dst_group = dst_adjset_groups[group_name];

            dst_group["neighbors"].set(src_group["neighbors"]);

            const conduit::Node &src_values = src_group["values"];
            conduit::Node &dst_values = dst_group["values"];
            dst_values.set(DataType(src_values.dtype().id(),
                src_values.dtype().number_of_elements()));

            // if the source is 'vertex', the process is a bit easier because we
            // can match coordinate set index w/ topology index fairly easily
            if(is_src_assoc_vertex)
            {
                for(index_t vi = 0; vi < src_values.dtype().number_of_elements(); vi++)
                {
                    src_data.set_external(DataType(src_values.dtype().id(), 1),
                        (void*)src_values.element_ptr(vi));

                    const index_t cset_index = src_data.to_index_t();
                    const index_t topo_index = domain_c2tmap[cset_index];

                    src_data.set_external(DataType::index_t(1),
                        (void*)&topo_index);
                    dst_data.set_external(DataType(dst_values.dtype().id(), 1),
                        (void*)dst_values.element_ptr(vi));
                    src_data.to_data_type(dst_data.dtype().id(), dst_data);
                }
            }
            // if the source is 'element', the process is more challenging;
            // to minimize communication, we just sort positions and we know
            // these orderings will be the same across processors
            else
            {
                std::set<index_t> group_pidxs;
                for(index_t ei = 0; ei < src_values.dtype().number_of_elements(); ei++)
                {
                    std::vector<index_t> entity_pidxs = bputils::topology::unstructured::points(src_topo, ei);
                    group_pidxs.insert(entity_pidxs.begin(), entity_pidxs.end());
                }

                std::vector<std::tuple<float64, float64, float64, index_t>> group_coords;
                for(index_t pi = 0; pi < (index_t)group_pidxs.size(); pi++)
                {
                    std::vector<float64> point_coords = bputils::coordset::_explicit::coords(src_cset, pi);
                    group_coords.emplace_back(
                        point_coords[0],
                        (point_coords.size() > 1) ? point_coords[1] : 0.0,
                        (point_coords.size() > 2) ? point_coords[2] : 0.0,
                        pi);
                }
                std::sort(group_coords.begin(), group_coords.end());

                // now that we have all the points sorted, we just need to extract their
                // index values and put those into the 'dst_values' array in order.
                for(index_t pi = 0; pi < src_values.dtype().number_of_elements(); pi++)
                {
                    src_data.set_external(DataType::index_t(1), &group_coords[pi]);
                    dst_data.set_external(DataType(dst_values.dtype().id(), 1),
                        (void*)dst_values.element_ptr(pi));
                    src_data.to_data_type(dst_data.dtype().id(), dst_data);
                }
            }
        }
    }

    // TODO(JRC): Waitall?
}


//-----------------------------------------------------------------------------
void
generate_lines(conduit::Node &/*mesh*/,
               const std::string& /*src_adjset_name*/,
               const std::string& /*dst_adjset_name*/,
               const std::string& /*dst_topo_name*/,
               conduit::Node& /*s2dmap*/,
               conduit::Node& /*d2smap*/)
{
    // TODO(JRC)
}


//-----------------------------------------------------------------------------
void
generate_faces(conduit::Node &/*mesh*/,
               const std::string& /*src_adjset_name*/,
               const std::string& /*dst_adjset_name*/,
               const std::string& /*dst_topo_name*/,
               conduit::Node& /*s2dmap*/,
               conduit::Node& /*d2smap*/)
{
    // TODO(JRC)
}


//-----------------------------------------------------------------------------
void
generate_centroids(conduit::Node& /*mesh*/,
                   const std::string& /*src_adjset_path*/,
                   const std::string& /*dst_adjset_path*/,
                   const std::string& /*dst_topo_path*/,
                   const std::string& /*dst_cset_path*/,
                   conduit::Node& /*s2dmap*/,
                   conduit::Node& /*d2smap*/)
{
    // TODO(JRC)
}


//-----------------------------------------------------------------------------
void
generate_sides(conduit::Node& /*mesh*/,
               const std::string& /*src_adjset_path*/,
               const std::string& /*dst_adjset_path*/,
               const std::string& /*dst_topo_path*/,
               const std::string& /*dst_cset_path*/,
               conduit::Node& /*s2dmap*/,
               conduit::Node& /*d2smap*/)
{
    // TODO(JRC)
}


//-----------------------------------------------------------------------------
void
generate_corners(conduit::Node& /*mesh*/,
                 const std::string& /*src_adjset_path*/,
                 const std::string& /*dst_adjset_path*/,
                 const std::string& /*dst_topo_path*/,
                 const std::string& /*dst_cset_path*/,
                 conduit::Node& /*s2dmap*/,
                 conduit::Node& /*d2smap*/)
{
    // TODO(JRC)
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

