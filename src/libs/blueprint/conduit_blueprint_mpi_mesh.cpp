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

// typedefs for verbose but commonly used types
typedef std::tuple<conduit::Node*, conduit::Node*, conduit::Node*> DomMapsTuple;
typedef std::tuple<conduit::float64, conduit::float64, conduit::float64> PointTuple;
// typedefs to enable passing around function pointers
typedef void (*GenDerivedFun)(const conduit::Node&, conduit::Node&, conduit::Node&, conduit::Node&);
typedef void (*GenDecomposedFun)(const conduit::Node&, conduit::Node&, conduit::Node&, conduit::Node&, conduit::Node&);

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
//
// This function is responsible for collecting the domains within the given mesh
// with subnodes of the given maps based on the domain's path, e.g.:
//
// input:
//   mesh: {"domain0": {/*1*/}, "domain1": {/*2*/}}
//   s2dmap: {}
//   d2smap: {}
//
// output:
//   mesh: {"domain0": {/*1*/}, "domain1": {/*2*/}}
//   s2dmap: {"domain0": {/*A*/}, "domain1": {/*B*/}}
//   d2smap: {"domain0": {/*a*/}, "domain1": {/*b*/}}
//   return: [<&1, &A, &a>, <&2, &B, &b>]
//
std::vector<DomMapsTuple>
group_domains_and_maps(conduit::Node &mesh, conduit::Node &s2dmap, conduit::Node &d2smap)
{
    std::vector<DomMapsTuple> doms_and_maps;

    s2dmap.reset();
    d2smap.reset();

    if(!conduit::blueprint::mesh::is_multi_domain(mesh))
    {
        doms_and_maps.emplace_back(&mesh, &s2dmap, &d2smap);
    }
    else
    {
        NodeIterator domains_it = mesh.children();
        while(domains_it.has_next())
        {
            conduit::Node& domain = domains_it.next();
            if(mesh.dtype().is_object())
            {
                doms_and_maps.emplace_back(&domain,
                                           &s2dmap[domain.name()],
                                           &d2smap[domain.name()]);
            }
            else
            {
                doms_and_maps.emplace_back(&domain,
                                           &s2dmap.append(),
                                           &d2smap.append());
            }
        }
    }

    return std::vector<DomMapsTuple>(std::move(doms_and_maps));
}


//-----------------------------------------------------------------------------
void
generate_derived_entities(conduit::Node &mesh,
                          const std::string &src_adjset_name,
                          const std::string &dst_adjset_name,
                          const std::string &dst_topo_name,
                          conduit::Node &s2dmap,
                          conduit::Node &d2smap,
                          GenDerivedFun generate_derived)
{
    Node src_data, dst_data;

    const std::vector<DomMapsTuple> doms_and_maps = group_domains_and_maps(mesh, s2dmap, d2smap);
    const conduit::Node &dom_delegate = *std::get<0>(doms_and_maps.front());
    const bool is_src_assoc_vertex = dom_delegate["adjsets"][src_adjset_name]["association"].as_string() == "vertex";

    { // Error Checking //
        if(!is_src_assoc_vertex)
        {
            CONDUIT_ERROR("<blueprint::mpi::mesh::generate_derived_entities> " <<
                          "Given adjacency set has an unsupported association type 'element.'\n" <<
                          "Supported associations:\n" <<
                          "  'vertex'");
        }

        for(index_t di = 0; di < (index_t)doms_and_maps.size(); di++)
        {
            conduit::Node &domain = *std::get<0>(doms_and_maps[di]);
            conduit::Node info;

            if(!domain["adjsets"].has_child(src_adjset_name))
            {
                CONDUIT_ERROR("<blueprint::mpi::mesh::generate_derived_entities> " <<
                              "Requested source adjacency set '" << src_adjset_name << "' " <<
                              "doesn't exist on domain '" << domain.name() << ".'");
            }

            const conduit::Node &src_adjset = domain["adjsets"][src_adjset_name];
            const Node *src_topo_ptr = bputils::find_reference_node(src_adjset, "topology");
            const Node &src_topo = *src_topo_ptr;
            if(!conduit::blueprint::mesh::topology::unstructured::verify(src_topo, info))
            {
                CONDUIT_ERROR("<blueprint::mpi::mesh::generate_derived_entities> " <<
                              "Requested source topology '" << src_topo.name() << "' " <<
                              "is of unsupported type '" << src_topo["type"].as_string() << ".'\n" <<
                              "Supported types:\n" <<
                              "  'unstructured'");
            }
        }
    }

    for(index_t di = 0; di < (index_t)doms_and_maps.size(); di++)
    {
        conduit::Node &domain = *std::get<0>(doms_and_maps[di]);
        conduit::Node &domain_s2dmap = *std::get<1>(doms_and_maps[di]);
        conduit::Node &domain_d2smap = *std::get<2>(doms_and_maps[di]);

        const conduit::Node &src_adjset = domain["adjsets"][src_adjset_name];
        const Node *src_topo_ptr = bputils::find_reference_node(src_adjset, "topology");
        const Node &src_topo = *src_topo_ptr;

        conduit::Node &dst_topo = domain["topologies"][dst_topo_name];
        generate_derived(src_topo, dst_topo, domain_s2dmap, domain_d2smap);

        conduit::Node &dst_adjset = domain["adjsets"][dst_adjset_name];
        dst_adjset.reset();
        dst_adjset["association"].set("element");
        dst_adjset["topology"].set(dst_topo_name);
    }

    src_data.reset();
    dst_data.reset();
    for(index_t di = 0; di < (index_t)doms_and_maps.size(); di++)
    {
        conduit::Node &domain = *std::get<0>(doms_and_maps[di]);

        const Node *src_topo_ptr = bputils::find_reference_node(domain["adjsets"][src_adjset_name], "topology");
        const Node &src_topo = *src_topo_ptr;
        const Node *src_cset_ptr = bputils::find_reference_node(src_topo, "coordset");
        const Node &src_cset = *src_cset_ptr;

        const Node &dst_topo = domain["topologies"][dst_topo_name];
        const index_t dst_topo_len = bputils::topology::length(dst_topo);

        const conduit::Node &src_adjset_groups = domain["adjsets"][src_adjset_name]["groups"];
        conduit::Node &dst_adjset_groups = domain["adjsets"][dst_adjset_name]["groups"];

        for(const std::string &group_name : src_adjset_groups.child_names())
        {
            const conduit::Node &src_group = src_adjset_groups[group_name];
            const conduit::Node &src_values = src_group["values"];

            // given the list of entity ids, we need to sort these ids based on the local
            //   sorting algorithm (uniform within an entity type)
            //
            // result: list of entity ids that constitute this group (std::set<index_t>)
            std::set<index_t> group_pidxs;
            for(index_t ei = 0; ei < src_values.dtype().number_of_elements(); ei++)
            {
                std::vector<index_t> entry_pidxs;
                if(is_src_assoc_vertex)
                {
                    // NOTE(JRC): This won't work if there is an indirection scheme
                    // on the source group's "values" array, but this shouldn't
                    // currently be allowed anyway.
                    src_data.set_external(DataType(src_values.dtype().id(), 1),
                        (void*)src_values.element_ptr(ei));
                    entry_pidxs.push_back(src_data.to_index_t());
                }
                else
                {
                    entry_pidxs = bputils::topology::unstructured::points(src_topo, ei);
                }
                group_pidxs.insert(entry_pidxs.begin(), entry_pidxs.end());
            }

            // given the list of vertex ids, we need to get the list of entities that
            //   are completely contained within the group
            //
            // result: list of entity ids that constitute this group (std::set<index_t>)
            std::vector<std::tuple<std::set<PointTuple>, index_t>> group_entities;
            for(index_t ei = 0; ei < dst_topo_len; ei++)
            {
                std::vector<index_t> entity_pidxs = bputils::topology::unstructured::points(dst_topo, ei);

                bool entity_in_group = true;
                for(index_t pi = 0; pi < (index_t)entity_pidxs.size() && entity_in_group; pi++)
                {
                    entity_in_group &= group_pidxs.find(entity_pidxs[pi]) != group_pidxs.end();
                }

                if(entity_in_group)
                {
                    std::tuple<std::set<PointTuple>, index_t> entity;

                    std::set<PointTuple> &entity_points = std::get<0>(entity);
                    for(const index_t &entity_pidx : entity_pidxs)
                    {
                        const std::vector<float64> point_coords = bputils::coordset::_explicit::coords(
                            src_cset, entity_pidx);
                        entity_points.emplace(
                            point_coords[0],
                            (point_coords.size() > 1) ? point_coords[1] : 0.0,
                            (point_coords.size() > 2) ? point_coords[2] : 0.0);
                    }

                    index_t &entity_id = std::get<1>(entity);
                    entity_id = ei;

                    // NOTE(JRC): Inserting with this method allows this algorithm to sort new
                    // elements as they're generated, rather than as a separate process at the
                    // end (slight optimization overall).
                    auto entity_itr = std::upper_bound(group_entities.begin(), group_entities.end(), entity);
                    group_entities.insert(entity_itr, entity);
                }
            }

            // NOTE(JRC): If there are no entities in the result for this group (e.g. the start
            // group is a single point and we're generating lines), then we omit it from the result.
            if(!group_entities.empty())
            {
                conduit::Node &dst_group = dst_adjset_groups[group_name];

                dst_group["neighbors"].set(src_group["neighbors"]);

                // given the sorted list of entities fully contained in the group, we need
                //   to push this data into the destination array
                //
                conduit::Node &dst_values = dst_group["values"];
                dst_values.set(DataType(src_values.dtype().id(), group_entities.size()));
                for(index_t ei = 0; ei < (index_t)group_entities.size(); ei++)
                {
                    src_data.set_external(DataType::index_t(1), &std::get<1>(group_entities[ei]));
                    dst_data.set_external(DataType(dst_values.dtype().id(), 1),
                        (void*)dst_values.element_ptr(ei));
                    src_data.to_data_type(dst_data.dtype().id(), dst_data);
                }
            }
        }
    }

    // TODO(JRC): Waitall?
}


//-----------------------------------------------------------------------------
void
generate_points(conduit::Node &mesh,
                const std::string &src_adjset_name,
                const std::string &dst_adjset_name,
                const std::string &dst_topo_name,
                conduit::Node &s2dmap,
                conduit::Node &d2smap)
{
    generate_derived_entities(
        mesh, src_adjset_name, dst_adjset_name, dst_topo_name, s2dmap, d2smap,
        conduit::blueprint::mesh::topology::unstructured::generate_points);
}


//-----------------------------------------------------------------------------
void
generate_lines(conduit::Node &mesh,
               const std::string &src_adjset_name,
               const std::string &dst_adjset_name,
               const std::string &dst_topo_name,
               conduit::Node &s2dmap,
               conduit::Node &d2smap)
{
    generate_derived_entities(
        mesh, src_adjset_name, dst_adjset_name, dst_topo_name, s2dmap, d2smap,
        conduit::blueprint::mesh::topology::unstructured::generate_lines);
}


//-----------------------------------------------------------------------------
void
generate_faces(conduit::Node &mesh,
               const std::string& src_adjset_name,
               const std::string& dst_adjset_name,
               const std::string& dst_topo_name,
               conduit::Node& s2dmap,
               conduit::Node& d2smap)
{
    generate_derived_entities(
        mesh, src_adjset_name, dst_adjset_name, dst_topo_name, s2dmap, d2smap,
        conduit::blueprint::mesh::topology::unstructured::generate_faces);
}


//-----------------------------------------------------------------------------
void
generate_decomposed_entities(conduit::Node &/*mesh*/,
                             const std::string &/*src_adjset_name*/,
                             const std::string &/*dst_adjset_name*/,
                             const std::string &/*dst_topo_name*/,
                             conduit::Node &/*s2dmap*/,
                             conduit::Node &/*d2smap*/,
                             GenDecomposedFun /*generate_decomposed*/)
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

