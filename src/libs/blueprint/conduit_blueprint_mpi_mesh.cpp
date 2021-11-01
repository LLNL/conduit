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
#include <cmath>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_mpi_mesh.hpp"
#include "conduit_blueprint_mpi_mesh_partition.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_o2mrelation_iterator.hpp"
#include "conduit_relay_mpi.hpp"

#include "conduit_blueprint_mpi_mesh.hpp"

// TODO(JRC): Consider moving an improved version of this type to a more accessible location.
struct ffloat64
{
    conduit::float64 data;

    ffloat64(conduit::float64 input = 0.0)
    {
        data = input;
    }

    operator conduit::float64() const
    {
        return this->data;
    }

    bool operator<(const ffloat64 &other) const
    {
        return this->data < other.data && std::abs(this->data - other.data) > CONDUIT_EPSILON;
    }
};

// access conduit blueprint mesh utilities
namespace bputils = conduit::blueprint::mesh::utils;
// access one-to-many index types
namespace O2MIndex = conduit::blueprint::o2mrelation;

// typedefs for verbose but commonly used types
typedef std::tuple<conduit::Node*, conduit::Node*, conduit::Node*> DomMapsTuple;
typedef std::tuple<ffloat64, ffloat64, ffloat64> PointTuple;
// typedefs to enable passing around function pointers
typedef void (*GenDerivedFun)(const conduit::Node&, conduit::Node&, conduit::Node&, conduit::Node&);
typedef void (*GenDecomposedFun)(const conduit::Node&, conduit::Node&, conduit::Node&, conduit::Node&, conduit::Node&);
typedef conduit::index_t (*IdDecomposedFun)(const bputils::TopologyMetadata&, const conduit::index_t /*ei*/, const conduit::index_t /*di*/);
typedef std::vector<conduit::index_t> (*CalcDimDecomposedFun)(const bputils::ShapeType&);

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
void
partition(const conduit::Node &n_mesh,
          const conduit::Node &options,
          conduit::Node &output,
          MPI_Comm comm)
{
    ParallelPartitioner p(comm);
    output.reset();

    // Partitioners on different ranks ought to return the same value but
    // perhaps some did not when they examined their own domains against
    // selection.
    int iinit, ginit;
    iinit = p.initialize(n_mesh, options) ? 1 : 0;
    MPI_Allreduce(&iinit, &ginit, 1, MPI_INT, MPI_MAX, comm);
    if(ginit > 0)
    {
        p.split_selections();
       p.execute(output);
    }
}

//-----------------------------------------------------------------------------
void
find_delegate_domain(const conduit::Node &n,
                     conduit::Node &domain,
                     MPI_Comm comm)
{
    const index_t par_rank = relay::mpi::rank(comm);
    const index_t par_size = relay::mpi::size(comm);

    const std::vector<const Node *> domains = ::conduit::blueprint::mesh::domains(n);
    const index_t local_ind = domains.empty() ? 0 : 1;

    std::vector<int64> local_ind_list(par_size, 0);
    local_ind_list[par_rank] = local_ind;

    Node local_ind_node, global_ind_node;
    local_ind_node.set_external(&local_ind_list[0], local_ind_list.size());
    relay::mpi::sum_all_reduce(local_ind_node, global_ind_node, comm);

    Node temp;
    index_t min_delegate_rank = -1;
    for(index_t ri = 0; ri < par_size && min_delegate_rank == -1; ri++)
    {
        temp.set_external(DataType(global_ind_node.dtype().id(), 1),
            global_ind_node.element_ptr(ri));

        const index_t rank_ind = temp.to_index_t();
        if(rank_ind == 1)
        {
            min_delegate_rank = ri;
        }
    }

    if(min_delegate_rank != -1)
    {
        if(par_rank == min_delegate_rank)
        {
            // TODO(JRC): Consider using a more consistent evaluation for the
            // "first" domain (e.g. the domain with the lowest "state/domain_id"
            // value).
            domain.set(*domains[0]);
        }
        relay::mpi::broadcast_using_schema(domain, min_delegate_rank, comm);
    }
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
std::vector<index_t>
calculate_decomposed_dims(const conduit::Node &mesh, const std::string &adjset_name, CalcDimDecomposedFun calc_dims)
{
    // NOTE(JRC): This strategy works even if some ranks have empty meshes because
    // the empty ranks won't use the resulting 'dims' array to compute indices and
    // thus it doesn't matter if the variable has the technically incorrect value.
    const std::vector<const Node *> domains = ::conduit::blueprint::mesh::domains(mesh);
    if(domains.empty())
    {
        return std::vector<index_t>();
    }
    else // if(!domains.empty())
    {
        const Node &domain = *domains.front();

        const Node &adjset = domain["adjsets"][adjset_name];
        const Node *topo_ptr = bputils::find_reference_node(adjset, "topology");
        const Node &topo = *topo_ptr;

        const bputils::ShapeType shape(topo);
        return calc_dims(shape);
    }
}


//-----------------------------------------------------------------------------
void
verify_generate_mesh(const conduit::Node &mesh,
                     const std::string &adjset_name)
{
    const std::vector<const Node *> domains = blueprint::mesh::domains(mesh);
    for(index_t di = 0; di < (index_t)domains.size(); di++)
    {
        const Node &domain = *domains[di];
        Node info;

        if(!domain["adjsets"].has_child(adjset_name))
        {
            CONDUIT_ERROR("<blueprint::mpi::mesh::generate_*> " <<
                          "Requested source adjacency set '" << adjset_name << "' " <<
                          "doesn't exist on domain '" << domain.name() << ".'");
        }

        if(domain["adjsets"][adjset_name]["association"].as_string() != "vertex")
        {
            CONDUIT_ERROR("<blueprint::mpi::mesh::generate_*> " <<
                          "Given adjacency set has an unsupported association type 'element.'\n" <<
                          "Supported associations:\n" <<
                          "  'vertex'");
        }

        const Node &adjset = domain["adjsets"][adjset_name];
        const Node *topo_ptr = bputils::find_reference_node(adjset, "topology");
        const Node &topo = *topo_ptr;
        if(!conduit::blueprint::mesh::topology::unstructured::verify(topo, info))
        {
            CONDUIT_ERROR("<blueprint::mpi::mesh::generate_*> " <<
                          "Requested source topology '" << topo.name() << "' " <<
                          "is of unsupported type '" << topo["type"].as_string() << ".'\n" <<
                          "Supported types:\n" <<
                          "  'unstructured'");
        }
    }
}


//-----------------------------------------------------------------------------
void
generate_derived_entities(conduit::Node &mesh,
                          const std::string &src_adjset_name,
                          const std::string &dst_adjset_name,
                          const std::string &dst_topo_name,
                          conduit::Node &s2dmap,
                          conduit::Node &d2smap,
                          MPI_Comm /*comm*/,
                          GenDerivedFun generate_derived)
{
    const std::vector<DomMapsTuple> doms_and_maps = group_domains_and_maps(mesh, s2dmap, d2smap);
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

    Node src_data, dst_data;
    for(index_t di = 0; di < (index_t)doms_and_maps.size(); di++)
    {
        conduit::Node &domain = *std::get<0>(doms_and_maps[di]);
        const index_t domain_id = domain["state/domain_id"].to_index_t();

        const Node *src_topo_ptr = bputils::find_reference_node(domain["adjsets"][src_adjset_name], "topology");
        const Node &src_topo = *src_topo_ptr;
        const Node *src_cset_ptr = bputils::find_reference_node(src_topo, "coordset");
        const Node &src_cset = *src_cset_ptr;

        const Node &dst_topo = domain["topologies"][dst_topo_name];
        const index_t dst_topo_len = bputils::topology::length(dst_topo);

        const conduit::Node &src_adjset_groups = domain["adjsets"][src_adjset_name]["groups"];
        conduit::Node &dst_adjset_groups = domain["adjsets"][dst_adjset_name]["groups"];

        // Organize Adjset Points into Interfaces (Pair-Wise Groups) //

        // {(neighbor domain id): <(participating points for domain interface)>}
        std::map<index_t, std::set<index_t>> neighbor_pidxs_map;
        for(const std::string &group_name : src_adjset_groups.child_names())
        {
            const conduit::Node &src_group = src_adjset_groups[group_name];
            const conduit::Node &src_neighbors = src_group["neighbors"];
            const conduit::Node &src_values = src_group["values"];

            for(index_t ni = 0; ni < src_neighbors.dtype().number_of_elements(); ni++)
            {
                src_data.set_external(DataType(src_neighbors.dtype().id(), 1),
                    (void*)src_neighbors.element_ptr(ni));
                std::set<index_t> &neighbor_pidxs = neighbor_pidxs_map[src_data.to_index_t()];
                for(index_t pi = 0; pi < src_values.dtype().number_of_elements(); pi++)
                {
                    src_data.set_external(DataType(src_values.dtype().id(), 1),
                        (void*)src_values.element_ptr(pi));
                    neighbor_pidxs.insert(src_data.to_index_t());
                }
            }
        }

        // Collect Viable Entities for All Interfaces //

        // {(entity id in topology): <(neighbor domain ids that contain this entity)>}
        std::map<index_t, std::set<index_t>> entity_neighbor_map;
        for(index_t ei = 0; ei < dst_topo_len; ei++)
        {
            std::vector<index_t> entity_pidxs = bputils::topology::unstructured::points(dst_topo, ei);
            for(const auto &neighbor_pair : neighbor_pidxs_map)
            {
                const index_t &ni = neighbor_pair.first;
                const std::set<index_t> &neighbor_pidxs = neighbor_pair.second;

                bool entity_in_neighbor = true;
                for(index_t pi = 0; pi < (index_t)entity_pidxs.size() && entity_in_neighbor; pi++)
                {
                    entity_in_neighbor &= neighbor_pidxs.find(entity_pidxs[pi]) != neighbor_pidxs.end();
                }

                if(entity_in_neighbor)
                {
                    entity_neighbor_map[ei].insert(ni);
                }
            }
        }

        // Use Entity Interfaces to Construct Group Entity Lists //

        std::map<std::set<index_t>, std::vector<std::tuple<std::set<PointTuple>, index_t>>> group_entity_map;
        for(const auto &entity_neighbor_pair : entity_neighbor_map)
        {
            const index_t &ei = entity_neighbor_pair.first;
            const std::set<index_t> &entity_neighbors = entity_neighbor_pair.second;
            std::tuple<std::set<PointTuple>, index_t> entity;

            std::vector<index_t> entity_pidxs = bputils::topology::unstructured::points(dst_topo, ei);
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
            std::vector<std::tuple<std::set<PointTuple>, index_t>> &group_entities =
                group_entity_map[entity_neighbors];
            auto entity_itr = std::upper_bound(group_entities.begin(), group_entities.end(), entity);
            group_entities.insert(entity_itr, entity);
        }

        for(const auto &group_pair : group_entity_map)
        {
            // NOTE(JRC): It's possible for the 'src_adjset_groups' node to be empty,
            // so we only want to query child data types if we know there is at least
            // 1 non-empty group.
            const conduit::DataType src_neighbors_dtype = src_adjset_groups.child(0)["neighbors"].dtype();
            const conduit::DataType src_values_dtype = src_adjset_groups.child(0)["values"].dtype();

            const std::set<index_t> &group_nidxs = group_pair.first;
            const std::vector<std::tuple<std::set<PointTuple>, index_t>> &group_entities = group_pair.second;
            std::string group_name;
            {
                // NOTE(JRC): The current domain is included in the domain name so that
                // it matches across all domains and processors (also, using std::set
                // ensures that items are sorted and the order is the same across ranks).
                std::set<index_t> group_all_nidxs = group_nidxs;
                group_all_nidxs.insert(domain_id);

                std::ostringstream oss;
                oss << "group";
                for(const index_t &group_nidx : group_all_nidxs)
                {
                    oss << "_" << group_nidx;
                }
                group_name = oss.str();
            }

            conduit::Node &dst_group = dst_adjset_groups[group_name];
            conduit::Node &dst_neighbors = dst_group["neighbors"];
            conduit::Node &dst_values = dst_group["values"];

            dst_neighbors.set(DataType(src_neighbors_dtype.id(), group_nidxs.size()));
            index_t ni = 0;
            for(auto nitr = group_nidxs.begin(); nitr != group_nidxs.end(); ++nitr)
            {
                src_data.set_external(DataType::index_t(1),
                    (void*)&(*nitr));
                dst_data.set_external(DataType(src_neighbors_dtype.id(), 1),
                    (void*)dst_neighbors.element_ptr(ni++));
                src_data.to_data_type(dst_data.dtype().id(), dst_data);
            }

            dst_values.set(DataType(src_values_dtype.id(), group_entities.size()));
            for(index_t ei = 0; ei < (index_t)group_entities.size(); ei++)
            {
                src_data.set_external(DataType::index_t(1),
                    (void*)&std::get<1>(group_entities[ei]));
                dst_data.set_external(DataType(src_values_dtype.id(), 1),
                    (void*)dst_values.element_ptr(ei));
                src_data.to_data_type(dst_data.dtype().id(), dst_data);
            }
        }
    }

    // TODO(JRC): Waitall?
}


//-----------------------------------------------------------------------------
void
generate_decomposed_entities(conduit::Node &mesh,
                             const std::string &src_adjset_name,
                             const std::string &dst_adjset_name,
                             const std::string &dst_topo_name,
                             const std::string &dst_cset_name,
                             conduit::Node &s2dmap,
                             conduit::Node &d2smap,
                             MPI_Comm /*comm*/,
                             GenDecomposedFun generate_decomposed,
                             IdDecomposedFun identify_decomposed,
                             const std::vector<index_t> &decomposed_centroid_dims)
{
    const std::vector<DomMapsTuple> doms_and_maps = group_domains_and_maps(mesh, s2dmap, d2smap);
    for(index_t di = 0; di < (index_t)doms_and_maps.size(); di++)
    {
        Node &domain = *std::get<0>(doms_and_maps[di]);
        Node &domain_s2dmap = *std::get<1>(doms_and_maps[di]);
        Node &domain_d2smap = *std::get<2>(doms_and_maps[di]);

        const Node &src_adjset = domain["adjsets"][src_adjset_name];
        const Node *src_topo_ptr = bputils::find_reference_node(src_adjset, "topology");
        const Node &src_topo = *src_topo_ptr;

        // NOTE(JRC): Diff- new code below
        Node &dst_topo = domain["topologies"][dst_topo_name];
        Node &dst_cset = domain["coordsets"][dst_cset_name];
        generate_decomposed(src_topo, dst_topo, dst_cset, domain_s2dmap, domain_d2smap);

        Node &dst_adjset = domain["adjsets"][dst_adjset_name];
        dst_adjset.reset();
        // NOTE(JRC): Diff- different association (decomposed entity -> assoc: vertex)
        dst_adjset["association"].set("vertex");
        dst_adjset["topology"].set(dst_topo_name);
    }

    Node src_data, dst_data;
    src_data.reset();
    dst_data.reset();
    for(index_t di = 0; di < (index_t)doms_and_maps.size(); di++)
    {
        Node &domain = *std::get<0>(doms_and_maps[di]);
        const index_t domain_id = domain["state/domain_id"].to_index_t();

        const Node *src_topo_ptr = bputils::find_reference_node(domain["adjsets"][src_adjset_name], "topology");
        const Node &src_topo = *src_topo_ptr;
        const Node *src_cset_ptr = bputils::find_reference_node(src_topo, "coordset");
        const Node &src_cset = *src_cset_ptr;
        // NOTE(JRC): Diff- generate topology metadata for source topology to find
        // centroids that may exist within an adjset group.
        const bputils::TopologyMetadata src_topo_data(src_topo, src_cset);

        // const Node &dst_topo = domain["topologies"][dst_topo_name];
        const Node &dst_cset = domain["coordsets"][dst_cset_name];

        const Node &src_adjset_groups = domain["adjsets"][src_adjset_name]["groups"];
        Node &dst_adjset_groups = domain["adjsets"][dst_adjset_name]["groups"];

        // Organize Adjset Points into Interfaces (Pair-Wise Groups) //

        // {(neighbor domain id): <(participating points for domain interface)>}
        std::map<index_t, std::set<index_t>> neighbor_pidxs_map;
        for(const std::string &group_name : src_adjset_groups.child_names())
        {
            const Node &src_group = src_adjset_groups[group_name];
            const Node &src_neighbors = src_group["neighbors"];
            const Node &src_values = src_group["values"];

            for(index_t ni = 0; ni < src_neighbors.dtype().number_of_elements(); ni++)
            {
                src_data.set_external(DataType(src_neighbors.dtype().id(), 1),
                    (void*)src_neighbors.element_ptr(ni));
                std::set<index_t> &neighbor_pidxs = neighbor_pidxs_map[src_data.to_index_t()];
                for(index_t pi = 0; pi < src_values.dtype().number_of_elements(); pi++)
                {
                    src_data.set_external(DataType(src_values.dtype().id(), 1),
                        (void*)src_values.element_ptr(pi));
                    neighbor_pidxs.insert(src_data.to_index_t());
                }
            }
        }

        // Collect Viable Entities for All Interfaces //

        // {(entity centroid id): <(neighbor domain ids that contain this entity)>}
        std::map<index_t, std::set<index_t>> entity_neighbor_map;
        // NOTE(JRC): Diff, entirely different iteration strategy for finding entities
        // to consider on individual adjset interfaces.
        for(const index_t &di : decomposed_centroid_dims)
        {
            const Node &dim_topo = src_topo_data.dim_topos[di];
            for(index_t ei = 0; ei < src_topo_data.get_length(di); ei++)
            {
                std::vector<index_t> entity_pidxs = bputils::topology::unstructured::points(dim_topo, ei);
                for(const auto &neighbor_pair : neighbor_pidxs_map)
                {
                    const index_t &ni = neighbor_pair.first;
                    const std::set<index_t> &neighbor_pidxs = neighbor_pair.second;

                    bool entity_in_neighbor = true;
                    for(index_t pi = 0; pi < (index_t)entity_pidxs.size() && entity_in_neighbor; pi++)
                    {
                        entity_in_neighbor &= neighbor_pidxs.find(entity_pidxs[pi]) != neighbor_pidxs.end();
                    }

                    if(entity_in_neighbor)
                    {
                        const index_t entity_cidx = identify_decomposed(src_topo_data, ei, di);
                        entity_neighbor_map[entity_cidx].insert(ni);
                    }
                }
            }
        }

        // Use Entity Interfaces to Construct Group Entity Lists //

        std::map<std::set<index_t>, std::vector<std::tuple<std::set<PointTuple>, index_t>>> group_entity_map;
        for(const auto &entity_neighbor_pair : entity_neighbor_map)
        {
            const index_t &entity_cidx = entity_neighbor_pair.first;
            const std::set<index_t> &entity_neighbors = entity_neighbor_pair.second;
            std::tuple<std::set<PointTuple>, index_t> entity;

            std::set<PointTuple> &entity_points = std::get<0>(entity);
            // NOTE(JRC): Diff: Substitute entity for centroid point at the end here.
            const std::vector<float64> point_coords = bputils::coordset::_explicit::coords(
                dst_cset, entity_cidx);
            entity_points.emplace(
                point_coords[0],
                (point_coords.size() > 1) ? point_coords[1] : 0.0,
                (point_coords.size() > 2) ? point_coords[2] : 0.0);

            index_t &entity_id = std::get<1>(entity);
            entity_id = entity_cidx;

            // NOTE(JRC): Inserting with this method allows this algorithm to sort new
            // elements as they're generated, rather than as a separate process at the
            // end (slight optimization overall).
            std::vector<std::tuple<std::set<PointTuple>, index_t>> &group_entities =
                group_entity_map[entity_neighbors];
            auto entity_itr = std::upper_bound(group_entities.begin(), group_entities.end(), entity);
            group_entities.insert(entity_itr, entity);
        }

        for(const auto &group_pair : group_entity_map)
        {
            // NOTE(JRC): It's possible for the 'src_adjset_groups' node to be empty,
            // so we only want to query child data types if we know there is at least
            // 1 non-empty group.
            const DataType src_neighbors_dtype = src_adjset_groups.child(0)["neighbors"].dtype();
            const DataType src_values_dtype = src_adjset_groups.child(0)["values"].dtype();

            const std::set<index_t> &group_nidxs = group_pair.first;
            const std::vector<std::tuple<std::set<PointTuple>, index_t>> &group_entities = group_pair.second;
            std::string group_name;
            {
                // NOTE(JRC): The current domain is included in the domain name so that
                // it matches across all domains and processors (also, using std::set
                // ensures that items are sorted and the order is the same across ranks).
                std::set<index_t> group_all_nidxs = group_nidxs;
                group_all_nidxs.insert(domain_id);

                std::ostringstream oss;
                oss << "group";
                for(const index_t &group_nidx : group_all_nidxs)
                {
                    oss << "_" << group_nidx;
                }
                group_name = oss.str();
            }

            Node &dst_group = dst_adjset_groups[group_name];
            Node &dst_neighbors = dst_group["neighbors"];
            Node &dst_values = dst_group["values"];

            dst_neighbors.set(DataType(src_neighbors_dtype.id(), group_nidxs.size()));
            index_t ni = 0;
            for(auto nitr = group_nidxs.begin(); nitr != group_nidxs.end(); ++nitr)
            {
                src_data.set_external(DataType::index_t(1),
                    (void*)&(*nitr));
                dst_data.set_external(DataType(src_neighbors_dtype.id(), 1),
                    (void*)dst_neighbors.element_ptr(ni++));
                src_data.to_data_type(dst_data.dtype().id(), dst_data);
            }

            dst_values.set(DataType(src_values_dtype.id(), group_entities.size()));
            for(index_t ei = 0; ei < (index_t)group_entities.size(); ei++)
            {
                src_data.set_external(DataType::index_t(1),
                    (void*)&std::get<1>(group_entities[ei]));
                dst_data.set_external(DataType(src_values_dtype.id(), 1),
                    (void*)dst_values.element_ptr(ei));
                src_data.to_data_type(dst_data.dtype().id(), dst_data);
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
                conduit::Node &d2smap,
                MPI_Comm comm)
{
    verify_generate_mesh(mesh, src_adjset_name);
    generate_derived_entities(
        mesh, src_adjset_name, dst_adjset_name, dst_topo_name, s2dmap, d2smap, comm,
        conduit::blueprint::mesh::topology::unstructured::generate_points);
}


//-----------------------------------------------------------------------------
void
generate_lines(conduit::Node &mesh,
               const std::string &src_adjset_name,
               const std::string &dst_adjset_name,
               const std::string &dst_topo_name,
               conduit::Node &s2dmap,
               conduit::Node &d2smap,
               MPI_Comm comm)
{
    verify_generate_mesh(mesh, src_adjset_name);
    generate_derived_entities(
        mesh, src_adjset_name, dst_adjset_name, dst_topo_name, s2dmap, d2smap, comm,
        conduit::blueprint::mesh::topology::unstructured::generate_lines);
}


//-----------------------------------------------------------------------------
void
generate_faces(conduit::Node &mesh,
               const std::string& src_adjset_name,
               const std::string& dst_adjset_name,
               const std::string& dst_topo_name,
               conduit::Node& s2dmap,
               conduit::Node& d2smap,
               MPI_Comm comm)
{
    verify_generate_mesh(mesh, src_adjset_name);
    generate_derived_entities(
        mesh, src_adjset_name, dst_adjset_name, dst_topo_name, s2dmap, d2smap, comm,
        conduit::blueprint::mesh::topology::unstructured::generate_faces);
}


//-----------------------------------------------------------------------------
void
generate_centroids(conduit::Node& mesh,
                   const std::string& src_adjset_name,
                   const std::string& dst_adjset_name,
                   const std::string& dst_topo_name,
                   const std::string& dst_cset_name,
                   conduit::Node& s2dmap,
                   conduit::Node& d2smap,
                   MPI_Comm comm)
{
    const static auto identify_centroid = []
        (const bputils::TopologyMetadata &/*topo_data*/, const index_t ei, const index_t /*di*/)
    {
        return ei;
    };

    const static auto calculate_centroid_dims = [] (const bputils::ShapeType &topo_shape)
    {
        return std::vector<index_t>(1, topo_shape.dim);
    };

    verify_generate_mesh(mesh, src_adjset_name);

    const std::vector<index_t> centroid_dims = calculate_decomposed_dims(
        mesh, src_adjset_name, calculate_centroid_dims);

    generate_decomposed_entities(
        mesh, src_adjset_name, dst_adjset_name, dst_topo_name, dst_cset_name, s2dmap, d2smap, comm,
        conduit::blueprint::mesh::topology::unstructured::generate_centroids, identify_centroid, centroid_dims);
}


//-----------------------------------------------------------------------------
void
generate_sides(conduit::Node& mesh,
               const std::string& src_adjset_name,
               const std::string& dst_adjset_name,
               const std::string& dst_topo_name,
               const std::string& dst_cset_name,
               conduit::Node& s2dmap,
               conduit::Node& d2smap,
               MPI_Comm comm)
{
    const static auto identify_side = []
        (const bputils::TopologyMetadata &topo_data, const index_t ei, const index_t di)
    {
        index_t doffset = 0;
        for(index_t dii = 0; dii < di; dii++)
        {
            if(dii != 1)
            {
                doffset += topo_data.get_length(dii);
            }
        }

        return doffset + ei;
    };

    const static auto calculate_side_dims = [] (const bputils::ShapeType &topo_shape)
    {
        std::vector<index_t> side_dims;

        side_dims.push_back(0);
        if(topo_shape.dim == 3)
        {
            side_dims.push_back(2);
        }

        return std::vector<index_t>(std::move(side_dims));
    };

    verify_generate_mesh(mesh, src_adjset_name);

    const std::vector<index_t> side_dims = calculate_decomposed_dims(
        mesh, src_adjset_name, calculate_side_dims);

    generate_decomposed_entities(
        mesh, src_adjset_name, dst_adjset_name, dst_topo_name, dst_cset_name, s2dmap, d2smap, comm,
        conduit::blueprint::mesh::topology::unstructured::generate_sides, identify_side, side_dims);
}


//-----------------------------------------------------------------------------
void
generate_corners(conduit::Node& mesh,
                 const std::string& src_adjset_name,
                 const std::string& dst_adjset_name,
                 const std::string& dst_topo_name,
                 const std::string& dst_cset_name,
                 conduit::Node& s2dmap,
                 conduit::Node& d2smap,
                 MPI_Comm comm)
{
    const static auto identify_corner = []
        (const bputils::TopologyMetadata &topo_data, const index_t ei, const index_t di)
    {
        index_t doffset = 0;
        for(index_t dii = 0; dii < di; dii++)
        {
            doffset += topo_data.get_length(dii);
        }

        return doffset + ei;
    };

    const static auto calculate_corner_dims = [] (const bputils::ShapeType &topo_shape)
    {
        std::vector<index_t> corner_dims;

        for(index_t di = 0; di < topo_shape.dim; di++)
        {
            corner_dims.push_back(di);
        }

        return std::vector<index_t>(std::move(corner_dims));
    };

    verify_generate_mesh(mesh, src_adjset_name);

    const std::vector<index_t> corner_dims = calculate_decomposed_dims(
        mesh, src_adjset_name, calculate_corner_dims);

    generate_decomposed_entities(
        mesh, src_adjset_name, dst_adjset_name, dst_topo_name, dst_cset_name, s2dmap, d2smap, comm,
        conduit::blueprint::mesh::topology::unstructured::generate_corners, identify_corner, corner_dims);
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
