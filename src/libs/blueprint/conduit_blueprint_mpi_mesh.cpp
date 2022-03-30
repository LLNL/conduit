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
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_blueprint_mpi_mesh.hpp"
#include "conduit_blueprint_mpi_mesh_flatten.hpp"
#include "conduit_blueprint_mpi_mesh_partition.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_o2mrelation_iterator.hpp"
#include "conduit_relay_mpi.hpp"
#include <assert.h>
#include <cmath>
#include <limits>
#include <list>
// access conduit blueprint mesh utilities
namespace bputils = conduit::blueprint::mesh::utils;

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

// TODO(JRC): Consider moving these structures somewhere else.

//-------------------------------------------------------------------------
struct SharedFace
{
    index_t m_face_id;
    std::vector<index_t> m_fine_subelem;
};


struct PolyBndry
{
    index_t side; //which 3D side 0-5
    index_t m_nbr_rank;
    index_t m_nbr_id;
    size_t m_nbrs_per_face;
    std::vector<index_t> m_elems; //elems of nbr domain that touch side
    std::map<index_t, index_t> m_bface; //map from nbr elem to face of nbr elem
    std::map<index_t, std::vector<index_t> > m_nbr_elems; //map from local
                                                          //elem to all
                                                          //nbr elems that
                                                          //touch it 
    //outer map: local elem, inner map: nbr elem to face
    std::map<index_t, std::map<index_t, SharedFace> > m_nbr_faces;
};

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
    // we need a list of all possible topos, coordsets, etc
    // for the blueprint index in the root file. 
    //
    // across ranks, domains may be sparse
    //  for example: a topo may only exist in one domain
    // so we union all local mesh indices, and then 
    // se an all gather and union the results together
    // to create an accurate global index. 

    index_t local_num_domains = blueprint::mesh::number_of_domains(mesh);
    // note: 
    // find global # of domains w/o conduit_blueprint_mpi for now
    // since we aren't yet linking conduit_blueprint_mpi
    Node n_src, n_reduce;
    n_src = local_num_domains;

    relay::mpi::sum_all_reduce(n_src,
                               n_reduce,
                               comm);

    index_t global_num_domains = n_reduce.to_int();

    index_out.reset();

    Node local_idx, gather_idx;

    if(local_num_domains > 0)
    {
        ::conduit::blueprint::mesh::generate_index(mesh,
                                                   ref_path,
                                                   global_num_domains,
                                                   local_idx);
    }

    relay::mpi::all_gather_using_schema(local_idx,
                                        gather_idx,
                                        comm);

    // union all entries into final index that reps
    // all domains
    NodeConstIterator itr = gather_idx.children();
    while(itr.has_next())
    {
        const Node &curr = itr.next();
        index_out.update(curr);
    }
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
            domain_id = domain["state/domain_id"].to_int64();
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
void to_polytopal(const Node &n,
                  Node &dest,
                  const std::string& name,
                  MPI_Comm comm)
{

    const std::vector<const conduit::Node *> doms = ::conduit::blueprint::mesh::domains(n);

    // make sure all topos match
    index_t ok = 1;
    index_t num_doms = (index_t) doms.size();
    index_t dims = -1;

    for (const auto& dom_ptr : doms)
    {
        if(dom_ptr->fetch("topologies").has_child(name))
        {
            const Node &topo = dom_ptr->fetch("topologies")[name];
            if(topo["type"].as_string() == "structured")
            {
                dims = topo["elements/dims"].number_of_children();
            }
            else
            {
                ok = 0;
            }
        }else
        {
            ok = 0;
        }
    }
    
    // reduce and check for consistency (all ok, and all doms are either 2d or 3d)
    Node local, gather;
    local.set(DataType::index_t(3));
    index_t_array local_vals = local.value();
    local_vals[0] = ok;
    local_vals[1] = num_doms;
    local_vals[2] = dims;

    // Note: this might be more efficient as
    // a set of flat gathers into separate arrays
    relay::mpi::all_gather_using_schema(local,
                                        gather,
                                        comm);

    NodeConstIterator gitr =  gather.children();
    index_t gather_dims = -1;
    while(gitr.has_next() && (ok == 1))
    {
        const Node &curr = gitr.next();
        index_t_array gather_vals = curr.value();
        if(gather_vals[0] != 1)
        {
            ok = 0;
        }
        else
        {
            // this proc has domains and we haven't inited dims
            if( gather_vals[1] > 0 && gather_dims == -1)
            {
                gather_dims = gather_vals[2];
            }
            else if(gather_vals[1] > 0) // this proc has domains
            {
                if(gather_dims != gather_vals[2])
                {
                    ok = 0;
                }
            }
        }
    }

    if(ok == 1)
    {
        if(dims == 2)
        {
            to_polygonal(n,dest,name,comm);
        }
        else if(dims == 3)
        {
            to_polyhedral(n,dest,name,comm);
        }
        else
        {
            CONDUIT_ERROR("to_polytopal only supports 2d or 3d structured toplogies"
                          " (passed mesh has dims = " << dims  << ")");
        }
    }
    else
    {
        CONDUIT_ERROR("to_polytopal only supports structured toplogies");
    }
}

//-------------------------------------------------------------------------
void to_polygonal(const Node &n,
                  Node &dest,
                  const std::string& name,
                  MPI_Comm comm)
{
    // Helper Functions //

    const static auto gen_default_name = [] (const std::string &prefix, const index_t id)
    {
        std::ostringstream oss;
        oss << prefix << "_" << std::setw(6) << std::setfill('0') << id;
        return oss.str();
    };

    const static auto is_domain_local = [] (const conduit::Node &root, const std::string &name)
    {
        return ::conduit::blueprint::mesh::is_multi_domain(root) && root.has_child(name);
    };

    // Implementation //

    // TODO(JRC): Use widest data type available in given Node for both ints and doubles
    // TODO(JRC): Communicate this across the network?

    dest.reset();

    Node temp;

    std::map<index_t, std::map<index_t, std::vector<index_t> > > poly_elems_map;
    std::map<index_t, std::map<index_t, std::vector<double> > > dom_to_nbr_to_xbuffer;
    std::map<index_t, std::map<index_t, std::vector<double> > > dom_to_nbr_to_ybuffer;

    const std::vector<const conduit::Node *> doms = ::conduit::blueprint::mesh::domains(n);
    for (index_t si = 0; si < 3; si++)
    {
        // TODO(JRC): Figure out better names for all of these stages.
        // TODO(JRC): Consider using a local enumeration to track this
        // instead, to have compile-time consistency.
        const std::string stage_name = (
            (si == 0) ? "send_missing_windows" : (
            (si == 1) ? "fill_all_windows" : (
            (si == 2) ? "generate_polygons" : (""))));

        for (const auto& dom_ptr : doms)
        {
            const Node &dom = *dom_ptr;
            Node &dest_dom = ::conduit::blueprint::mesh::is_multi_domain(n) ?
                dest[dom.name()] : dest;

            const Node &in_topo = dom["topologies"][name];
            const Node *in_cset = 
                bputils::find_reference_node(in_topo, "coordset");
            Node &out_cset = dest_dom["coordsets"][in_topo["coordset"].as_string()];

            const index_t domain_id = dom["state/domain_id"].to_index_t();
            const index_t level_id = dom["state/level_id"].to_index_t();
            const std::string ref_name = gen_default_name("window", domain_id);

            const index_t i_lo = in_topo["elements/origin/i0"].to_index_t();
            const index_t j_lo = in_topo["elements/origin/j0"].to_index_t();
            const index_t iwidth = in_topo["elements/dims/i"].to_index_t();
            const index_t jwidth = in_topo["elements/dims/j"].to_index_t();
            const index_t niwidth = in_topo["elements/dims/i"].to_index_t() + 1;

            // TODO(JRC): There's an implicit assumption here about each
            // domain having a 'state' path, which is optional in the general
            // case. This needs to be checked and warned about before any
            // processing occurs.
            if (si == 0)
            {
                if (level_id == 0)
                {
                    continue;
                }
            }
            else if (si == 2)
            {
                if (blueprint::mesh::coordset::uniform::verify(*in_cset, temp))
                {
                    blueprint::mesh::coordset::uniform::to_explicit(*in_cset, out_cset);
                }
                else if (blueprint::mesh::coordset::rectilinear::verify(*in_cset, temp))
                {
                    blueprint::mesh::coordset::rectilinear::to_explicit(*in_cset, out_cset);
                }
                else
                {
                    out_cset.set(*in_cset);
                }
            }

            auto &poly_elems = poly_elems_map[domain_id];
            auto &nbr_to_xbuffer = dom_to_nbr_to_xbuffer[domain_id];
            auto &nbr_to_ybuffer = dom_to_nbr_to_ybuffer[domain_id];

            if (dom.has_path("adjsets/adjset/groups"))
            {
                NodeConstIterator grp_itr = dom["adjsets/adjset/groups"].children();
                while(grp_itr.has_next())
                {
                    const Node& group = grp_itr.next();
                    if (group.has_child("neighbors") && group.has_child("windows"))
                    {
                        temp.reset();
                        temp.set_external(DataType(group["neighbors"].dtype().id(), 1),
                                          (void*)group["neighbors"].element_ptr(1));
                        const index_t nbr_id = temp.to_index_t();
                        const std::string nbr_name = gen_default_name("domain", nbr_id);

                        const Node &in_windows = group["windows"];
                        const std::string nbr_win_name = gen_default_name("window", nbr_id);

                        const Node &ref_win = in_windows[ref_name];
                        const Node &nbr_win = in_windows[nbr_win_name];

                        const index_t nbr_level = nbr_win["level_id"].to_index_t();
                        const index_t ref_level = ref_win["level_id"].to_index_t();

                        if ((si == 0 && nbr_level < ref_level) ||
                            (si != 0 && nbr_level > ref_level))
                        {
                            const index_t ref_size_i = ref_win["dims/i"].to_index_t();
                            const index_t ref_size_j = ref_win["dims/j"].to_index_t();
                            const index_t ref_size = ref_size_i * ref_size_j;

                            const index_t nbr_size_i = nbr_win["dims/i"].to_index_t();
                            const index_t nbr_size_j = nbr_win["dims/j"].to_index_t();
                            const index_t nbr_size = nbr_size_i * nbr_size_j;

                            const index_t ratio_i = nbr_win["ratio/i"].to_index_t();
                            const index_t ratio_j = nbr_win["ratio/j"].to_index_t();

                            if (si == 0 && nbr_size < ref_size && !is_domain_local(n, nbr_name))
                            {
                                std::vector<double> xbuffer, ybuffer;
                                const Node &fcoords = (*in_cset)["values"];

                                const index_t origin_i = ref_win["origin/i"].to_index_t();
                                const index_t origin_j = ref_win["origin/j"].to_index_t();

                                temp.reset();

                                if (ref_size_i == 1)
                                {
                                    const index_t icnst = origin_i - i_lo;
                                    const index_t jstart = origin_j - j_lo;
                                    const index_t jend = jstart + ref_size_j;
                                    for (index_t jidx = jstart; jidx < jend; ++jidx)
                                    {
                                        const index_t offset = jidx * niwidth + icnst;
                                        temp.set_external(DataType(fcoords["x"].dtype().id(), 1),
                                                          (void*)fcoords["x"].element_ptr(offset));
                                        xbuffer.push_back(temp.to_double());
                                        temp.set_external(DataType(fcoords["y"].dtype().id(), 1),
                                                          (void*)fcoords["y"].element_ptr(offset));
                                        ybuffer.push_back(temp.to_double());
                                    }
                                }
                                else if (ref_size_j == 1)
                                {
                                    const index_t jcnst = origin_j - j_lo;
                                    const index_t istart = origin_i - i_lo;
                                    const index_t iend = istart + ref_size_i;
                                    for (index_t iidx = istart; iidx < iend; ++iidx)
                                    {
                                        const index_t offset = jcnst * niwidth + iidx;
                                        temp.set_external(DataType(fcoords["x"].dtype().id(), 1),
                                                          (void*)fcoords["x"].element_ptr(offset));
                                        xbuffer.push_back(temp.to_double());
                                        temp.set_external(DataType(fcoords["y"].dtype().id(), 1),
                                                          (void*)fcoords["y"].element_ptr(offset));
                                        ybuffer.push_back(temp.to_double());
                                    }
                                }
                                const index_t nbr_rank = group["rank"].to_index_t();
                                MPI_Send(&xbuffer[0],
                                         xbuffer.size(),
                                         MPI_DOUBLE,
                                         nbr_rank,
                                         domain_id,
                                         comm);
                                MPI_Send(&ybuffer[0],
                                         ybuffer.size(),
                                         MPI_DOUBLE,
                                         nbr_rank,
                                         domain_id,
                                         comm);
                            }
                            else if (si == 1 && nbr_size > ref_size)
                            {
                                auto& xbuffer = nbr_to_xbuffer[nbr_id];
                                auto& ybuffer = nbr_to_ybuffer[nbr_id];

                                if (!is_domain_local(n, nbr_name))
                                {
                                    if (nbr_size_i == 1)
                                    {
                                        xbuffer.resize(nbr_size_j);
                                        ybuffer.resize(nbr_size_j);
                                    }
                                    else if (nbr_size_j == 1)
                                    {
                                        xbuffer.resize(nbr_size_i);
                                        ybuffer.resize(nbr_size_i);
                                    }

                                    index_t nbr_rank = group["rank"].to_index_t();
                                    MPI_Recv(&xbuffer[0],
                                             xbuffer.size(),
                                             MPI_DOUBLE,
                                             nbr_rank,
                                             nbr_id,
                                             comm,
                                             MPI_STATUS_IGNORE);
                                    MPI_Recv(&ybuffer[0],
                                             ybuffer.size(),
                                             MPI_DOUBLE,
                                             nbr_rank,
                                             nbr_id,
                                             comm,
                                             MPI_STATUS_IGNORE);

                                }
                                else
                                {
                                    const Node& nbr_dom = n[nbr_name];
                                    const Node& nbr_coords = nbr_dom["coordsets/coords"];

                                    const Node& ntopo = nbr_dom["topologies"][name];
                                    index_t ni_lo = ntopo["elements/origin/i0"].to_index_t();
                                    index_t nj_lo = ntopo["elements/origin/j0"].to_index_t();
                                    index_t nbr_iwidth = ntopo["elements/dims/i"].to_index_t() + 1;

                                    // FIXME(JRC): These arrays aren't guaranteed to be double arrays;
                                    // this code must be changed to accomodate arbitrary types.
                                    const Node& fcoords = nbr_coords["values"];
                                    const double_array& xarray = fcoords["x"].as_double_array();
                                    const double_array& yarray = fcoords["y"].as_double_array();

                                    index_t origin_i = nbr_win["origin/i"].to_index_t();
                                    index_t origin_j = nbr_win["origin/j"].to_index_t();

                                    index_t istart = origin_i - ni_lo;
                                    index_t jstart = origin_j - nj_lo;
                                    index_t iend = istart + nbr_size_i;
                                    index_t jend = jstart + nbr_size_j;
                                    for (index_t jidx = jstart; jidx < jend; ++jidx)
                                    {
                                        index_t joffset = jidx*nbr_iwidth;
                                        for (index_t iidx = istart; iidx < iend; ++iidx)
                                        {
                                            index_t offset = joffset+iidx;
                                            xbuffer.push_back(xarray[offset]);
                                            ybuffer.push_back(yarray[offset]);
                                        }
                                    }
                                }
                            }
                            else if (si == 2 && ref_size < nbr_size)
                            {
                                bputils::connectivity::create_elements_2d(ref_win,
                                                                          i_lo,
                                                                          j_lo,
                                                                          iwidth,
                                                                          poly_elems);

                                const index_t use_ratio = (
                                    (nbr_size_j == 1) ? ratio_i : (
                                    (nbr_size_i == 1) ? ratio_j : (0)));

                                auto& xbuffer = nbr_to_xbuffer[nbr_id];
                                auto& ybuffer = nbr_to_ybuffer[nbr_id];

                                const index_t added = (
                                    (nbr_size_j == 1) ? (xbuffer.size() - ref_size_i) : (
                                    (nbr_size_i == 1) ? (ybuffer.size() - ref_size_j) : (0)));

                                // FIXME(JRC): These arrays aren't guaranteed to be double arrays;
                                // this code must be changed to accomodate arbitrary types.
                                const auto& out_x = out_cset["values"]["x"].as_double_array();
                                const auto& out_y = out_cset["values"]["y"].as_double_array();
                                index_t new_vertex = out_x.number_of_elements();

                                const index_t out_x_size = out_x.number_of_elements();
                                const index_t out_y_size = out_y.number_of_elements();

                                std::vector<double> new_x;
                                std::vector<double> new_y;
                                new_x.reserve(out_x_size + added);
                                new_y.reserve(out_y_size + added);
                                const double* out_x_ptr = static_cast<const double*>(out_x.element_ptr(0));
                                const double* out_y_ptr = static_cast<const double*>(out_y.element_ptr(0));

                                new_x.insert(new_x.end(), out_x_ptr, out_x_ptr + out_x_size);
                                new_y.insert(new_y.end(), out_y_ptr, out_y_ptr + out_y_size);

                                if ((xbuffer.size()-1)%use_ratio)
                                {
                                    new_x.reserve(out_x_size + added*2);
                                }
                                for (index_t ni = 0; ni < (index_t)xbuffer.size(); ++ni)
                                {
                                    if (ni % use_ratio)
                                    {
                                        new_x.push_back(xbuffer[ni]);
                                        new_y.push_back(ybuffer[ni]);
                                    }
                                }

                                out_cset["values"]["x"].set(new_x);
                                out_cset["values"]["y"].set(new_y);

                                bputils::connectivity::connect_elements_2d(ref_win,
                                                                           i_lo,
                                                                           j_lo,
                                                                           iwidth,
                                                                           use_ratio,
                                                                           new_vertex,
                                                                           poly_elems);
                            }
                        }
                    }
                }
            }

            if (si == 2)
            {
                dest_dom["state"].set(dom["state"]);
                dest_dom["topologies"][name]["coordset"].set(dom["topologies"][name]["coordset"]);

                Node& out_topo = dest_dom["topologies"][name];
                out_topo["type"].set("unstructured");
                out_topo["elements/shape"].set("polygonal");

                const index_t elemsize = iwidth*jwidth;

                std::vector<index_t> connect;
                std::vector<index_t> num_vertices;
                std::vector<index_t> elem_offsets;
                index_t offset_sum = 0;
                for (index_t elem = 0; elem < elemsize; ++elem)
                {
                    auto elem_itr = poly_elems.find(elem);
                    if (elem_itr == poly_elems.end())
                    {
                        bputils::connectivity::make_element_2d(connect, elem, iwidth);
                        num_vertices.push_back(4);
                    }
                    else
                    {
                        std::vector<index_t>& poly_elem = elem_itr->second;
                        connect.insert(connect.end(), poly_elem.begin(), poly_elem.end());
                        num_vertices.push_back(poly_elem.size());
                    }
                    elem_offsets.push_back(offset_sum);
                    offset_sum += num_vertices.back();
                }

                // TODO(JRC): Remove extra copy here.
                out_topo["elements/connectivity"].set(connect);
                out_topo["elements/sizes"].set(num_vertices);
                out_topo["elements/offsets"].set(elem_offsets);

                //TODO:  zero copy
                if (dom.has_child("fields"))
                {
                    dest_dom["fields"].set(dom["fields"]);
                }
            }
        }
    }
}

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

void
partition_map_back(const conduit::Node& repart_mesh,
                   const conduit::Node& options,
                   conduit::Node& orig_mesh,
                   MPI_Comm comm)
{
    ParallelPartitioner p(comm);
    p.map_back_fields(repart_mesh, options, orig_mesh);
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
void match_nbr_elems(PolyBndry& pbnd,
                     std::map<index_t, bputils::connectivity::ElemType>& nbr_elems,
                     std::map<index_t, bputils::connectivity::SubelemMap>& allfaces_map,
                     const Node& ref_topo,
                     const Node& ref_win,
                     const Node& nbr_win,
                     index_t nbr_iwidth, index_t nbr_jwidth,
                     index_t nbr_kwidth,
                     index_t ni_lo, index_t nj_lo, index_t nk_lo,
                     index_t ratio_i, index_t ratio_j, index_t ratio_k,
                     bool local)
{
    index_t nbr_size_i = nbr_win["dims/i"].to_index_t();
    index_t nbr_size_j = nbr_win["dims/j"].to_index_t();
    index_t nbr_size_k = nbr_win["dims/k"].to_index_t();

    index_t ri_lo = ref_topo["elements/origin/i0"].to_index_t();
    index_t rj_lo = ref_topo["elements/origin/j0"].to_index_t();
    index_t rk_lo = ref_topo["elements/origin/k0"].to_index_t();

    index_t ref_iwidth = ref_topo["elements/dims/i"].to_index_t();
    index_t ref_jwidth = ref_topo["elements/dims/j"].to_index_t();

    index_t ref_origin_i = ref_win["origin/i"].to_index_t();
    index_t ref_origin_j = ref_win["origin/j"].to_index_t();
    index_t ref_origin_k = ref_win["origin/k"].to_index_t();

    index_t origin_i = nbr_win["origin/i"].to_index_t();
    index_t origin_j = nbr_win["origin/j"].to_index_t();
    index_t origin_k = nbr_win["origin/k"].to_index_t();

    index_t side = pbnd.side;

    if (side == 0 || side == 1)
    {
        pbnd.m_nbrs_per_face = static_cast<size_t>(ratio_j*ratio_k);

        index_t nside = (side+1)%2;     //nbr side counterpart to  ref side
        index_t shift = -(nside%2); //0 on low side, -1 on high side
        index_t icnst = origin_i - ni_lo + shift;
        index_t jstart = origin_j;
        index_t jend = jstart + nbr_size_j - 1;
        index_t kstart = origin_k;
        index_t kend = kstart + nbr_size_k - 1;

        index_t rshift = -(side%2); 
        index_t ricnst = ref_origin_i - ri_lo + rshift;

        for (index_t kidx = kstart; kidx < kend; ++kidx)
        {
            index_t rkidx = kidx/ratio_k;
            index_t nk = kidx - nk_lo;
            index_t rk = rkidx - rk_lo;
            index_t nkoffset = icnst + nk*nbr_iwidth*nbr_jwidth;
            index_t rkoffset = ricnst + rk*ref_iwidth*ref_jwidth;
            for (index_t jidx = jstart; jidx < jend; ++jidx)
            {
                index_t rjidx = jidx/ratio_j;
                index_t nj = jidx - nj_lo;
                index_t rj = rjidx - rj_lo;
                index_t noffset = nkoffset + nj*nbr_iwidth;
                index_t roffset = rkoffset + rj*ref_iwidth;

                pbnd.m_nbr_elems[roffset].push_back(noffset);

                if (local)
                {
                    auto& nbr_elem = nbr_elems[noffset];
                    index_t face_id =
                        nside == 0 ?
                        nbr_elem[0] :
                        nbr_elem[1];
                    pbnd.m_nbr_faces[roffset][noffset].m_face_id = face_id;
                    pbnd.m_nbr_faces[roffset][noffset].m_fine_subelem =
                        allfaces_map[pbnd.m_nbr_id][face_id]; 
                }
                else
                {
                    std::vector<index_t> nbr_elem;
                    std::map< index_t, std::vector<index_t> > elem_faces;
                    bputils::connectivity::make_element_3d(nbr_elem,
                                                           noffset,
                                                           nbr_iwidth,
                                                           nbr_jwidth,
                                                           nbr_kwidth,
                                                           elem_faces);

                    index_t jfoffset = nj*(nbr_iwidth+1);
                    index_t kfoffset = nk*nbr_jwidth*(nbr_iwidth+1);
                    index_t face_id;
                    if (nside == 0)
                    {
                        face_id =
                            icnst + jfoffset + kfoffset;
                    }
                    else
                    {
                        face_id =
                            icnst + jfoffset + kfoffset + 1;
                    }
                    pbnd.m_nbr_faces[roffset][noffset].m_face_id =
                        face_id;
                    pbnd.m_nbr_faces[roffset][noffset].m_fine_subelem =
                        elem_faces[face_id];
                }
            }
        }
    }
    else if (side == 2 || side == 3)
    {
        pbnd.m_nbrs_per_face = static_cast<size_t>(ratio_i*ratio_k);

        index_t jface_start = (nbr_iwidth+1)*nbr_jwidth*nbr_kwidth;
        index_t nside = 2 + (side+1)%2;     //nbr side counterpart to  ref side
        index_t shift = -(nside%2); //0 on low side, -1 on high side
        index_t jcnst = origin_j - nj_lo + shift;
        index_t istart = origin_i;
        index_t iend = istart + nbr_size_i - 1;
        index_t kstart = origin_k;
        index_t kend = kstart + nbr_size_k - 1;

        index_t rshift = -(side%2); 
        index_t rjcnst = ref_origin_j - rj_lo + rshift;

        for (index_t kidx = kstart; kidx < kend; ++kidx)
        {
            index_t rkidx = kidx/ratio_k;
            index_t nk = kidx - nk_lo;
            index_t rk = rkidx - rk_lo;
            index_t nkoffset = nk*nbr_iwidth*nbr_jwidth;
            index_t rkoffset = rk*ref_iwidth*ref_jwidth;
            for (index_t iidx = istart; iidx < iend; ++iidx)
            {
                index_t riidx = iidx/ratio_i;
                index_t ni = iidx - ni_lo;
                index_t ri = riidx - ri_lo;
                index_t noffset = nkoffset + jcnst*nbr_iwidth + ni;
                index_t roffset = rkoffset + rjcnst*ref_iwidth + ri;

                pbnd.m_nbr_elems[roffset].push_back(noffset);

                if (local)
                {
                    auto& nbr_elem = nbr_elems[noffset];
                    index_t face_id =
                        nside == 2 ?
                        nbr_elem[2] :
                        nbr_elem[3];
                    pbnd.m_nbr_faces[roffset][noffset].m_face_id = face_id;
                    pbnd.m_nbr_faces[roffset][noffset].m_fine_subelem =
                        allfaces_map[pbnd.m_nbr_id][face_id];
                }
                else
                {
                    std::vector<index_t> nbr_elem;
                    std::map< index_t, std::vector<index_t> > elem_faces;
                    bputils::connectivity::make_element_3d(nbr_elem,
                                                           noffset,
                                                           nbr_iwidth,
                                                           nbr_jwidth,
                                                           nbr_kwidth,
                                                           elem_faces);

                    index_t ifoffset = ni;
                    index_t kfoffset = nk*(nbr_jwidth+1)*nbr_iwidth;
                    index_t face_id;
                    if (nside == 2)
                    {
                        face_id =
                            jface_start +
                            nbr_iwidth*jcnst + ifoffset + kfoffset;
                    }
                    else
                    {
                        face_id =
                            jface_start +
                            nbr_iwidth*(jcnst+1) + ifoffset + kfoffset;
                    }
                    pbnd.m_nbr_faces[roffset][noffset].m_face_id =
                        face_id;
                    pbnd.m_nbr_faces[roffset][noffset].m_fine_subelem =
                        elem_faces[face_id];
                }
            }
        }
    }
    else if (side == 4 || side == 5)
    {
        pbnd.m_nbrs_per_face = static_cast<size_t>(ratio_i*ratio_j);

        index_t jface_start = (nbr_iwidth+1)*nbr_jwidth*nbr_kwidth;
        index_t kface_start = jface_start +
                              nbr_iwidth*(nbr_jwidth+1)*nbr_kwidth;
        index_t nside = 4 + (side+1)%2;     //nbr side counterpart to  ref side
        index_t shift = -(nside%2); //0 on low side, -1 on high side
        index_t kcnst = origin_k - nk_lo + shift;
        index_t jstart = origin_j;
        index_t jend = jstart + nbr_size_j - 1;
        index_t istart = origin_i;
        index_t iend = istart + nbr_size_i - 1;

        index_t rshift = -(side%2); 
        index_t rkcnst = ref_origin_k - rk_lo + rshift;

        for (index_t jidx = jstart; jidx < jend; ++jidx)
        {
            index_t rjidx = jidx/ratio_j;
            index_t nj = jidx - nj_lo;
            index_t rj = rjidx - rj_lo;
            index_t njoffset = kcnst*nbr_iwidth*nbr_jwidth + nj*nbr_iwidth;
            index_t rjoffset = rkcnst*ref_iwidth*ref_jwidth + rj*ref_iwidth;
            for (index_t iidx = istart; iidx < iend; ++iidx)
            {
                index_t riidx = iidx/ratio_i;
                index_t ni = iidx - ni_lo;
                index_t ri = riidx - ri_lo;
                index_t noffset = njoffset + ni;
                index_t roffset = rjoffset + ri;

                pbnd.m_nbr_elems[roffset].push_back(noffset);

                if (local)
                { 
                    auto& nbr_elem = nbr_elems[noffset];
                    index_t face_id =
                        nside == 4 ?
                        nbr_elem[4] :
                        nbr_elem[5];
                    pbnd.m_nbr_faces[roffset][noffset].m_face_id = face_id;
                    pbnd.m_nbr_faces[roffset][noffset].m_fine_subelem =
                        allfaces_map[pbnd.m_nbr_id][face_id];
                }
                else
                {
                    std::vector<index_t> nbr_elem;
                    std::map< index_t, std::vector<index_t> > elem_faces;
                    bputils::connectivity::make_element_3d(nbr_elem,
                                                           noffset,
                                                           nbr_iwidth,
                                                           nbr_jwidth,
                                                           nbr_kwidth,
                                                           elem_faces);
                    index_t ifoffset = ni;
                    index_t jfoffset = nj*nbr_iwidth;
                    index_t face_id;
                    if (nside == 4)
                    {
                        face_id =
                            kface_start +
                            nbr_iwidth*nbr_jwidth*kcnst + ifoffset + jfoffset;
                    }
                    else
                    {
                        face_id =
                            kface_start +
                            nbr_iwidth*nbr_jwidth*(kcnst+1) + ifoffset + jfoffset;
                    }
                    pbnd.m_nbr_faces[roffset][noffset].m_face_id =
                        face_id;
                    pbnd.m_nbr_faces[roffset][noffset].m_fine_subelem =
                        elem_faces[face_id];
                } 
            }
        }
    }
    else
    {
        //not a side
    }

}


//-------------------------------------------------------------------------
void fix_duplicated_vertices(std::set<index_t>& check_verts,
std::vector<index_t>& elem,
bputils::connectivity::SubelemMap& allfaces,
        const double_array& xarray,
        const double_array& yarray,
        const double_array& zarray)
{

    double eps = sqrt(std::numeric_limits<double>::epsilon());

    std::set<index_t> erase_verts;
    for (auto it0 = check_verts.begin(); it0 != check_verts.end(); ++it0)
    {
        const index_t& v0 = *it0;
        double x0 = xarray[v0]; 
        double y0 = yarray[v0]; 
        double z0 = zarray[v0]; 

        auto it1 = it0;
        ++it1;

        for ( ; it1 != check_verts.end(); ++it1)
        {
            const index_t& v1 = *it1;
            double x1 = xarray[v1];
            double y1 = yarray[v1];
            double z1 = zarray[v1];
            double xdiff = x0-x1;
            double ydiff = y0-y1;
            double zdiff = z0-z1;
            double distsqr = xdiff*xdiff+ydiff*ydiff+zdiff*zdiff;

            if (distsqr < eps)
            {
                for (auto eitr = elem.begin(); eitr != elem.end(); ++eitr)
                {
                    auto& face = allfaces[*eitr]; 
                    for (auto fitr = face.begin(); fitr != face.end(); ++fitr)
                    {
                        index_t& vert = *fitr;
                        if (vert == v1)
                        {
                            vert = v0;
                        }
                    }
                }
                erase_verts.insert(v1);
            }
        }
    }

    for (auto ev = erase_verts.begin(); ev != erase_verts.end(); ++ev)
    {
        check_verts.erase(*ev);
    }
}

//-------------------------------------------------------------------------
void to_polyhedral(const Node &n,
                   Node &dest,
                   const std::string& name,
                   MPI_Comm comm)
{
    dest.reset();

    Node temp;

    index_t par_rank = relay::mpi::rank(comm);

    NodeConstIterator itr = n.children();

    std::map<index_t, std::map<index_t, bputils::connectivity::ElemType> > poly_elems_map;
    std::map<index_t, std::map<index_t, std::set<index_t> > > new_vert_map;
    std::map<index_t, std::map<index_t, std::set<index_t> > > new_face_map;
    std::map<index_t, bputils::connectivity::SubelemMap> allfaces_map;

    std::map<index_t, std::vector<index_t> > elem_connect;
    std::map<index_t, std::vector<index_t> > elem_sizes;
    std::map<index_t, std::vector<index_t> > elem_offsets;
    std::map<index_t, std::vector<index_t> > subelem_connect;
    std::map<index_t, std::vector<index_t> > subelem_sizes;
    std::map<index_t, std::vector<index_t> > subelem_offsets;

    while(itr.has_next())
    {
        const Node& chld = itr.next();
        std::string domain_name = itr.name();
        dest[domain_name]["state"] = chld["state"];
        const Node& in_coords = chld["coordsets/coords"];
        const Node& in_topo = chld["topologies"][name];

        index_t iwidth = in_topo["elements/dims/i"].to_index_t();
        index_t jwidth = in_topo["elements/dims/j"].to_index_t();
        index_t kwidth = in_topo["elements/dims/k"].to_index_t();

        Node& out_coords = dest[domain_name]["coordsets/coords"];

        index_t domain_id = chld["state/domain_id"].to_index_t();
        Node& out_values = out_coords["values"];
        if (in_coords["type"].as_string() == "uniform")
        {
            blueprint::mesh::coordset::uniform::to_explicit(in_coords, out_coords);
        }
        else
        {
            out_coords["type"] = in_coords["type"];
            const Node& in_values = in_coords["values"];
            out_values = in_values;
        }

        auto& poly_elems = poly_elems_map[domain_id];
        auto& allfaces = allfaces_map[domain_id];

        index_t elemsize = iwidth*jwidth*kwidth;

        for (index_t elem = 0; elem < elemsize; ++elem)
        {
            bputils::connectivity::make_element_3d(poly_elems[elem],
                                                   elem,
                                                   iwidth,
                                                   jwidth,
                                                   kwidth,
                                                   allfaces);
        }
    }

    itr = n.children();
    while(itr.has_next())
    {
        const Node& chld = itr.next();
        std::string domain_name = itr.name();

        index_t domain_id = chld["state/domain_id"].to_index_t();

        const Node& in_topo = chld["topologies"][name];

        const Node *in_cset = bputils::find_reference_node(in_topo, "coordset");

        const index_t i_lo = in_topo["elements/origin/i0"].to_index_t();
        const index_t j_lo = in_topo["elements/origin/j0"].to_index_t();
        const index_t k_lo = in_topo["elements/origin/k0"].to_index_t();
        const index_t niwidth = in_topo["elements/dims/i"].to_index_t() + 1;
        const index_t njwidth = in_topo["elements/dims/j"].to_index_t() + 1;

        const Node* in_parent = chld.parent();

        std::ostringstream win_oss;
        win_oss << "window_" << std::setw(6) << std::setfill('0') << domain_id;
        std::string win_name = win_oss.str();

        if (chld.has_path("adjsets/adjset/groups"))
        {
            const Node& in_groups = chld["adjsets/adjset/groups"];
            NodeConstIterator grp_itr = in_groups.children();
            while(grp_itr.has_next())
            {
                const Node& group = grp_itr.next();
                if (group.has_child("neighbors") && group.has_child("windows"))
                {
                    temp.reset();
                    temp.set_external(DataType(group["neighbors"].dtype().id(), 1),
                                      (void*)group["neighbors"].element_ptr(1));
                    const index_t nbr_id = temp.to_index_t();
                    const Node& in_windows = group["windows"];
                    std::ostringstream nw_oss;
                    nw_oss << "window_" << std::setw(6)
                            << std::setfill('0') << nbr_id;
                    std::string nbr_win_name = nw_oss.str();

                    const Node& ref_win = in_windows[win_name];
                    const Node& nbr_win = in_windows[nbr_win_name];

                    if (nbr_win["level_id"].to_index_t() < ref_win["level_id"].to_index_t())
                    {
                        index_t ref_size_i = ref_win["dims/i"].to_index_t();
                        index_t ref_size_j = ref_win["dims/j"].to_index_t();
                        index_t ref_size_k = ref_win["dims/k"].to_index_t();
                        index_t ref_size = ref_size_i*ref_size_j*ref_size_k;

                        index_t nbr_size_i = nbr_win["dims/i"].to_index_t();
                        index_t nbr_size_j = nbr_win["dims/j"].to_index_t();
                        index_t nbr_size_k = nbr_win["dims/k"].to_index_t();
                        index_t nbr_size = nbr_size_i*nbr_size_j*nbr_size_k;

                        if (nbr_size < ref_size)
                        {
                            std::ostringstream nbr_oss;
                            nbr_oss << "domain_" << std::setw(6)
                                    << std::setfill('0') << nbr_id;
                            std::string nbr_name = nbr_oss.str();

                            bool is_side = true;
                            if (nbr_size_i * nbr_size_j == 1 ||
                                nbr_size_i * nbr_size_k == 1 ||
                                nbr_size_j * nbr_size_k == 1)
                            {
                                is_side = false;
                            }

                            if (is_side && !in_parent->has_child(nbr_name))
                            {
                                index_t buffer[6];
                                buffer[0] = in_topo["elements/dims/i"].to_index_t();
                                buffer[1] = in_topo["elements/dims/j"].to_index_t();
                                buffer[2] = in_topo["elements/dims/k"].to_index_t();
                                buffer[3] = in_topo["elements/origin/i0"].to_index_t();
                                buffer[4] = in_topo["elements/origin/j0"].to_index_t();
                                buffer[5] = in_topo["elements/origin/k0"].to_index_t();

                                index_t nbr_rank = group["rank"].to_index_t();
                                MPI_Send(buffer,
                                         6,
                                         MPI_INT64_T,
                                         nbr_rank,
                                         domain_id,
                                         comm);

                                std::vector<index_t> vertices;
                                std::vector<double> xbuffer, ybuffer, zbuffer;
                                const Node &fcoords = (*in_cset)["values"];

                                const index_t origin_i = ref_win["origin/i"].to_index_t();
                                const index_t origin_j = ref_win["origin/j"].to_index_t();
                                const index_t origin_k = ref_win["origin/k"].to_index_t();

                                Node temp;

                                if (ref_size_i == 1)
                                {
                                    const index_t icnst = origin_i - i_lo;
                                    const index_t jstart = origin_j - j_lo;
                                    const index_t jend = jstart + ref_size_j;
                                    const index_t kstart = origin_k - k_lo;
                                    const index_t kend = kstart + ref_size_k;
                                    for (index_t kidx = kstart; kidx < kend; ++kidx)
                                    {
                                        const index_t koffset = kidx * niwidth * njwidth;
                                        for (index_t jidx = jstart; jidx < jend; ++jidx)
                                        {
                                            const index_t offset = koffset + jidx * niwidth + icnst;
                                            vertices.push_back(offset);
                                            temp.set_external(DataType(fcoords["x"].dtype().id(), 1),
                                                              (void*)fcoords["x"].element_ptr(offset));
                                            xbuffer.push_back(temp.to_double());
                                            temp.set_external(DataType(fcoords["y"].dtype().id(), 1),
                                                              (void*)fcoords["y"].element_ptr(offset));
                                            ybuffer.push_back(temp.to_double());
                                            temp.set_external(DataType(fcoords["z"].dtype().id(), 1),
                                                              (void*)fcoords["z"].element_ptr(offset));
                                            zbuffer.push_back(temp.to_double());
                                        }
                                    }
                                }
                                else if (ref_size_j == 1)
                                {
                                    const index_t istart = origin_i - i_lo;
                                    const index_t iend = istart + ref_size_i;
                                    const index_t jcnst = origin_j - j_lo;
                                    const index_t kstart = origin_k - k_lo;
                                    const index_t kend = kstart + ref_size_k;
                                    for (index_t kidx = kstart; kidx < kend; ++kidx)
                                    {
                                        const index_t koffset = kidx * niwidth * njwidth;
                                        for (index_t iidx = istart; iidx < iend; ++iidx)
                                        {
                                            const index_t offset = koffset + jcnst * niwidth + iidx;
                                            vertices.push_back(offset);
                                            temp.set_external(DataType(fcoords["x"].dtype().id(), 1),
                                                              (void*)fcoords["x"].element_ptr(offset));
                                            xbuffer.push_back(temp.to_double());
                                            temp.set_external(DataType(fcoords["y"].dtype().id(), 1),
                                                              (void*)fcoords["y"].element_ptr(offset));
                                            ybuffer.push_back(temp.to_double());
                                            temp.set_external(DataType(fcoords["z"].dtype().id(), 1),
                                                              (void*)fcoords["z"].element_ptr(offset));
                                            zbuffer.push_back(temp.to_double());
                                        }
                                    }
                                }
                                else
                                {
                                    const index_t istart = origin_i - i_lo;
                                    const index_t iend = istart + ref_size_i;
                                    const index_t jstart = origin_j - j_lo;
                                    const index_t jend = jstart + ref_size_j;
                                    const index_t kcnst = origin_k - k_lo;
                                    const index_t koffset = kcnst * niwidth * njwidth;
                                    for (index_t jidx = jstart; jidx < jend; ++jidx)
                                    {
                                        const index_t joffset = jidx * niwidth;
                                        for (index_t iidx = istart; iidx < iend; ++iidx)
                                        {
                                            const index_t offset = koffset + joffset + iidx;
                                            vertices.push_back(offset);
                                            temp.set_external(DataType(fcoords["x"].dtype().id(), 1),
                                                              (void*)fcoords["x"].element_ptr(offset));
                                            xbuffer.push_back(temp.to_double());
                                            temp.set_external(DataType(fcoords["y"].dtype().id(), 1),
                                                              (void*)fcoords["y"].element_ptr(offset));
                                            ybuffer.push_back(temp.to_double());
                                            temp.set_external(DataType(fcoords["z"].dtype().id(), 1),
                                                              (void*)fcoords["z"].element_ptr(offset));
                                            zbuffer.push_back(temp.to_double());
                                        }
                                    }
                                }


                                MPI_Send(&vertices[0],
                                         vertices.size(),
                                         MPI_INT64_T,
                                         nbr_rank,
                                         domain_id,
                                         comm);
                                MPI_Send(&xbuffer[0],
                                         xbuffer.size(),
                                         MPI_DOUBLE,
                                         nbr_rank,
                                         domain_id,
                                         comm);
                                MPI_Send(&ybuffer[0],
                                         ybuffer.size(),
                                         MPI_DOUBLE,
                                         nbr_rank,
                                         domain_id,
                                         comm);
                                MPI_Send(&zbuffer[0],
                                         zbuffer.size(),
                                         MPI_DOUBLE,
                                         nbr_rank,
                                         domain_id,
                                         comm);


                            }
                        }
                    }
                }
            }
        }
    }

    //Outer map: domain_id, inner map: nbr_id to PolyBndry
    std::map<index_t, std::map<index_t, PolyBndry> > poly_bndry_map;

    std::map<index_t, std::vector<index_t> > nbr_ratio;

    itr = n.children();
    while(itr.has_next())
    {
        const Node& chld = itr.next();
        std::string domain_name = itr.name();

        index_t domain_id = chld["state/domain_id"].to_index_t();

        auto& poly_elems = poly_elems_map[domain_id];

        const Node& in_topo = chld["topologies"][name];

        index_t iwidth = in_topo["elements/dims/i"].to_index_t();
        index_t jwidth = in_topo["elements/dims/j"].to_index_t();

        index_t i_lo = in_topo["elements/origin/i0"].to_index_t();
        index_t j_lo = in_topo["elements/origin/j0"].to_index_t();
        index_t k_lo = in_topo["elements/origin/k0"].to_index_t();

        const Node* in_parent = chld.parent();

        std::ostringstream win_oss;
        win_oss << "window_" << std::setw(6) << std::setfill('0') << domain_id;
        std::string win_name = win_oss.str();

        if (chld.has_path("adjsets/adjset/groups"))
        {
            const Node& in_groups = chld["adjsets/adjset/groups"];
            NodeConstIterator grp_itr = in_groups.children();
            while(grp_itr.has_next())
            {
                const Node& group = grp_itr.next();

                if (group.has_child("neighbors") && group.has_child("windows"))
                {
                    temp.reset();
                    temp.set_external(DataType(group["neighbors"].dtype().id(), 1),
                                      (void*)group["neighbors"].element_ptr(1));
                    const index_t nbr_id = temp.to_index_t();
                    const Node& in_windows = group["windows"];
                    std::ostringstream nw_oss;
                    nw_oss << "window_" << std::setw(6)
                            << std::setfill('0') << nbr_id;
                    std::string nbr_win_name = nw_oss.str();

                    const Node& ref_win = in_windows[win_name];
                    const Node& nbr_win = in_windows[nbr_win_name];
                    if (nbr_win["level_id"].to_index_t() > ref_win["level_id"].to_index_t())
                    {
                        index_t ratio_i = nbr_win["ratio/i"].to_index_t();
                        index_t ratio_j = nbr_win["ratio/j"].to_index_t();
                        index_t ratio_k = nbr_win["ratio/k"].to_index_t();

                        nbr_ratio[domain_id] = {ratio_i, ratio_j, ratio_k};

                        index_t ref_size_i = ref_win["dims/i"].to_index_t();
                        index_t ref_size_j = ref_win["dims/j"].to_index_t();
                        index_t ref_size_k = ref_win["dims/k"].to_index_t();
                        index_t ref_size = ref_size_i*ref_size_j*ref_size_k;

                        index_t nbr_size_i = nbr_win["dims/i"].to_index_t();
                        index_t nbr_size_j = nbr_win["dims/j"].to_index_t();
                        index_t nbr_size_k = nbr_win["dims/k"].to_index_t();
                        index_t nbr_size = nbr_size_i*nbr_size_j*nbr_size_k;

                        if (nbr_size > ref_size)
                        {
                            index_t origin_i = ref_win["origin/i"].to_index_t();
                            index_t origin_j = ref_win["origin/j"].to_index_t();
                            index_t origin_k = ref_win["origin/k"].to_index_t();

                            PolyBndry& pbnd = poly_bndry_map[domain_id][nbr_id];

                            if (ref_size_i == 1 && ref_size_j != 1 &&
                                ref_size_k != 1)
                            {
                                index_t shift = 0;
                                if (origin_i == i_lo)
                                {
                                    pbnd.side = 0;
                                }
                                else
                                {
                                    pbnd.side = 1;
                                    shift = -1;
                                }
                                index_t icnst = origin_i - i_lo + shift;
                                index_t jstart = origin_j - j_lo;
                                index_t jend = jstart + ref_size_j - 1;
                                index_t kstart = origin_k - k_lo;
                                index_t kend = kstart + ref_size_k - 1;
                                for (index_t kidx = kstart; kidx < kend; ++kidx)
                                    {
                                    index_t koffset = icnst + kidx*iwidth*jwidth;
                                    for (index_t jidx = jstart; jidx < jend; ++jidx)
                                    {
                                        index_t offset = koffset + jidx*iwidth;
                                        pbnd.m_elems.push_back(offset);
                                        pbnd.m_bface[offset] =
                                            pbnd.side == 0 ?
                                            poly_elems[offset][0] :
                                            poly_elems[offset][1];
                                    }
                                }
                            }
                            else if (ref_size_j == 1 && ref_size_i != 1 &&
                                     ref_size_k != 1)
                            {
                                index_t shift = 0;
                                if (origin_j == j_lo)
                                {
                                    pbnd.side = 2;
                                }
                                else
                                {
                                    pbnd.side = 3;
                                    shift = -1;
                                }
                                index_t jcnst = origin_j - j_lo + shift;
                                index_t istart = origin_i - i_lo;
                                index_t iend = istart + ref_size_i - 1;
                                index_t kstart = origin_k - k_lo;
                                index_t kend = kstart + ref_size_k - 1;
                                for (index_t kidx = kstart; kidx < kend; ++kidx)
                                {
                                    index_t koffset = jcnst*iwidth + kidx*iwidth*jwidth;
                                    for (index_t iidx = istart; iidx < iend; ++iidx)
                                    {
                                        index_t offset = koffset + iidx;
                                        pbnd.m_elems.push_back(offset);
                                        pbnd.m_bface[offset] =
                                            pbnd.side == 2 ?
                                            poly_elems[offset][2] :
                                            poly_elems[offset][3];
                                    }
                                }
                            }
                            else if (ref_size_k == 1 && ref_size_i != 1 &&
                                     ref_size_j != 1)
                            {
                                index_t shift = 0; 
                                if (origin_k == k_lo)
                                {
                                    pbnd.side = 4;
                                }
                                else
                                {
                                    pbnd.side = 5;
                                    shift = -1;
                                }
                                index_t kcnst = origin_k - k_lo + shift;
                                index_t istart = origin_i - i_lo;
                                index_t iend = istart + ref_size_i - 1;
                                index_t jstart = origin_j - j_lo;
                                index_t jend = jstart + ref_size_j - 1;
                                for (index_t jidx = jstart; jidx < jend; ++jidx)
                                {
                                    index_t joffset = jidx*iwidth + kcnst*iwidth*jwidth;
                                    for (index_t iidx = istart; iidx < iend; ++iidx)
                                    {
                                        index_t offset = joffset + iidx;
                                        pbnd.m_elems.push_back(offset);
                                        pbnd.m_bface[offset] =
                                            pbnd.side == 4 ?
                                            poly_elems[offset][4] :
                                            poly_elems[offset][5];
                                    }
                                }
                            }
                            else
                            {
                                pbnd.side = -1;
                            }


                            std::ostringstream nbr_oss;
                            nbr_oss << "domain_" << std::setw(6)
                                    << std::setfill('0') << nbr_id;
                            std::string nbr_name = nbr_oss.str();

                            if (in_parent->has_child(nbr_name))
                            {
                                const Node& nbr_dom =
                                   (*in_parent)[nbr_name];

                                const Node& ntopo =
                                   nbr_dom["topologies"][name];

                                index_t ni_lo = ntopo["elements/origin/i0"].to_index_t();
                                index_t nj_lo = ntopo["elements/origin/j0"].to_index_t();
                                index_t nk_lo = ntopo["elements/origin/k0"].to_index_t();

                                index_t nbr_iwidth =
                                   ntopo["elements/dims/i"].to_index_t();
                                index_t nbr_jwidth =
                                   ntopo["elements/dims/j"].to_index_t();
                                index_t nbr_kwidth =
                                   ntopo["elements/dims/k"].to_index_t();

                                auto& nbr_elems = poly_elems_map[nbr_id];

                                if (pbnd.side >= 0)
                                {
                                    pbnd.m_nbr_rank = par_rank;
                                    pbnd.m_nbr_id = nbr_id;
                                    match_nbr_elems(pbnd, nbr_elems, allfaces_map, in_topo,
                                                    ref_win, nbr_win,
                                                    nbr_iwidth, nbr_jwidth,
                                                    nbr_kwidth,
                                                    ni_lo, nj_lo, nk_lo,
                                                    ratio_i,ratio_j,ratio_k,
                                                    true);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::map<index_t, std::map<index_t, std::map<index_t,index_t> > > dom_to_nbr_to_buffidx;
    std::map<index_t, std::map<index_t, std::vector<double> > > dom_to_nbr_to_xbuffer;
    std::map<index_t, std::map<index_t, std::vector<double> > > dom_to_nbr_to_ybuffer;
    std::map<index_t, std::map<index_t, std::vector<double> > > dom_to_nbr_to_zbuffer;

    itr = n.children();
    while(itr.has_next())
    {
        const Node& chld = itr.next();
        std::string domain_name = itr.name();

        index_t domain_id = chld["state/domain_id"].to_index_t();
        const Node& in_topo = chld["topologies"][name];

        std::ostringstream win_oss;
        win_oss << "window_" << std::setw(6) << std::setfill('0') << domain_id;
        std::string win_name = win_oss.str();

        const Node* in_parent = chld.parent();

        auto& nbr_to_buffidx = dom_to_nbr_to_buffidx[domain_id];
        auto& nbr_to_xbuffer = dom_to_nbr_to_xbuffer[domain_id];
        auto& nbr_to_ybuffer = dom_to_nbr_to_ybuffer[domain_id];
        auto& nbr_to_zbuffer = dom_to_nbr_to_zbuffer[domain_id];

        if (chld.has_path("adjsets/adjset/groups"))
        {
            const Node& in_groups = chld["adjsets/adjset/groups"];
            NodeConstIterator grp_itr = in_groups.children();
            while(grp_itr.has_next())
            {
                const Node& group = grp_itr.next();

                if (group.has_child("neighbors") && group.has_child("windows"))
                {
                    temp.reset();
                    temp.set_external(DataType(group["neighbors"].dtype().id(), 1),
                                      (void*)group["neighbors"].element_ptr(1));
                    const index_t nbr_id = temp.to_index_t();
                    const Node& in_windows = group["windows"];
                    std::ostringstream nw_oss;
                    nw_oss << "window_" << std::setw(6)
                            << std::setfill('0') << nbr_id;
                    std::string nbr_win_name = nw_oss.str();

                    const Node& ref_win = in_windows[win_name];
                    const Node& nbr_win = in_windows[nbr_win_name];
                    if (nbr_win["level_id"].to_index_t() > ref_win["level_id"].to_index_t())
                    {
                        index_t ratio_i = nbr_win["ratio/i"].to_index_t();
                        index_t ratio_j = nbr_win["ratio/j"].to_index_t();
                        index_t ratio_k = nbr_win["ratio/k"].to_index_t();

                        index_t ref_size_i = ref_win["dims/i"].to_index_t();
                        index_t ref_size_j = ref_win["dims/j"].to_index_t();
                        index_t ref_size_k = ref_win["dims/k"].to_index_t();
                        index_t ref_size = ref_size_i*ref_size_j*ref_size_k;

                        index_t nbr_size_i = nbr_win["dims/i"].to_index_t();
                        index_t nbr_size_j = nbr_win["dims/j"].to_index_t();
                        index_t nbr_size_k = nbr_win["dims/k"].to_index_t();
                        index_t nbr_size = nbr_size_i*nbr_size_j*nbr_size_k;

                        if (nbr_size > ref_size)
                        {
                            std::vector<index_t> vertices;
                            auto& buffidx = nbr_to_buffidx[nbr_id];
                            auto& xbuffer = nbr_to_xbuffer[nbr_id];
                            auto& ybuffer = nbr_to_ybuffer[nbr_id];
                            auto& zbuffer = nbr_to_zbuffer[nbr_id];

                            std::ostringstream nbr_oss;
                            nbr_oss << "domain_" << std::setw(6)
                                    << std::setfill('0') << nbr_id;
                            std::string nbr_name = nbr_oss.str();

                            if (!in_parent->has_child(nbr_name))
                            {
                                PolyBndry& pbnd = poly_bndry_map[domain_id][nbr_id];

                                if (pbnd.side >= 0)
                                {
                                    index_t nbr_rank = group["rank"].to_index_t();
                                    index_t buffer[6];

                                    MPI_Recv(buffer,
                                             6,
                                             MPI_INT64_T,
                                             nbr_rank,
                                             nbr_id,
                                             comm,
                                             MPI_STATUS_IGNORE);

                                    index_t nbr_iwidth = buffer[0];
                                    index_t nbr_jwidth = buffer[1];
                                    index_t nbr_kwidth = buffer[2];
                                    index_t ni_lo = buffer[3];
                                    index_t nj_lo = buffer[4];
                                    index_t nk_lo = buffer[5];

                                    auto& nbr_elems = poly_elems_map[nbr_id];

                                    pbnd.m_nbr_rank = nbr_rank;
                                    pbnd.m_nbr_id = nbr_id;
                                    match_nbr_elems(pbnd, nbr_elems, allfaces_map, in_topo,
                                                    ref_win, nbr_win,
                                                    nbr_iwidth, nbr_jwidth,
                                                    nbr_kwidth,
                                                    ni_lo, nj_lo, nk_lo,
                                                    ratio_i,ratio_j,ratio_k,
                                                    false);
                                    if (nbr_size_i == 1)
                                    {
                                        vertices.resize(nbr_size_j*nbr_size_k);
                                        xbuffer.resize(nbr_size_j*nbr_size_k);
                                        ybuffer.resize(nbr_size_j*nbr_size_k);
                                        zbuffer.resize(nbr_size_j*nbr_size_k);
                                    }
                                    else if (nbr_size_j == 1)
                                    {
                                        vertices.resize(nbr_size_i*nbr_size_k);
                                        xbuffer.resize(nbr_size_i*nbr_size_k);
                                        ybuffer.resize(nbr_size_i*nbr_size_k);
                                        zbuffer.resize(nbr_size_i*nbr_size_k);
                                    }
                                    else
                                    {
                                        vertices.resize(nbr_size_i*nbr_size_j);
                                        xbuffer.resize(nbr_size_i*nbr_size_j);
                                        ybuffer.resize(nbr_size_i*nbr_size_j);
                                        zbuffer.resize(nbr_size_i*nbr_size_j);
                                    }
    
                                    MPI_Recv(&vertices[0],
                                             vertices.size(),
                                             MPI_INT64_T,
                                             nbr_rank,
                                             nbr_id,
                                             comm,
                                             MPI_STATUS_IGNORE);
                                    MPI_Recv(&xbuffer[0],
                                             xbuffer.size(),
                                             MPI_DOUBLE,
                                             nbr_rank,
                                             nbr_id,
                                             comm,
                                             MPI_STATUS_IGNORE);
                                    MPI_Recv(&ybuffer[0],
                                             ybuffer.size(),
                                             MPI_DOUBLE,
                                             nbr_rank,
                                             nbr_id,
                                             comm,
                                             MPI_STATUS_IGNORE);
                                    MPI_Recv(&zbuffer[0],
                                             zbuffer.size(),
                                             MPI_DOUBLE,
                                             nbr_rank,
                                             nbr_id,
                                             comm,
                                             MPI_STATUS_IGNORE);
                                    index_t v = 0;
                                    for (auto vitr = vertices.begin();
                                         vitr != vertices.end(); ++vitr)
                                    {
                                        buffidx[*vitr] = v;
                                        ++v;
                                    }
                                }
                            }
                            else
                            {
                                const Node& nbr_dom =
                                   (*in_parent)[nbr_name];
                                const Node& nbr_coords =
                                   nbr_dom["coordsets/coords"];

                                const Node& ntopo =
                                   nbr_dom["topologies"][name];
                                index_t ni_lo =
                                   ntopo["elements/origin/i0"].to_index_t();
                                index_t nj_lo =
                                   ntopo["elements/origin/j0"].to_index_t();
                                index_t nk_lo =
                                   ntopo["elements/origin/k0"].to_index_t();
                                index_t nbr_iwidth =
                                   ntopo["elements/dims/i"].to_index_t() + 1;
                                index_t nbr_jwidth =
                                   ntopo["elements/dims/j"].to_index_t() + 1;

                                const Node& fcoords =
                                   nbr_coords["values"];
                                const double_array& xarray =
                                   fcoords["x"].as_double_array();
                                const double_array& yarray =
                                   fcoords["y"].as_double_array();
                                const double_array& zarray =
                                   fcoords["z"].as_double_array();

                                index_t origin_i = nbr_win["origin/i"].to_index_t();
                                index_t origin_j = nbr_win["origin/j"].to_index_t();
                                index_t origin_k = nbr_win["origin/k"].to_index_t();

                                index_t istart = origin_i - ni_lo;
                                index_t jstart = origin_j - nj_lo;
                                index_t kstart = origin_k - nk_lo;
                                index_t iend = istart + nbr_size_i;
                                index_t jend = jstart + nbr_size_j;
                                index_t kend = kstart + nbr_size_k;

                                for (index_t kidx = kstart; kidx < kend; ++kidx)
                                {
                                    index_t koffset = kidx*nbr_iwidth*nbr_jwidth;
                                    for (index_t jidx = jstart; jidx < jend; ++jidx)
                                    {
                                        index_t joffset = jidx*nbr_iwidth;
                                        for (index_t iidx = istart; iidx < iend; ++iidx)
                                        {
                                            index_t offset = koffset+joffset+iidx;
                                            index_t new_idx = xbuffer.size();
                                            buffidx[offset] = new_idx;
                                            xbuffer.push_back(xarray[offset]);
                                            ybuffer.push_back(yarray[offset]);
                                            zbuffer.push_back(zarray[offset]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::map<index_t, std::map<index_t, std::map<index_t,index_t> > > dom_to_nbr_to_sharedmap;

    itr = n.children();
    while(itr.has_next())
    {
        const Node& chld = itr.next();
        std::string domain_name = itr.name();

        index_t domain_id = chld["state/domain_id"].to_index_t();

        const Node& dom_coords = chld["coordsets/coords"];

        const double_array& ref_xarray =
           dom_coords["values/x"].as_double_array();
        const double_array& ref_yarray =
           dom_coords["values/y"].as_double_array();
        const double_array& ref_zarray =
           dom_coords["values/z"].as_double_array();

        Node& out_coords = dest[domain_name]["coordsets/coords"];

        std::vector<double> out_xvec;
        std::vector<double> out_yvec;
        std::vector<double> out_zvec;
        if (dom_coords["type"].as_string() == "uniform")
        {
            Node tmp_coords;
            blueprint::mesh::coordset::uniform::to_explicit(dom_coords, tmp_coords);
            const Node& tmp_x = tmp_coords["values/x"];
            const Node& tmp_y = tmp_coords["values/y"];
            const Node& tmp_z = tmp_coords["values/z"];
            const double* tmp_x_ptr = static_cast< const double*>(tmp_x.data_ptr());
            const double* tmp_y_ptr = static_cast< const double*>(tmp_y.data_ptr());
            const double* tmp_z_ptr = static_cast< const double*>(tmp_z.data_ptr());
            index_t num_coords = tmp_x.dtype().number_of_elements();
            out_xvec.insert(out_xvec.end(), tmp_x_ptr, tmp_x_ptr + num_coords);
            out_yvec.insert(out_yvec.end(), tmp_y_ptr, tmp_y_ptr + num_coords);
            out_zvec.insert(out_zvec.end(), tmp_z_ptr, tmp_z_ptr + num_coords);
            out_coords["type"] = "explicit"; 
        }
        else
        {
            out_coords["type"] = dom_coords["type"];
            const double* tmp_x_ptr = static_cast< const double*>(ref_xarray.data_ptr());
            const double* tmp_y_ptr = static_cast< const double*>(ref_yarray.data_ptr());
            const double* tmp_z_ptr = static_cast< const double*>(ref_zarray.data_ptr());

            index_t num_coords = ref_xarray.dtype().number_of_elements();
            out_xvec.insert(out_xvec.end(), tmp_x_ptr, tmp_x_ptr + num_coords);
            out_yvec.insert(out_yvec.end(), tmp_y_ptr, tmp_y_ptr + num_coords);
            out_zvec.insert(out_zvec.end(), tmp_z_ptr, tmp_z_ptr + num_coords);
        }


        auto& poly_elems = poly_elems_map[domain_id];
        auto& new_vertices = new_vert_map[domain_id];
        auto& new_subelems = new_face_map[domain_id];
        auto& allfaces = allfaces_map[domain_id];

        auto& nbr_to_sharedmap = dom_to_nbr_to_sharedmap[domain_id];
        auto& nbr_to_buffidx = dom_to_nbr_to_buffidx[domain_id];
        auto& nbr_to_xbuffer = dom_to_nbr_to_xbuffer[domain_id];
        auto& nbr_to_ybuffer = dom_to_nbr_to_ybuffer[domain_id];
        auto& nbr_to_zbuffer = dom_to_nbr_to_zbuffer[domain_id];

        std::set<index_t> elems_on_bdry;
        std::set<index_t> multi_nbrs;

        if (poly_bndry_map.find(domain_id) != poly_bndry_map.end())
        {
            std::map<index_t, std::map<index_t, size_t> > elem_to_new_faces;

            //std::map<index_t, PolyBndry> bndries
            auto& bndries = poly_bndry_map[domain_id];
            for (auto bitr = bndries.begin(); bitr != bndries.end(); ++bitr)
            {
                //One PolyBndry for each fine neighbor domain
                PolyBndry& pbnd = bitr->second;
                index_t nbr_id = pbnd.m_nbr_id; //domain id of nbr

                auto& sharedmap = nbr_to_sharedmap[nbr_id];
                auto& buffidx = nbr_to_buffidx[nbr_id];
                auto& xbuffer = nbr_to_xbuffer[nbr_id];
                auto& ybuffer = nbr_to_ybuffer[nbr_id];
                auto& zbuffer = nbr_to_zbuffer[nbr_id];

                index_t orig_num_vertices = out_xvec.size();

                std::map<index_t, std::map<index_t, std::vector<index_t> > >fv_map;

                auto& nbr_elems = pbnd.m_nbr_elems;
                for (auto eitr = nbr_elems.begin(); eitr != nbr_elems.end(); ++eitr) 
                {
                    //offset for a single domain elem
                    index_t ref_offset = eitr->first;

                    auto& num_new_faces = elem_to_new_faces[ref_offset];
                    if (num_new_faces.find(pbnd.side) == num_new_faces.end())
                    {
                        num_new_faces[pbnd.side] = 0;
                    }

                    //Holds neighbor elem offsets
                    std::vector<index_t>& nbrs = eitr->second;
                    size_t num_nbrs = nbrs.size();
                    for (size_t n = 0; n < num_nbrs; ++n)
                    {
                        index_t nbr_elem = nbrs[n];

                        //nbr_face is subelem offset for the face of
                        //nbr_elem touching the boundary
                        index_t nbr_face = pbnd.m_nbr_faces[ref_offset][nbr_elem].m_face_id;
                        //face_subelem holds the vertices for the nbr_face
                        auto& face_subelem = pbnd.m_nbr_faces[ref_offset][nbr_elem].m_fine_subelem;

                        //subelem at ref/nbr interface
                        fv_map[ref_offset][nbr_face] = face_subelem;
                    }
                }

                std::map<index_t, std::set<index_t> > new_nbr_vertices;

                //fv_map maps a coarse subelem to a vector of its
                //fine subelems;
                for (auto fitr = fv_map.begin(); fitr != fv_map.end(); ++fitr)
                {
                    index_t ref_offset = fitr->first;
                    bputils::connectivity::ElemType& ref_elem = poly_elems[ref_offset];
                    index_t ref_face = ref_elem[pbnd.side];
                    std::vector<index_t>& ref_subelem = allfaces[ref_face];

                    auto& new_nbr_verts = new_nbr_vertices[ref_offset];
                    std::map<index_t, index_t> face_verts;

                    auto nbrs = fitr->second;
                    std::set<index_t> sh_verts;
                    std::set<index_t> others;

                    // This loop causes sh_verts to be filled with the
                    // vertices that only exist once in the neighbor
                    // subelems.  Those vertices are the ones shared by
                    // the reference subelem.
                    for (auto nitr = nbrs.begin(); nitr != nbrs.end(); ++nitr)
                    {
                        std::vector<index_t>& nbr_subelem = nitr->second;
                        for (auto nv = nbr_subelem.begin(); nv != nbr_subelem.end(); ++nv)
                        {
                            index_t nvert = *nv;
                            if (sh_verts.find(nvert) == sh_verts.end())
                            {
                                if (others.find(nvert) == others.end())
                                {
                                    sh_verts.insert(nvert);
                                }
                                else
                                {
                                    ++face_verts[nvert];
                                } 
                            }
                            else
                            {
                                sh_verts.erase(nvert);
                                others.insert(nvert);
                                face_verts[nvert] = 1;
                            }
                        }
                    }
                    for (auto fv = face_verts.begin(); fv != face_verts.end();
                         ++fv)
                    {
                        if (fv->second == 1)
                        {
                            new_nbr_verts.insert(fv->first);
                        }
                    }

                    double eps = sqrt(std::numeric_limits<double>::epsilon());

                    // Determine which reference vertex is shared with
                    // each neighbor vertex.
                    for (auto rv = ref_subelem.begin(); rv != ref_subelem.end(); ++rv)
                    {
                        double min_dist_sqr = std::numeric_limits<double>::max();
                        index_t shared_vert = -1;
                        for (auto sv = sh_verts.begin(); sv != sh_verts.end();
                             ++sv)
                        {
                            index_t nvert = *sv; 

                            index_t buf_offset = buffidx[nvert];
                            double nbr_x = xbuffer[buf_offset];
                            double nbr_y = ybuffer[buf_offset];
                            double nbr_z = zbuffer[buf_offset];

                            double xdiff = nbr_x-ref_xarray[*rv];
                            double ydiff = nbr_y-ref_yarray[*rv];
                            double zdiff = nbr_z-ref_zarray[*rv];

                            double dist_sqr = xdiff*xdiff + ydiff*ydiff + 
                                              zdiff*zdiff;

                            if (dist_sqr < min_dist_sqr)
                            {
                                min_dist_sqr = dist_sqr;
                                shared_vert = nvert;
                            }
                        }

                        if (min_dist_sqr < eps)
                        {
                            sharedmap[shared_vert] = *rv;
                            sh_verts.erase(shared_vert);
                        }
                    }
                }

                index_t num_vertices = out_xvec.size();

                //Replace the coarse subelem with new fine subelems
                for (auto eitr = nbr_elems.begin(); eitr != nbr_elems.end(); ++eitr)
                {
                    index_t ref_offset = eitr->first;
                    auto& ref_elem = poly_elems[ref_offset];
                    auto& new_nbr_verts = new_nbr_vertices[ref_offset];
                    auto& new_verts = new_vertices[ref_offset];

                    index_t ref_face = ref_elem[pbnd.side];

                    auto& new_subs = new_subelems[ref_offset];

                    auto& nbr_subelems = fv_map[ref_offset];

                    auto& num_added_faces =
                        elem_to_new_faces[ref_offset][pbnd.side];

                    if (elems_on_bdry.find(ref_offset) == elems_on_bdry.end())
                    {
                        elems_on_bdry.insert(ref_offset);
                    }
                    else
                    {
                        multi_nbrs.insert(ref_offset);
                    }

                    // Set to true if the current neighbor domain
                    // has faces that cover only part of the current coarse
                    // face.  Another neighbor domain will provide faces
                    // to cover the remainder, and we need to keep track
                    // of all new vertices added. 
                    bool partial_nbr = false;
                    if (nbr_subelems.size() != pbnd.m_nbrs_per_face)
                    {
                        partial_nbr = true;
                    }

                    index_t last_added_face = allfaces.rbegin()->first;
                    for (auto nb = nbr_subelems.begin();
                         nb != nbr_subelems.end(); ++nb)
                    {
                        std::vector<index_t> new_face;
                        auto& n_subelem = nb->second;

                        for (unsigned int v = 0; v < n_subelem.size(); ++v)
                        {
                            index_t n_vtx = n_subelem[v];
                            if (sharedmap.find(n_vtx) == sharedmap.end())
                            {
                                new_face.push_back(num_vertices);
                                sharedmap[n_vtx] = num_vertices;

                                ++num_vertices;

                                index_t buf_offset = buffidx[n_vtx];
                                out_xvec.push_back(xbuffer[buf_offset]); 
                                out_yvec.push_back(ybuffer[buf_offset]); 
                                out_zvec.push_back(zbuffer[buf_offset]);
                                if (partial_nbr)
                                {
                                    new_verts.insert(new_face.back());
                                }
                            }
                            else
                            {
                                new_face.push_back(sharedmap[n_vtx]);
                                if (partial_nbr &&
                                    new_face.back() > orig_num_vertices)
                                {
                                    new_verts.insert(new_face.back());
                                }
                            }
                            if (!partial_nbr &&
                                new_nbr_verts.find(n_vtx) !=
                                new_nbr_verts.end())
                            {   
                                new_verts.insert(new_face.back());
                            }
                        }
                        ++num_added_faces;
                        ++last_added_face;
                        allfaces[last_added_face] = new_face;
                        ref_elem.push_back(last_added_face);
                        new_subs.insert(last_added_face);
                    }

                    if (num_added_faces == pbnd.m_nbrs_per_face)
                    {
                        allfaces[ref_face] = allfaces[last_added_face];
                        allfaces.erase(last_added_face);
                        ref_elem.pop_back();
                        new_subs.erase(last_added_face);
                        new_subs.insert(ref_face);
                    }
                }
            }
        }

        out_coords["values/x"].set(out_xvec);
        out_coords["values/y"].set(out_yvec);
        out_coords["values/z"].set(out_zvec);
        const double_array& out_xarray =
           out_coords["values/x"].as_double_array();
        const double_array& out_yarray =
           out_coords["values/y"].as_double_array();
        const double_array& out_zarray =
           out_coords["values/z"].as_double_array();

        // Elements that had vertices contributed by multiple neighbor
        // domains need a fix-up step.
        for (auto mn = multi_nbrs.begin(); mn != multi_nbrs.end(); ++mn)
        {
            const index_t& offset = *mn;
            fix_duplicated_vertices(new_vertices[offset], poly_elems[offset],
                                    allfaces, out_xarray, out_yarray,
                                    out_zarray);
        }
    }

    itr = n.children();
    while(itr.has_next())
    {
        const Node& chld = itr.next();
        std::string domain_name = itr.name();

        index_t domain_id = chld["state/domain_id"].to_index_t();

        const Node& in_topo = chld["topologies"][name];

        index_t iwidth = in_topo["elements/dims/i"].to_index_t();
        index_t jwidth = in_topo["elements/dims/j"].to_index_t();
        index_t kwidth = in_topo["elements/dims/k"].to_index_t();

        index_t istride = 1;
        index_t jstride = iwidth+1;
        index_t kstride = (iwidth+1)*(jwidth+1);

        const auto& ratio = nbr_ratio[domain_id];

        Node& out_coords = dest[domain_name]["coordsets/coords"];

        const double_array& xarray =
           out_coords["values/x"].as_double_array();
        const double_array& yarray =
           out_coords["values/y"].as_double_array();
        const double_array& zarray =
           out_coords["values/z"].as_double_array();

        auto& poly_elems = poly_elems_map[domain_id];
        auto& new_vertices = new_vert_map[domain_id];
        auto& new_subelems = new_face_map[domain_id];
        auto& allfaces = allfaces_map[domain_id];

        index_t elemsize = iwidth*jwidth*kwidth;

        for (index_t elem = 0; elem < elemsize; ++elem)
        {
            auto& poly_elem = poly_elems[elem];

            //Unchanged elems have 6 sides, no need to do more
            if (poly_elem.size() > 6)
            {
                auto& new_faces = new_subelems[elem];

                auto& new_verts = new_vertices[elem];

                for (auto pi = poly_elem.begin(); pi != poly_elem.end();
                     ++pi)
                {
                    index_t pface = *pi;
                    //only if "pface" is an old (coarse) subelem 
                    if (new_faces.find(pface) == new_faces.end())
                    {
                        auto& subelem = allfaces[pface];
                        if (subelem.size() > 4)
                        {
                            //This subelem has already added new vertices
                            continue;
                        }

                        //new subelems that are adjacent to pface.
                        std::set<index_t> edge_verts;
                        for (auto vi = subelem.begin(); vi != subelem.end();
                             ++vi)
                        {
                            for (auto si = new_faces.begin();
                                 si != new_faces.end(); ++si)
                            {
                                auto& nface = allfaces[*si];
                                for (auto ni = nface.begin();
                                     ni != nface.end(); ++ni)
                                {
                                    if (*ni == *vi)
                                    {
                                        edge_verts.insert(*vi);
                                    }
                                }
                            }
                        }

                        if (edge_verts.empty())
                        {
                            continue;
                        }

                        
                        size_t num_edges;
                        if (edge_verts.size() == 4)
                        {
                            //Special case when all 4 vertices of subelem
                            //are selected.  There may be only 3 edges that
                            //get new vertices but we have to check them all.
                            num_edges = edge_verts.size();
                        }
                        else
                        {
                            num_edges = edge_verts.size() - 1;
                        }

                        size_t edge_count = 0;
                        std::vector<std::vector<index_t> > edges(num_edges);
 
                        for (auto vi = subelem.begin(); vi != subelem.end();
                             ++vi)
                        {
                            if (edge_count == num_edges)
                            {
                                break;
                            }
                            auto next = vi + 1;
                            if (next == subelem.end())
                            {
                                next = subelem.begin();
                            }
                            if (edge_verts.find(*vi) != edge_verts.end() &&
                                edge_verts.find(*next) != edge_verts.end())
                            {
                                edges[edge_count].push_back(*vi);
                                edges[edge_count].push_back(*next);
                                ++edge_count;
                            }
                        }
               

                        //The edges are going to get more vertices.
                        //Figure out which of the "new verts" fit on these
                        //edges.

                        for (auto ei = edges.begin(); ei != edges.end();
                             ++ei)
                        {
                            auto this_edge = *ei;

                            index_t stride =
                                std::abs(this_edge[1]-this_edge[0]);

                            index_t edge_ratio;
                            if (stride == istride)
                            {
                                edge_ratio = ratio[0]; 
                            }
                            else if (stride == jstride)
                            {
                                edge_ratio = ratio[1];
                            } 
                            else
                            {
                                assert(stride == kstride);
                                edge_ratio = ratio[2];
                            }

                            //Test each new_vert point (xn,yn,zn) to see
                            //if it's colinear with (x0,y0,z0) and (x1,y1,z1).
                            //
                            //We need something better than these
                            //epsilon-based floating point comparisons
                            double x0 = xarray[this_edge[0]];
                            double y0 = yarray[this_edge[0]];
                            double z0 = zarray[this_edge[0]];
                            double x1 = xarray[this_edge[1]];
                            double y1 = yarray[this_edge[1]];
                            double z1 = zarray[this_edge[1]];

                            double eps = sqrt(std::numeric_limits<double>::epsilon());
                            double xedge = x1-x0+eps;
                            double yedge = y1-y0+eps;
                            double zedge = z1-z0+eps;

                            std::list<index_t> add_verts;
                            std::multimap<double,index_t> test_verts;
                            size_t verts_needed =
                                static_cast<size_t>(edge_ratio-1);
                            for (auto nv = new_verts.begin(); nv != new_verts.end();
                                 ++nv)
                            {
                                index_t nvert = *nv;
                                double xn = xarray[nvert];
                                double yn = yarray[nvert];
                                double zn = zarray[nvert];
                                double xnedge = (xn-x0)*(xn-x0) > eps ? xn-x0 : 0.0;
                                double ynedge = (yn-y0)*(yn-y0) > eps ? yn-y0 : 0.0;
                                double znedge = (zn-z0)*(zn-z0) > eps ? zn-z0 : 0.0;
                                double edgesum = xnedge+ynedge+znedge;

                                double xlam = xnedge/xedge;
                                double ylam = ynedge/yedge;
                                double zlam = znedge/zedge;
                                double lam = (xnedge+ynedge+znedge) /
                                              (xedge+yedge+zedge);
                                double testsum = 0.0;
                                if (xlam*xlam > eps)
                                {
                                    testsum += (xlam-lam)*(xlam-lam);
                                }
                                if (ylam*ylam > eps)
                                {
                                    testsum += (ylam-lam)*(ylam-lam);
                                }
                                if (zlam*zlam > eps)
                                {
                                    testsum += (zlam-lam)*(zlam-lam);
                                }
                                if (testsum > eps && edgesum/testsum < eps)
                                {
                                    continue;
                                }
                                else
                                {
                                    test_verts.emplace(testsum, nvert);
                                }
                            }

                            assert(test_verts.size() >= verts_needed ||
                                   num_edges == 4);

                            for (auto tv = test_verts.begin();
                                 tv != test_verts.end(); ++tv)
                            {
                                add_verts.push_back(tv->second);
                                if (add_verts.size() == verts_needed)
                                {
                                    break; 
                                }
                            }

                            if (add_verts.size() != 1)
                            {
                                std::multimap<double,index_t> map_verts;

                                std::list<index_t> tmp_list;
                                tmp_list.swap(add_verts);
                                for (auto av = tmp_list.begin();
                                     av != tmp_list.end(); ++av)
                                {
                                    double xa = xarray[*av];
                                    double ya = yarray[*av];
                                    double za = zarray[*av];

                                    double dstsq = (xa-x0)*(xa-x0) +
                                                   (ya-y0)*(ya-y0) +
                                                   (za-z0)*(za-z0);
                                    map_verts.emplace(dstsq, *av);
                                }
                                for (auto mv = map_verts.begin();
                                     mv != map_verts.end(); ++mv) 
                                {
                                    add_verts.push_back(mv->second);
                                    new_verts.erase(mv->second);
                                }
                            }
                            else
                            {
                                new_verts.erase(add_verts.front());
                            }

                            if (this_edge[0] == subelem.back())
                            {
                                subelem.insert(subelem.end(),
                                    add_verts.begin(), add_verts.end());
                            }
                            else
                            {
                                index_t ctr = 0; 
                                for (auto vi = subelem.begin(); vi != subelem.end();
                                     ++vi)
                                {
                                    if (this_edge[0] == *vi)
                                    {
                                        subelem.insert(vi+1,
                                            add_verts.begin(), add_verts.end());
                                        break;
                                    }
                                    ++ctr;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    itr = n.children();
    while(itr.has_next())
    {
        const Node& chld = itr.next();
        std::string domain_name = itr.name();

        index_t domain_id = chld["state/domain_id"].to_index_t();

        const Node& in_topo = chld["topologies"][name];

        index_t iwidth = in_topo["elements/dims/i"].to_index_t();
        index_t jwidth = in_topo["elements/dims/j"].to_index_t();
        index_t kwidth = in_topo["elements/dims/k"].to_index_t();

        auto& poly_elems = poly_elems_map[domain_id];
        auto& allfaces = allfaces_map[domain_id];

        index_t elemsize = iwidth*jwidth*kwidth;

        std::vector<index_t>& e_connect = elem_connect[domain_id];
        std::vector<index_t>& e_sizes = elem_sizes[domain_id];
        std::vector<index_t>& e_offsets = elem_offsets[domain_id];
        std::vector<index_t>& sub_connect = subelem_connect[domain_id];
        std::vector<index_t>& sub_sizes = subelem_sizes[domain_id];
        std::vector<index_t>& sub_offsets = subelem_offsets[domain_id];
        index_t elem_offset_sum = 0;
        index_t subelem_offset_sum = 0;
        for (index_t elem = 0; elem < elemsize; ++elem)
        {
            auto& poly_elem = poly_elems[elem];
            e_connect.insert(e_connect.end(), poly_elem.begin(), poly_elem.end());
            e_sizes.push_back(poly_elem.size());
            e_offsets.push_back(elem_offset_sum);
            elem_offset_sum += e_sizes.back();
        }
        for (auto if_itr = allfaces.begin(); if_itr != allfaces.end(); ++if_itr)
        {
            auto& if_elem = if_itr->second;
            sub_connect.insert(sub_connect.end(), if_elem.begin(), if_elem.end());
            sub_sizes.push_back(if_elem.size());
            sub_offsets.push_back(subelem_offset_sum);
            subelem_offset_sum += sub_sizes.back();
        }
    }

    itr = n.children();
    while(itr.has_next())
    {
        const Node& chld = itr.next();
        std::string domain_name = itr.name();

        index_t domain_id = chld["state/domain_id"].to_index_t();

        std::vector<index_t>& e_connect = elem_connect[domain_id];
        std::vector<index_t>& e_sizes = elem_sizes[domain_id];
        std::vector<index_t>& e_offsets = elem_offsets[domain_id];
        std::vector<index_t>& sub_connect = subelem_connect[domain_id];
        std::vector<index_t>& sub_sizes = subelem_sizes[domain_id];
        std::vector<index_t>& sub_offsets = subelem_offsets[domain_id];

        Node& topo = dest[domain_name]["topologies"][name];
        const Node& in_topo = chld["topologies"][name];

        topo["coordset"] = in_topo["coordset"];

        topo["type"] = "unstructured";
        topo["elements/shape"] = "polyhedral";
        topo["elements/shape"].set_string("polyhedral");
        topo["elements/connectivity"].set(e_connect);
        topo["elements/sizes"].set(e_sizes);
        topo["elements/offsets"].set(e_offsets);
        topo["subelements/shape"].set_string("polygonal");
        topo["subelements/connectivity"].set(sub_connect);
        topo["subelements/sizes"].set(sub_sizes);
        topo["subelements/offsets"].set(sub_offsets);

    }
}

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

//-------------------------------------------------------------------------
void
flatten(const conduit::Node &mesh, const conduit::Node &options,
    conduit::Node &output, MPI_Comm comm)
{
    output.reset();

    ParallelMeshFlattener do_flatten(comm);
    do_flatten.set_options(options);
    do_flatten.execute(mesh, output);
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
