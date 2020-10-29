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

    Node do_par_map;
    do_par_map.set_uint8(0);
    if(par_rank == selected_rank )
    {
        if(::conduit::blueprint::mesh::is_multi_domain(mesh))
        {
            if(mesh.child(0).has_child("state") &&
               mesh.child(0)["state"].has_child("domain_id"))
            {
                do_par_map.set_uint8(1);
            }
        }
    }

    relay::mpi::broadcast(do_par_map, selected_rank, comm); 

    Node partition;
    if(do_par_map.as_uint8())
    {
        generate_partition(mesh, partition, comm);
    }

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

        if (!partition.dtype().is_empty())
        {
            index_out["state/partition_size"].set(par_size);
            index_out["state/domain_to_partition_map"].set(partition);
        }
    }

    // broadcast the resulting index to all other ranks
    relay::mpi::broadcast_using_schema(index_out,
                                       selected_rank,
                                       comm);
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


//-------------------------------------------------------------------------
void to_polygonal(const Node &n,
                  Node &dest,
                  const std::string& name)
{

    dest.reset();
    NodeConstIterator itr = n.children();

    std::map<int, std::map<int, std::vector<int64_t> > > poly_elems_map;

    while(itr.has_next())
    {
        const Node& chld = itr.next();
        std::string domain_name = itr.name();
        dest[domain_name]["state"] = chld["state"];
        const Node& in_coords = chld["coordsets/coords"];

        int64_t domain_id = chld["state/domain_id"].as_int64();
        int64_t level_id = chld["state/level_id"].as_int64();
        if (level_id == 0) continue; 

        std::ostringstream win_oss;
        win_oss << "window_" << std::setw(6) << std::setfill('0') << domain_id;
        std::string win_name = win_oss.str();

        const Node& in_topo = chld["topologies"][name];

        int64_t niwidth = in_topo["elements/dims/i"].as_int64() + 1;

        int64_t i_lo = in_topo["elements/origin/i0"].as_int64();
        int64_t j_lo = in_topo["elements/origin/j0"].as_int64();

        const Node* in_parent = chld.parent();

        if (chld.has_path("adjsets/adjset/groups")) {
            const Node& in_groups = chld["adjsets/adjset/groups"];
            NodeConstIterator grp_itr = in_groups.children();
            while(grp_itr.has_next())
            {
                const Node& group = grp_itr.next();

                if (group.has_child("neighbors") && group.has_child("windows"))
                {
                    int64_array neighbors =
                       group["neighbors"].as_int64_array();

                    int nbr_id = neighbors[1];

                    const Node& in_windows = group["windows"];
                    std::ostringstream nw_oss;
                    nw_oss << "window_" << std::setw(6)
                            << std::setfill('0') << nbr_id;
                    std::string nbr_win_name = nw_oss.str();
                    const Node& ref_win = in_windows[win_name];
                    const Node& nbr_win = in_windows[nbr_win_name];

                    if (nbr_win["level_id"].as_int64() < ref_win["level_id"].as_int64())
                    {

                        int64_t ref_size_i = ref_win["dims/i"].as_int64();
                        int64_t ref_size_j = ref_win["dims/j"].as_int64();
                        int64_t ref_size = ref_size_i * ref_size_j;

                        int64_t nbr_size_i = nbr_win["dims/i"].as_int64();
                        int64_t nbr_size_j = nbr_win["dims/j"].as_int64();
                        int64_t nbr_size = nbr_size_i * nbr_size_j;

                        std::ostringstream nbr_oss;
                        nbr_oss << "domain_" << std::setw(6)
                                << std::setfill('0') << nbr_id;
                        std::string nbr_name = nbr_oss.str();

                        if (nbr_size < ref_size && !in_parent->has_child(nbr_name))
                        {
                            std::vector<double> xbuffer;
                            std::vector<double> ybuffer;
                            const Node& fcoords =
                               in_coords["values"];
                            const double_array& xarray =
                               fcoords["x"].as_double_array();
                            const double_array& yarray =
                               fcoords["y"].as_double_array();

                            int64_t origin_i = ref_win["origin/i"].as_int64();
                            int64_t origin_j = ref_win["origin/j"].as_int64();

                            if (ref_size_i == 1) {
                               int icnst = origin_i - i_lo;
                               int jstart = origin_j - j_lo;
                               int jend = jstart + ref_size_j;
                               for (int jidx = jstart; jidx < jend; ++jidx) {
                                  int offset = jidx * niwidth + icnst;
                                  xbuffer.push_back(xarray[offset]);
                                  ybuffer.push_back(yarray[offset]);
                               }
                            } else if (ref_size_j == 1) {
                               int jcnst = origin_j - j_lo;
                               int istart = origin_i - i_lo;
                               int iend = istart + ref_size_i;
                               for (int iidx = istart; iidx < iend; ++iidx) {
                                  int offset = jcnst * niwidth + iidx;
                                  xbuffer.push_back(xarray[offset]);
                                  ybuffer.push_back(yarray[offset]);
                               }
                            }
                            int64_t nbr_rank = group["rank"].as_int64();
                            MPI_Send(&xbuffer[0],
                                     xbuffer.size(),
                                     MPI_DOUBLE,
                                     nbr_rank,
                                     domain_id,
                                     MPI_COMM_WORLD);
                            MPI_Send(&ybuffer[0],
                                     ybuffer.size(),
                                     MPI_DOUBLE,
                                     nbr_rank,
                                     domain_id,
                                     MPI_COMM_WORLD);
                        }
                    }
                }
            }
        }
    }

    std::map<int, std::map<int, std::vector<double> > > dom_to_nbr_to_xbuffer;
    std::map<int, std::map<int, std::vector<double> > > dom_to_nbr_to_ybuffer;

    itr = n.children();
    while(itr.has_next())
    {
        const Node& chld = itr.next();
        std::string domain_name = itr.name();

        int64_t domain_id = chld["state/domain_id"].as_int64();

        std::ostringstream win_oss;
        win_oss << "window_" << std::setw(6) << std::setfill('0') << domain_id;
        std::string win_name = win_oss.str();

        const Node* in_parent = chld.parent();

        auto& nbr_to_xbuffer = dom_to_nbr_to_xbuffer[domain_id];
        auto& nbr_to_ybuffer = dom_to_nbr_to_ybuffer[domain_id];

        if (chld.has_path("adjsets/adjset/groups"))
        {
            const Node& in_groups = chld["adjsets/adjset/groups"];
            NodeConstIterator grp_itr = in_groups.children();
            while(grp_itr.has_next())
            {
                const Node& group = grp_itr.next();

                if (group.has_child("neighbors") && group.has_child("windows"))
                {
                    int64_array neighbors = group["neighbors"].as_int64_array();

                    int64_t nbr_id = neighbors[1];
                    const Node& in_windows = group["windows"];
                    std::ostringstream nw_oss;
                    nw_oss << "window_" << std::setw(6)
                            << std::setfill('0') << nbr_id;
                    std::string nbr_win_name = nw_oss.str();

                    const Node& ref_win = in_windows[win_name];
                    const Node& nbr_win = in_windows[nbr_win_name];
                    if (nbr_win["level_id"].as_int64() > ref_win["level_id"].as_int64())
                    {
                        int64_t ref_size_i = ref_win["dims/i"].as_int64();
                        int64_t ref_size_j = ref_win["dims/j"].as_int64();
                        int64_t ref_size = ref_size_i * ref_size_j;

                        int64_t nbr_size_i = nbr_win["dims/i"].as_int64();
                        int64_t nbr_size_j = nbr_win["dims/j"].as_int64();
                        int64_t nbr_size = nbr_size_i * nbr_size_j;

                        if (nbr_size > ref_size)
                        {

                            auto& xbuffer = nbr_to_xbuffer[nbr_id];
                            auto& ybuffer = nbr_to_ybuffer[nbr_id];

                            std::ostringstream nbr_oss;
                            nbr_oss << "domain_" << std::setw(6)
                                    << std::setfill('0') << nbr_id;
                            std::string nbr_name = nbr_oss.str();

                            if (!in_parent->has_child(nbr_name))
                            {

                                if (nbr_size_i == 1) {
                                    xbuffer.resize(nbr_size_j);
                                    ybuffer.resize(nbr_size_j);
                                }
                                else if (nbr_size_j == 1)
                                {
                                    xbuffer.resize(nbr_size_i);
                                    ybuffer.resize(nbr_size_i);
                                }

                                int64_t nbr_rank = group["rank"].as_int64();
                                MPI_Recv(&xbuffer[0],
                                         xbuffer.size(),
                                         MPI_DOUBLE,
                                         nbr_rank,
                                         nbr_id,
                                         MPI_COMM_WORLD,
                                         MPI_STATUS_IGNORE);
                                MPI_Recv(&ybuffer[0],
                                         ybuffer.size(),
                                         MPI_DOUBLE, nbr_rank,
                                         nbr_id, MPI_COMM_WORLD,
                                         MPI_STATUS_IGNORE);

                            }
                            else
                            {
                                const Node& nbr_dom =
                                   (*in_parent)[nbr_name];
                                const Node& nbr_coords =
                                   nbr_dom["coordsets/coords"];

                                const Node& ntopo =
                                   nbr_dom["topologies"][name];
                                int64_t ni_lo =
                                   ntopo["elements/origin/i0"].as_int64();
                                int64_t nj_lo =
                                   ntopo["elements/origin/j0"].as_int64();
                                int64_t nbr_iwidth =
                                   ntopo["elements/dims/i"].as_int64() + 1;

                                const Node& fcoords =
                                   nbr_coords["values"];
                                const double_array& xarray =
                                   fcoords["x"].as_double_array();
                                const double_array& yarray =
                                   fcoords["y"].as_double_array();

                                int64_t origin_i = nbr_win["origin/i"].as_int64();
                                int64_t origin_j = nbr_win["origin/j"].as_int64();

                                int64_t istart = origin_i - ni_lo;
                                int64_t jstart = origin_j - nj_lo;
                                int64_t iend = istart + nbr_size_i;
                                int64_t jend = jstart + nbr_size_j;
                                for (int64_t jidx = jstart; jidx < jend; ++jidx)
                                {
                                    int64_t joffset = jidx*nbr_iwidth;
                                    for (int64_t iidx = istart; iidx < iend; ++iidx)
                                    {
                                        int64_t offset = joffset+iidx;
                                        xbuffer.push_back(xarray[offset]);
                                        ybuffer.push_back(yarray[offset]);
                                    }
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
        Node& out_coords = dest[domain_name]["coordsets/coords"];
        const Node& in_coords = chld["coordsets/coords"];

        int64_t domain_id = chld["state/domain_id"].as_int64();
        std::ostringstream win_oss;
        win_oss << "window_" << std::setw(6) << std::setfill('0') << domain_id;
        std::string win_name = win_oss.str();

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

        auto& nbr_to_xbuffer = dom_to_nbr_to_xbuffer[domain_id];
        auto& nbr_to_ybuffer = dom_to_nbr_to_ybuffer[domain_id];

        const Node& in_topo = chld["topologies"][name];

        int64_t iwidth = in_topo["elements/dims/i"].as_int64();
        int64_t jwidth = in_topo["elements/dims/j"].as_int64();

        int64_t i_lo = in_topo["elements/origin/i0"].as_int64();
        int64_t j_lo = in_topo["elements/origin/j0"].as_int64();

        auto& poly_elems = poly_elems_map[domain_id];

        if (chld.has_path("adjsets/adjset/groups"))
        {
            const Node& in_groups = chld["adjsets/adjset/groups"];
            NodeConstIterator grp_itr = in_groups.children();
            while(grp_itr.has_next())
            {
                const Node& group = grp_itr.next();

                if (group.has_child("neighbors") && group.has_child("windows"))
                {
                    int64_array neighbors = group["neighbors"].as_int64_array();

                    int nbr_id = neighbors[1];
                    const Node& in_windows = group["windows"];
                    std::ostringstream nw_oss;
                    nw_oss << "window_" << std::setw(6)
                            << std::setfill('0') << nbr_id;
                    std::string nbr_win_name = nw_oss.str();

                    const Node& ref_win = in_windows[win_name];
                    const Node& nbr_win = in_windows[nbr_win_name];
                    if (nbr_win["level_id"].as_int64() > ref_win["level_id"].as_int64())
                    {

                        int64_t ratio_i = nbr_win["ratio/i"].as_int64();
                        int64_t ratio_j = nbr_win["ratio/j"].as_int64();

                        int64_t ref_size_i = ref_win["dims/i"].as_int64();
                        int64_t ref_size_j = ref_win["dims/j"].as_int64();
                        int64_t ref_size = ref_size_i * ref_size_j;

                        int64_t nbr_size_i = nbr_win["dims/i"].as_int64();
                        int64_t nbr_size_j = nbr_win["dims/j"].as_int64();
                        int64_t nbr_size = nbr_size_i * nbr_size_j;

                        if (ref_size < nbr_size)
                        {

                            blueprint::mesh::connectivity::create_elements_2d(ref_win,
                                                                   i_lo,
                                                                   j_lo,
                                                                   iwidth,
                                                                   poly_elems);
                            int use_ratio = 0;
                            if (nbr_size_j == 1)
                            {
                                use_ratio = ratio_i;
                            }
                            else if (nbr_size_i == 1)
                            {
                                use_ratio = ratio_j;
                            }

                            auto& xbuffer = nbr_to_xbuffer[nbr_id];
                            auto& ybuffer = nbr_to_ybuffer[nbr_id];

                            size_t added = 0;
                            if (nbr_size_j == 1)
                            {
                                added = xbuffer.size() - ref_size_i;
                            } else if (nbr_size_i == 1)
                            {
                                added = ybuffer.size() - ref_size_j;
                            }

                            const auto& out_x = out_values["x"].as_double_array();
                            const auto& out_y = out_values["y"].as_double_array();
                            int64_t new_vertex = out_x.number_of_elements();

                            size_t out_x_size = out_x.number_of_elements();
                            size_t out_y_size = out_y.number_of_elements();

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
                            for (size_t ni = 0; ni < xbuffer.size(); ++ni)
                            {
                                if (ni % use_ratio)
                                {
                                    new_x.push_back(xbuffer[ni]);
                                    new_y.push_back(ybuffer[ni]);
                                }
                            }

                            out_values["x"].set(new_x);
                            out_values["y"].set(new_y);

                            blueprint::mesh::connectivity::connect_elements_2d(ref_win,
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

        std::string coords =
            chld["topologies"][name]["coordset"].as_string();
        dest[domain_name]["topologies"][name]["coordset"] = coords;

        Node& topo = dest[domain_name]["topologies"][name];

        topo["type"] = "unstructured";
        topo["elements/shape"] = "polygonal";

        int64_t elemsize = iwidth*jwidth;

        std::vector<int64_t> connect;
        std::vector<int64_t> num_vertices;
        std::vector<int64_t> elem_offsets;
        int64_t offset_sum = 0;
        for (int elem = 0; elem < elemsize; ++elem)
        {
            auto elem_itr = poly_elems.find(elem);
            if (elem_itr == poly_elems.end())
            {
                blueprint::mesh::connectivity::make_element_2d(connect, elem, iwidth);
                num_vertices.push_back(4);
            }
            else
            {
                std::vector<int64_t>& poly_elem = elem_itr->second;
                connect.insert(connect.end(), poly_elem.begin(), poly_elem.end());
                num_vertices.push_back(poly_elem.size());
            }
            elem_offsets.push_back(offset_sum);
            offset_sum += num_vertices.back();
        }

        topo["elements/connectivity"].set(connect);
        topo["elements/sizes"].set(num_vertices);
        topo["elements/offsets"].set(elem_offsets);

        //TODO:  zero copy
        if (chld.has_child("fields"))
        {
            dest[domain_name]["fields"] = chld["fields"];
        }
    }
}


//-----------------------------------------------------------------------------
void to_polyhedral(const Node &n,
                         Node &dest,
                         const std::string& name)
{

    dest.reset();

    NodeConstIterator itr = n.children();

    std::map<int, std::map<int, blueprint::mesh::connectivity::PolyElemType> > poly_elems_map;
    std::map<int, std::map<int, std::vector<int64_t> > > ifaces_map;
    std::map<int, std::map<int, std::vector<int64_t> > > jfaces_map;
    std::map<int, std::map<int, std::vector<int64_t> > > kfaces_map;

    while(itr.has_next())
    {
        const Node& chld = itr.next();
        std::string domain_name = itr.name();
        dest[domain_name]["state"] = chld["state"];
        const Node& in_coords = chld["coordsets/coords"];
        const Node& in_topo = chld["topologies"][name];

        int64_t iwidth = in_topo["elements/dims/i"].as_int64();
        int64_t jwidth = in_topo["elements/dims/j"].as_int64();
        int64_t kwidth = in_topo["elements/dims/k"].as_int64();

        Node& out_coords = dest[domain_name]["coordsets/coords"];

        int64_t domain_id = chld["state/domain_id"].as_int64();
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
        auto& ifaces = ifaces_map[domain_id];
        auto& jfaces = jfaces_map[domain_id];
        auto& kfaces = kfaces_map[domain_id];

        int64_t elemsize = iwidth*jwidth*kwidth;

        for (int elem = 0; elem < elemsize; ++elem)
        {
            blueprint::mesh::connectivity::make_element_3d(
                    poly_elems[elem], elem,
                    iwidth, jwidth, kwidth,
                    ifaces, jfaces, kfaces);
        }

        std::vector<int64_t> elem_connect;
        std::vector<int64_t> elem_sizes;
        std::vector<int64_t> elem_offsets;
        std::vector<int64_t> subelem_connect;
        std::vector<int64_t> subelem_sizes;
        std::vector<int64_t> subelem_offsets;
        int64_t elem_offset_sum = 0;
        int64_t subelem_offset_sum = 0;
        for (int elem = 0; elem < elemsize; ++elem)
        {
            auto& poly_elem = poly_elems[elem].m_elems;
            elem_connect.insert(elem_connect.end(), poly_elem.begin(), poly_elem.end());
            elem_sizes.push_back(poly_elem.size());
            elem_offsets.push_back(elem_offset_sum);
            elem_offset_sum += elem_sizes.back();
        }
        for (auto if_itr = ifaces.begin(); if_itr != ifaces.end(); ++if_itr)
        {
            auto& if_elem = if_itr->second;
            subelem_connect.insert(subelem_connect.end(), if_elem.begin(), if_elem.end());
            subelem_sizes.push_back(if_elem.size());
            subelem_offsets.push_back(subelem_offset_sum);
            subelem_offset_sum += subelem_sizes.back();
        }
        for (auto jf_itr = jfaces.begin(); jf_itr != jfaces.end(); ++jf_itr)
        {
            auto& jf_elem = jf_itr->second;
            subelem_connect.insert(subelem_connect.end(), jf_elem.begin(), jf_elem.end());
            subelem_sizes.push_back(jf_elem.size());
            subelem_offsets.push_back(subelem_offset_sum);
            subelem_offset_sum += subelem_sizes.back();
        }
        for (auto kf_itr = kfaces.begin(); kf_itr != kfaces.end(); ++kf_itr)
        {
            auto& kf_elem = kf_itr->second;
            subelem_connect.insert(subelem_connect.end(), kf_elem.begin(), kf_elem.end());
            subelem_sizes.push_back(kf_elem.size());
            subelem_offsets.push_back(subelem_offset_sum);
            subelem_offset_sum += subelem_sizes.back();
        }

        Node& topo = dest[domain_name]["topologies"][name];

        topo["coordset"] = in_topo["coordset"];

        topo["type"] = "unstructured";
        topo["elements/shape"] = "polyhedral";
        topo["elements/shape"].set_string("polyhedral");
        topo["elements/connectivity"].set(elem_connect);
        topo["elements/sizes"].set(elem_sizes);
        topo["elements/offsets"].set(elem_offsets);
        topo["subelements/shape"].set_string("polygonal");
        topo["subelements/connectivity"].set(subelem_connect);
        topo["subelements/sizes"].set(subelem_sizes);
        topo["subelements/offsets"].set(subelem_offsets);
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

