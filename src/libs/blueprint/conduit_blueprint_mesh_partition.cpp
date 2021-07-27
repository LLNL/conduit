// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_partition.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh_partition.hpp"

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <algorithm>
#include <deque>
#include <cmath>
#include <cstring>
#include <memory>
#include <set>
#include <vector>
#include <cstddef>
#include <type_traits>
#include <map>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_node.hpp"
#include "conduit_blueprint_mcarray.hpp"
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_blueprint_mesh.hpp"
#include "conduit_log.hpp"

//#ifdef CONDUIT_PARALLEL_PARTITION
//#include <mpi.h>
//#endif

// #define DEBUG_POINT_MERGE
#ifndef DEBUG_POINT_MERGE
#define PM_DEBUG_PRINT(stream)
#else
#define PM_DEBUG_PRINT(stream) do { std::cerr << stream; } while(0)
#endif

using index_t=conduit::index_t;
using std::cout;
using std::endl;

extern void grid_ijk_to_id(const index_t *ijk, const index_t *dims, index_t &grid_id);
extern void grid_id_to_ijk(const index_t id, const index_t *dims, index_t *grid_ijk);

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
// -- begin conduit::blueprint::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{

//-------------------------------------------------------------------------
/**
  options["comm"] = MPI_COMM_WORLD;

  "Information on how to re-decompose the grids will be provided or be easily calculatable."

  To go from 1-N, we need a way to pull the mesh apart. We would be able to
  provide the ways that this is done to select the bits we want so each selection
  makes a piece in the output mesh.

# Selections provide criteria used to split input meshes
options:
  selections:
    -
     type: "logical"
     domain: 0
     start: [0,0,0]
     end:   [10,10,10]
    -
     type: "logical"
     domain: 1
# We might have to specify a topo name too if there are multiple topos in the mesh.
# topology: "topo2"
     start: [50,50,50]
     end:   [60,60,60]
    -
     type: "explicit"
     domain: 0
     indices: [0,1,2,3,4,5,6,7,8,9]
    -
     type: "ranges"
     domain: 0
     ranges: [0,100, 1000,2000]
    -
     type: "explicit"
     domain: 1
     global_indices: [100,101,102,110,116,119,220]
    -
     type: "spatial"
     domain: 2
     box: [0., 0., 0., 11., 12., 22.]
    -
     type: "spatial"
     domain: 2
     box: [0., 0., 22., 11., 12., 44.]
  target: 7
  mapping: true
  replicate: false
  merging:
    enabled: true
    radius: 0.001
  fill_value: 0.0  # fill value to use for when not all chunks entirely cover space.
  root: 0
  mpi_comm: 11223
  fields: ["name1", "name2", "name3"]
      

  If selections re not present in the options then we assume that we're selecting
  all cells in all domains that we passed. This is fine for N-1. For 1-N if we 
  do not pass selections then in serial, we're passing back the input mesh. In parallel,
  we could be doing that, or gathering to root rank.


  // For structured grids, select logically...
  options["selections/logical/0/domain"] = 0;
  options["selections/logical/0/start"] = {0,0,0};
  options["selections/logical/0/end"] = {10,10,10};
  options["selections/logical/1/domain"] = 1;
  options["selections/logical/1/start"] = {50,50,50};
  options["selections/logical/1/end"] = {60,60,60};

  // For any grid type, we could select explicit

  options["selections/cells/0/domain"] = 0;
  options["selections/cells/0/indices"] = {0,1,2,3,4,5,6,7,8,9};

  options["selections/cells/0/ranges"] = {0,100, 1000,2000};

  options["selections/cells/1/global_indices"] = {100,101,102,110,116,119,220};

  // Pull the mesh apart using spatial boxes

  options["selections/spatial/0/domain"] = 0;
  options["selections/spatial/0/box"] = {0., 0., 0., 11., 12., 22.};

  options["target"] = 1;        // The target number of domains across all ranks
                                // participating in the partition. If we have 1
                                // rank, 4 selections, and target=2, make 2 domains
                                // out of the 4 selected chunks.

  options["mapping"] = true;    // If on, preserve original cells, conduit::Node ids 
                                // so it is known how the output mesh was created.

  options["merging/enabled"] = false;
  options["merging/radius"] = 0.0001; // Point merging radius


  options["replicate"] = false; // If we're moving N-1 then indicate whether we
                                // want the data to be replicated on all ranks

  options["root"] = 0;  // Indicate which is the root rank if we're gathering data N-1.
  options["mpi_comm"] = integer representing MPI comm.


  Suppose we're in parallel on 100 ranks and each rank has 1 domain. Then say that we are
  making 10 target domains. Which ranks get them? The first 10 ranks in the comm? Does this
  need to be an option? dest_ranks=[10,20,30,40,...]?
*/

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
const std::string selection::DOMAIN_KEY("domain");
const std::string selection::TOPOLOGY_KEY("topology");

//---------------------------------------------------------------------------
selection::selection() : whole(selection::WHOLE_UNDETERMINED), domain(0), topology()
{
}

//---------------------------------------------------------------------------
selection::~selection()
{
}

//---------------------------------------------------------------------------
index_t
selection::length() const
{
    return 0;
}

//---------------------------------------------------------------------------
bool
selection::get_whole(const conduit::Node &n_mesh)
{
    // If we have not yet determined whether the selection appears whole, 
    // do it now.
    if(whole == WHOLE_UNDETERMINED)
        set_whole(determine_is_whole(n_mesh));

    return whole == WHOLE_DETERMINED_TRUE;
}

//---------------------------------------------------------------------------
void
selection::set_whole(bool value)
{
    whole = value ? WHOLE_DETERMINED_TRUE : WHOLE_DETERMINED_FALSE;
}

//---------------------------------------------------------------------------
index_t
selection::get_domain() const
{
    return domain;
}

//---------------------------------------------------------------------------
void
selection::set_domain(index_t value)
{
    domain = value;
}

//---------------------------------------------------------------------------
const std::string &
selection::get_topology() const
{
    return topology;
}

//---------------------------------------------------------------------------
void
selection::set_topology(const std::string &value)
{
    topology = value;
}

//---------------------------------------------------------------------------
bool
selection::init(const conduit::Node &n_options)
{
    bool retval = true;

    try
    {
        if(n_options.has_child(DOMAIN_KEY))
            domain = n_options[DOMAIN_KEY].to_index_t();

        if(n_options.has_child(TOPOLOGY_KEY))
            topology = n_options[TOPOLOGY_KEY].as_string();
    }
    catch(...)
    {
        retval = false;
    }

    return retval;
}

//---------------------------------------------------------------------------
const conduit::Node &
selection::selected_topology(const conduit::Node &n_mesh) const
{
    if(n_mesh.has_child("topologies"))
    {
        const conduit::Node &n_topos = n_mesh["topologies"];
        if(topology.empty())
            return n_topos[0];
        else if(n_topos.has_child(topology))
            return n_topos[topology];
    }

    std::stringstream oss;
    oss << "The input mesh does not contain a topology with name " << topology;
    CONDUIT_ERROR(oss.str());
    throw conduit::Error(oss.str(), __FILE__, __LINE__);
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
/**
 @brief This class represents a logical IJK selection with start and end
        values. Start begins at 0 and End is the size of the mesh (in terms
        of cells) minus 1.

        A cell with 10x10x10 cells would have the following selection to
        select it all: start={0,0,0}, end={9,9,9}. To select a single cell,
        make start the same as end.
*/
class selection_logical : public selection
{
public:
    selection_logical();
    virtual ~selection_logical();

    static std::string name() { return "logical"; }

    // Initializes the selection from a conduit::Node.
    virtual bool init(const conduit::Node &n_options) override;

    virtual bool applicable(const conduit::Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length() const override
    {
        return cells_for_axis(0) * 
               cells_for_axis(1) * 
               cells_for_axis(2);
    }

    virtual std::vector<std::shared_ptr<selection> > partition(const conduit::Node &n_mesh) const override;

    void get_start(index_t &s0, index_t &s1, index_t &s2) const
    {
        s0 = start[0];
        s1 = start[1];
        s2 = start[2];
    }

    void set_start(index_t s0, index_t s1, index_t s2)
    {
        start[0] = s0;
        start[1] = s1;
        start[2] = s2;
    }

    void get_end(index_t &e0, index_t &e1, index_t &e2) const
    {
        e0 = end[0];
        e1 = end[1];
        e2 = end[2];
    }

    void set_end(index_t e0, index_t e1, index_t e2)
    {
        end[0] = e0;
        end[1] = e1;
        end[2] = e2;
    }

    virtual void get_element_ids(const conduit::Node &n_mesh,
                                 std::vector<index_t> &element_ids) const override;

    void get_vertex_ids(const conduit::Node &n_mesh,
                        std::vector<index_t> &element_ids) const;

    virtual void print(std::ostream &os) const override;

protected:
    virtual bool determine_is_whole(const conduit::Node &n_mesh) const override;

    index_t cells_for_axis(int axis) const
    {
        index_t nc = static_cast<index_t>(end[axis] - start[axis] + 1);
        return (axis >= 0 && axis <= 2) ? nc : 0;
    }

    static const std::string START_KEY;
    static const std::string END_KEY;

    index_t start[3];
    index_t end[3];
};

const std::string selection_logical::START_KEY("start");
const std::string selection_logical::END_KEY("end");

//---------------------------------------------------------------------------
selection_logical::selection_logical() : selection()
{
    start[0] = start[1] = start[2] = 0;
    end[0] = end[1] = end[2] = 0;
}

//---------------------------------------------------------------------------
selection_logical::~selection_logical()
{
}

//---------------------------------------------------------------------------
bool
selection_logical::init(const conduit::Node &n_options)
{
    bool retval = false;
    if(selection::init(n_options))
    {
        if(n_options.has_child(START_KEY) && n_options.has_child(END_KEY))
        {
            conduit::Node n_s, n_e;
            n_options[START_KEY].to_uint64_array(n_s);
            n_options[END_KEY].to_uint64_array(n_e);

            auto s = n_s.as_uint64_array();
            auto e = n_e.as_uint64_array();
            if(s.number_of_elements() == 3 &&  e.number_of_elements() == 3)
            {
                for(int i = 0; i < 3; i++)
                {
                    start[i] = static_cast<index_t>(s[i]);
                    end[i] = static_cast<index_t>(e[i]);
                }
                retval = true;
            }
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Returns whether the logical selection applies to the input mesh.
 */
bool
selection_logical::applicable(const conduit::Node &n_mesh)
{
    bool retval = false;
    try
    {
        // Get the selected topology and coordset.
        const conduit::Node &n_topo = selected_topology(n_mesh);
        std::string csname(n_topo["coordset"].as_string());
        const conduit::Node &n_coords = n_mesh["coordsets"][csname];

        bool is_uniform = n_coords["type"].as_string() == "uniform";
        bool is_rectilinear = n_coords["type"].as_string() == "rectilinear";
        bool is_structured = n_coords["type"].as_string() == "explicit" && 
                             n_topo["type"].as_string() == "structured";
        if(is_uniform || is_rectilinear || is_structured)
        {
            index_t dims[3] = {1,1,1};
            conduit::blueprint::mesh::utils::topology::logical_dims(n_topo, dims, 3);

            // See that the selection starts inside the dimensions.
            if(start[0] < dims[0] && start[1] < dims[1] && start[2] < dims[2])
            {
                // Clamp the selection to the dimensions of the mesh.
                end[0] = std::min(end[0], dims[0]-1);
                end[1] = std::min(end[1], dims[1]-1);
                end[2] = std::min(end[2], dims[2]-1);

                retval = true;
            }
        }
    }
    catch(conduit::Error &)
    {
        retval = false;
    }

    return retval;
}

//---------------------------------------------------------------------------
bool
selection_logical::determine_is_whole(const conduit::Node &n_mesh) const
{
    bool retval = false;
    try
    {
        // Get the selected topology and coordset.
        const conduit::Node &n_topo = selected_topology(n_mesh);
        index_t len = conduit::blueprint::mesh::utils::topology::length(n_topo);
        retval = len == length();
    }
    catch(conduit::Error &)
    {
        retval = false;
    }

    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Partitions along the longest axis and returns a vector containing 2
        logical selections.
 */
std::vector<std::shared_ptr<selection> >
selection_logical::partition(const conduit::Node &/*n_mesh*/) const
{
    std::vector<std::shared_ptr<selection> > parts;
    if(length() > 1)
    {
        int la = 0;
        if(cells_for_axis(1) > cells_for_axis(la))
            la = 1;
        if(cells_for_axis(2) > cells_for_axis(la))
            la = 2;
        auto n = cells_for_axis(la);

        auto p0 = std::make_shared<selection_logical>();
        auto p1 = std::make_shared<selection_logical>();
        p0->set_whole(false);
        p1->set_whole(false);
        p0->set_domain(domain);
        p1->set_domain(domain);
        p0->set_topology(topology);
        p1->set_topology(topology);

        if(la == 0)
        {
            p0->set_start(start[0],       start[1],       start[2]);
            p0->set_end(start[0]+n/2-1,   end[1],         end[2]);
            p1->set_start(start[0]+n/2,   start[1],       start[2]);
            p1->set_end(end[0],           end[1],         end[2]);
        }
        else if(la == 1)
        {
            p0->set_start(start[0],       start[1],       start[2]);
            p0->set_end(end[0],           start[1]+n/2-1, end[2]);
            p1->set_start(start[0],       start[1]+n/2,   start[2]);
            p1->set_end(end[0],           end[1],         end[2]);
        }
        else
        {
            p0->set_start(start[0],       start[1],       start[2]);
            p0->set_end(end[0],           end[1],         start[2]+n/2-1);
            p1->set_start(start[0],       start[1],       start[2]+n/2);
            p1->set_end(end[0],           end[1],         end[2]);
        }
#if 0
cout << "****\nselection_logical::partition: la=" << la << endl << "\t";
p0->print(cout);
cout << endl << "\t";
p1->print(cout);
cout << endl << "****" << endl;
#endif
        parts.push_back(p0);
        parts.push_back(p1);
    }
    return parts;
}

//---------------------------------------------------------------------------
void
selection_logical::get_vertex_ids(const conduit::Node &n_mesh,
    std::vector<index_t> &ids) const
{
    try
    {
        const conduit::Node &n_topo = selected_topology(n_mesh);
        index_t dims[3] = {1,1,1};
        conduit::blueprint::mesh::utils::topology::logical_dims(n_topo, dims, 3);
        index_t ndims = conduit::blueprint::mesh::utils::topology::dims(n_topo);
        dims[0]++;
        dims[1]++;
        if(ndims > 2)
            dims[2]++;

        ids.clear();
        ids.reserve(dims[0] * dims[1] * dims[2]);
        auto mesh_NXNY = dims[0] * dims[1];
        auto mesh_NX   = dims[0];
        index_t n_end[3];
        n_end[0] = end[0] + 1;
        n_end[1] = end[1] + 1;
        n_end[2] = (ndims > 2) ? (end[2] + 1) : start[2];

        for(index_t k = start[2]; k <= n_end[2]; k++)
        for(index_t j = start[1]; j <= n_end[1]; j++)
        for(index_t i = start[0]; i <= n_end[0]; i++)
        {
            index_t id = k*mesh_NXNY + j*mesh_NX + i;
            ids.push_back(id);
        }
    }
    catch(conduit::Error &)
    {
    }
}

//---------------------------------------------------------------------------
void
selection_logical::get_element_ids(const conduit::Node &n_mesh,
    std::vector<index_t> &element_ids) const
{
    try
    {
        const conduit::Node &n_topo = selected_topology(n_mesh);
        index_t dims[3] = {1,1,1};
        conduit::blueprint::mesh::utils::topology::logical_dims(n_topo, dims, 3);
cout << "selection_logical::get_element_ids: dims=" << dims[0] << ", " << dims[1] << ", " << dims[2] << endl;

        element_ids.clear();
        element_ids.reserve(length());
        auto mesh_CXCY = dims[0] * dims[1];
        auto mesh_CX   = dims[0];
        for(index_t k = start[2]; k <= end[2]; k++)
        for(index_t j = start[1]; j <= end[1]; j++)
        for(index_t i = start[0]; i <= end[0]; i++)
        {
            auto eid = k*mesh_CXCY + j*mesh_CX + i;
            element_ids.push_back(eid);
        }
#if 1
        cout << "selection_logical::get_element_ids={";
        for(size_t i = 0; i < element_ids.size(); i++)
            cout << element_ids[i] << ", ";
        cout << "}" << endl;
#endif
    }
    catch(conduit::Error &)
    {
    }
}

//---------------------------------------------------------------------------
void
selection_logical::print(std::ostream &os) const
{
    os << "{"
       << "\"name\":\"" << name() << "\","
       << "\"domain\":" << get_domain() << ", "
       << "\"topology\":\"" << get_topology() << "\", "
       << "\"start\":[" << start[0] << ", " << start[1] << ", " << start[2] << "],"
       << "\"end\":[" << end[0] << ", " << end[1] << ", " << end[2] << "]"
       << "}";
}

//---------------------------------------------------------------------------
/**
   @brief This selection explicitly defines which cells we're pulling out from
          a mesh, and in which order.
 */
class selection_explicit : public selection
{
public:
    selection_explicit();

    virtual ~selection_explicit();

    static std::string name() { return "explicit"; }

    virtual bool init(const conduit::Node &n_options) override;

    virtual bool applicable(const conduit::Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length() const override
    {
        return ids_storage.dtype().number_of_elements();
    }

    virtual std::vector<std::shared_ptr<selection> > partition(const conduit::Node &n_mesh) const override;

    const index_t *get_indices() const
    {
        // Access the converted data as index_t.
        return reinterpret_cast<const index_t *>(ids_storage.data_ptr());
    }

    void set_indices(const conduit::Node &value)
    {
        ids_storage.reset();
#ifdef CONDUIT_INDEX_32
        value.to_uint32_array(ids_storage);
#else
        value.to_uint64_array(ids_storage);
#endif
    }

    index_t num_indices() const { return ids_storage.dtype().number_of_elements(); }

    virtual void get_element_ids(const conduit::Node &n_mesh,
                                 std::vector<index_t> &element_ids) const override;

    virtual void print(std::ostream &os) const override;

protected:
    virtual bool determine_is_whole(const conduit::Node &n_mesh) const override;

    static const std::string ELEMENTS_KEY;
    conduit::Node ids_storage;
};

const std::string selection_explicit::ELEMENTS_KEY("elements");

//---------------------------------------------------------------------------
selection_explicit::selection_explicit() : selection(), ids_storage()
{
}

//---------------------------------------------------------------------------
selection_explicit::~selection_explicit()
{
}

//---------------------------------------------------------------------------
bool
selection_explicit::init(const conduit::Node &n_options)
{
    bool retval = false;
    if(selection::init(n_options))
    {
        if(n_options.has_child(ELEMENTS_KEY))
        {
            const conduit::Node &n_elem = n_options[ELEMENTS_KEY];
            if(n_elem.dtype().is_number())
            {
                // Convert to the right type for index_t
#ifdef CONDUIT_INDEX_32
                n_elem.to_uint32_array(ids_storage);
#else
                n_elem.to_uint64_array(ids_storage);
#endif
                retval = true;
            }
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Returns whether the explicit selection applies to the input mesh.
 */
bool
selection_explicit::applicable(const conduit::Node &/*n_mesh*/)
{
    return true;
}

//---------------------------------------------------------------------------
bool
selection_explicit::determine_is_whole(const conduit::Node &n_mesh) const
{
    bool whole = false;
    try
    {
        // Get the selected topology and coordset.
        const conduit::Node &n_topo = selected_topology(n_mesh);
        auto num_cells_in_mesh = topology::length(n_topo);
        auto n = num_indices();
        if(n == num_cells_in_mesh)
        {
            auto indices = get_indices();
            std::set<index_t> unique;
            for(index_t i = 0; i < n; i++)
                unique.insert(indices[i]);
            whole = static_cast<index_t>(unique.size()) == num_cells_in_mesh;
        }
    }
    catch(conduit::Error &)
    {
        whole = false;
    }

    return whole;
}

//---------------------------------------------------------------------------
std::vector<std::shared_ptr<selection> >
selection_explicit::partition(const conduit::Node &n_mesh) const
{
    // Get the number of elements in the topology.
    index_t num_elem_in_topo = 0;
    try
    {
        // Get the selected topology and coordset.
        const conduit::Node &n_topo = selected_topology(n_mesh);
        num_elem_in_topo = topology::length(n_topo);
    }
    catch(conduit::Error &)
    {
    }

    // Divide the elements into 2 vectors.
    auto n = num_indices();
    auto n_2 = n/2;
    auto indices = get_indices();
    std::vector<index_t> ids0, ids1;
    ids0.reserve(n_2);
    ids1.reserve(n_2);
    for(index_t i = 0; i < n; i++)
    {
        if(indices[i] < num_elem_in_topo)
        {
            if(i < n_2)
                ids0.push_back(indices[i]);
            else
                ids1.push_back(indices[i]);
        }
    }

    // Make partitioned selections.
    auto p0 = std::make_shared<selection_explicit>();
    auto p1 = std::make_shared<selection_explicit>();
    p0->ids_storage.set(ids0);
    p1->ids_storage.set(ids1);
    p0->set_whole(false);
    p1->set_whole(false);
    p0->set_domain(domain);
    p1->set_domain(domain);
    p0->set_topology(topology);
    p1->set_topology(topology);

    std::vector<std::shared_ptr<selection> > parts;
    parts.push_back(p0);
    parts.push_back(p1);

    return parts;
}

//---------------------------------------------------------------------------
void
selection_explicit::get_element_ids(const conduit::Node &n_mesh,
    std::vector<index_t> &element_ids) const
{
    try
    {
        const conduit::Node &n_topo = selected_topology(n_mesh);
        auto topolen = topology::length(n_topo);
        
        auto n = ids_storage.dtype().number_of_elements();
        auto indices = get_indices();
        element_ids.reserve(n);
        for(index_t i = 0; i < n; i++)
        {
            auto eid = indices[i];
            if(eid < topolen)
                element_ids.push_back(eid);
        }
    }
    catch(conduit::Error &)
    {
    }
}

//---------------------------------------------------------------------------
void
selection_explicit::print(std::ostream &os) const
{
    os << "{"
       << "\"name\":\"" << name() << "\","
       << "\"domain\":" << get_domain() << ", "
       << "\"topology\":\"" << get_topology() << "\", "
       << "\"indices\":[";
    auto n = length();
    auto indices = get_indices();
    for(index_t i = 0; i < n; i++)
    {
        if(i > 0)
            os << ", ";
        os << indices[i];
    }
    os << "]}";
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
class selection_ranges : public selection
{
public:
    selection_ranges();
    virtual ~selection_ranges();

    static std::string name() { return "ranges"; }

    virtual bool init(const conduit::Node &n_options) override;

    virtual bool applicable(const conduit::Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length() const override;

    virtual std::vector<std::shared_ptr<selection> > partition(const conduit::Node &n_mesh) const override;

    virtual void get_element_ids(const conduit::Node &n_mesh,
                                 std::vector<index_t> &element_ids) const override;

    void set_ranges(const conduit::Node &n)
    {
        ranges_storage.reset();
#ifdef CONDUIT_INDEX_32
        n.to_uint32_array(ranges_storage);
#else
        n.to_uint64_array(ranges_storage);
#endif
    }

    const index_t *get_ranges() const
    {
        // Access the converted data as index_t.
        return reinterpret_cast<const index_t *>(ranges_storage.data_ptr());
    }

    index_t num_ranges() const;

    virtual void print(std::ostream &os) const override;

protected:
    virtual bool determine_is_whole(const conduit::Node &n_mesh) const override;

    static const std::string RANGES_KEY;
    conduit::Node ranges_storage;
};

const std::string selection_ranges::RANGES_KEY("ranges");

//---------------------------------------------------------------------------
selection_ranges::selection_ranges() : selection(), ranges_storage()
{
}

//---------------------------------------------------------------------------
selection_ranges::~selection_ranges()
{
}

//---------------------------------------------------------------------------
bool
selection_ranges::init(const conduit::Node &n_options)
{
    bool retval = false;
    if(selection::init(n_options))
    {
        if(n_options.has_child(RANGES_KEY))
        {
            const conduit::Node &n_ranges = n_options[RANGES_KEY];
            if(n_ranges.dtype().is_number())
            {
                // Convert to the right type for index_t
#ifdef CONDUIT_INDEX_32
                n_ranges.to_uint32_array(ranges_storage);
#else
                n_ranges.to_uint64_array(ranges_storage);
#endif
                retval = (ranges_storage.dtype().number_of_elements() % 2 == 0);
            }
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
bool
selection_ranges::applicable(const conduit::Node &/*n_mesh*/)
{
    return true;
}

//---------------------------------------------------------------------------
index_t
selection_ranges::length() const
{
    index_t ncells = 0;
    const index_t *ranges = get_ranges();
    auto n = num_ranges();
    for(index_t i = 0; i < n; i++)
    {
        ncells += ranges[2*i+1] - ranges[2*i] + 1;
    }
    return ncells;
}

//---------------------------------------------------------------------------
index_t
selection_ranges::num_ranges() const
{
    return ranges_storage.dtype().number_of_elements() / 2;
}

//---------------------------------------------------------------------------
bool
selection_ranges::determine_is_whole(const conduit::Node &n_mesh) const
{
    bool whole = false;
    index_t num_elem_in_topo = 0;
    try
    {
        // Get the selected topology and coordset.
        const conduit::Node &n_topo = selected_topology(n_mesh);
        num_elem_in_topo = topology::length(n_topo);
    }
    catch(conduit::Error &)
    {
        num_elem_in_topo = 0;
    }

    auto n = num_ranges();
    if(n == 1)
    {
        auto upper_bound = std::min(get_ranges()[1], num_elem_in_topo-1);
        whole = get_ranges()[0] == 0 &&
                upper_bound == num_elem_in_topo-1;
    }
    else
    {
        auto indices = get_ranges();
        std::set<index_t> unique;
        for(index_t i = 0; i < n; i++)
        {
            index_t start = indices[2*i];
            index_t end = std::min(indices[2*i+1], num_elem_in_topo-1);
            for(index_t eid = start; eid <= end; eid++)
                unique.insert(eid);
        }
        whole = static_cast<index_t>(unique.size()) == num_elem_in_topo;
    }

    return whole;
}

//---------------------------------------------------------------------------
std::vector<std::shared_ptr<selection> >
selection_ranges::partition(const conduit::Node &/*n_mesh*/) const
{
    index_t ncells = length();
    auto ncells_2 = ncells / 2;
    auto n = num_ranges();
    auto ranges = get_ranges();
    index_t count = 0;
    index_t split_index = 0;
    for(index_t i = 0; i < n; i++)
    {
        auto rc = ranges[2*i+1] - ranges[2*i] + 1;
        if(count + rc > ncells_2)
        {
            split_index = i;
            break;
        }
        else
        {
            count += rc;
        }
    }

    std::vector<index_t> r0, r1;
    for(index_t i = 0; i < n; i++)
    {
        if(i < split_index)
        {
            r0.push_back(ranges[2*i+0]);
            r0.push_back(ranges[2*i+1]);
        }
        else if(i == split_index)
        {
            auto rc = (ranges[2*i+1] - ranges[2*i]) + 1;
            if(rc == 1)
            {
                r0.push_back(ranges[2*i+0]);
                r0.push_back(ranges[2*i+0]);
            }
            else if(rc == 2)
            {
                r0.push_back(ranges[2*i+0]);
                r0.push_back(ranges[2*i+0]);

                r1.push_back(ranges[2*i+1]);
                r1.push_back(ranges[2*i+1]);
            }
            else
            {
                auto rc_2 = rc / 2;
                r0.push_back(ranges[2*i+0]);
                r0.push_back(ranges[2*i+0] + rc_2);

                r1.push_back(ranges[2*i+0] + rc_2 + 1);
                r1.push_back(ranges[2*i+1]);
            }
        }
        else //if(i > split_index)
        {
            r1.push_back(ranges[2*i+0]);
            r1.push_back(ranges[2*i+1]);
        }
    }

    auto p0 = std::make_shared<selection_ranges>();
    auto p1 = std::make_shared<selection_ranges>();
    p0->ranges_storage.set(r0);
    p1->ranges_storage.set(r1);
    p0->set_whole(false);
    p1->set_whole(false);
    p0->set_domain(domain);
    p1->set_domain(domain);
    p0->set_topology(topology);
    p1->set_topology(topology);

    std::vector<std::shared_ptr<selection> > parts;
    parts.push_back(p0);
    parts.push_back(p1);

    return parts;
}

//---------------------------------------------------------------------------
void
selection_ranges::get_element_ids(const conduit::Node &n_mesh,
    std::vector<index_t> &element_ids) const
{
    try
    {
        const conduit::Node &n_topo = selected_topology(n_mesh);
        auto topolen = topology::length(n_topo);
        auto n = num_ranges();
        auto indices = get_ranges();
        for(index_t i = 0; i < n; i++)
        {
            index_t start = indices[2*i];
            index_t end = indices[2*i+1];
            for(index_t eid = start; eid <= end; eid++)
            {
                if(eid < topolen)
                    element_ids.push_back(eid);
            }
        }
    }
    catch(conduit::Error &)
    {
    }
}

//---------------------------------------------------------------------------
void
selection_ranges::print(std::ostream &os) const
{
    os << "{"
       << "\"name\":\"" << name() << "\","
       << "\"domain\":" << get_domain() << ", "
       << "\"topology\":\"" << get_topology() << "\", "
       << "\"ranges\":[";
    auto n = num_ranges() * 2;
    auto indices = get_ranges();
    for(index_t i = 0; i < n; i++)
    {
        if(i > 0)
            os << ", ";
        os << indices[i];
    }
    os << "]}";
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
partitioner::chunk::chunk() : mesh(nullptr), owns(false)
{
}

//---------------------------------------------------------------------------
partitioner::chunk::chunk(const Node *m, bool own) : mesh(m), owns(own)
{
}

//---------------------------------------------------------------------------
void
partitioner::chunk::free()
{
    if(owns)
    {
        Node *m = const_cast<Node *>(mesh);
        delete m;
        mesh = nullptr;
        owns = false;
    }
}

//---------------------------------------------------------------------------
partitioner::partitioner() : rank(0), size(1), target(1), meshes(), selections(),
    selected_fields(), mapping(true), merge_tolerance(1.e-8)
{
}

//---------------------------------------------------------------------------
partitioner::~partitioner()
{
}

//---------------------------------------------------------------------------
std::shared_ptr<selection>
partitioner::create_selection(const std::string &type) const
{
    std::shared_ptr<selection> retval;

    if(type == selection_logical::name())
        retval = std::make_shared<selection_logical>();
    else if(type == selection_explicit::name())
        retval = std::make_shared<selection_explicit>();
    else if(type == selection_ranges::name())
        retval = std::make_shared<selection_ranges>();
    else
    {
        CONDUIT_ERROR("Unknown selection type: " << type);
    }
    return retval;
}

//---------------------------------------------------------------------------
std::shared_ptr<selection>
partitioner::create_selection_all_elements(const conduit::Node &n_mesh) const
{
    std::shared_ptr<selection> retval;

    // We're making a selection that includes all elements for the mesh.
    // Take the first topology and its coordset.
    const conduit::Node &n_topo = n_mesh["topologies"][0];
    std::string csname(n_topo["coordset"].as_string());
    const conduit::Node &n_coords = n_mesh["coordsets"][csname];

    // Does the topo+coordset combo look structured?
    bool is_uniform = n_coords["type"].as_string() == "uniform";
    bool is_rectilinear = n_coords["type"].as_string() == "rectilinear";
    bool is_structured = n_coords["type"].as_string() == "explicit" && 
                         n_topo["type"].as_string() == "structured";

    if(is_uniform || is_rectilinear || is_structured)
    {
        index_t edims[3] = {1,1,1};
        conduit::blueprint::mesh::utils::topology::logical_dims(n_topo, edims, 3);
        retval = create_selection(selection_logical::name());
        retval->set_whole(true);
        retval->set_topology(n_topo.name());
        auto typed_sel = dynamic_cast<selection_logical *>(retval.get());
        if(typed_sel != nullptr)
        {
            typed_sel->set_end(edims[0] > 0 ? edims[0]-1 : 0,
                               edims[1] > 0 ? edims[1]-1 : 0,
                               edims[2] > 0 ? edims[2]-1 : 0);
        }
    }
    else
    {
        index_t nelem = topology::length(n_topo);

        // Create a range that selects the topology.
        retval = create_selection(selection_ranges::name());
        retval->set_whole(true);
        retval->set_topology(n_topo.name());
        auto typed_sel = dynamic_cast<selection_ranges *>(retval.get());
        if(typed_sel != nullptr)
        {
#ifdef CONDUIT_INDEX_32
            std::vector<uint32> range;
            range.push_back(0);
            range.push_back(static_cast<uint32>(nelem - 1));
            conduit::Node n_range;
            n_range.set(range);
            typed_sel->set_ranges(n_range);
#else
            std::vector<uint64> range;
            range.push_back(0);
            range.push_back(static_cast<uint64>(nelem - 1));
            conduit::Node n_range;
            n_range.set(range);
            typed_sel->set_ranges(n_range);
#endif
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
bool
partitioner::initialize(const conduit::Node &n_mesh, const conduit::Node &options)
{
    auto doms = conduit::blueprint::mesh::domains(n_mesh);

    // Iterate over the selections in the options and check them against the
    // domains that were passed in to make a vector of meshes and selections
    // that can be used to partition the meshes.
    if(options.has_child("selections"))
    {
        const conduit::Node &n_selections = options["selections"];
        for(index_t i = 0; i < n_selections.number_of_children(); i++)
        {
            const conduit::Node &n_sel = n_selections[i];
            try
            {
                // It has to have type to be a valid selection.
                if(!n_sel.has_child("type"))
                    continue;

                std::string type(n_sel["type"].as_string());
                auto sel = create_selection(type);
                if(sel != nullptr && sel->init(n_sel))
                {
                    // The selection is good. See if it applies to the domains.
                    auto n = static_cast<index_t>(doms.size());
                    for(index_t domid = 0; n; domid++)
                    {
                        // Q: What is the overall domain number for this domain?

                        if(domid == sel->get_domain() && sel->applicable(*doms[domid]))
                        {
                            meshes.push_back(doms[domid]);
                            selections.push_back(sel);
                            break;
                        }
                    }
                }
            }
            catch(const conduit::Error &)
            {
                return false;
            }
        }
    }
    else
    {
        // Add null selections to indicate that we take the whole domain.
        for(size_t domid = 0; domid < doms.size(); domid++)
        {
            auto sel = create_selection_all_elements(*doms[domid]);
            sel->set_domain(domid);
            selections.push_back(sel);
            meshes.push_back(doms[domid]);
        }
    }

    // Get the number of target partitions that we're making.
    if(options.has_child("target"))
        target = options["target"].to_unsigned_int();
    else
    {
        // target was not provided. Use the number of selections as the target.
        target = selections.size();
    }

    // Get any fields that we're using to limit the selection.
    if(options.has_child("fields"))
    {
        selected_fields.clear();
        const conduit::Node &n_fields = options["fields"];
        for(index_t i = 0; i < n_fields.number_of_children(); i++)
            selected_fields.push_back(n_fields[i].name());
    }

    // Get whether we want to preserve old numbering of vertices, elements.
    if(options.has_child("mapping"))
        mapping = options["mapping"].to_unsigned_int() != 0;

    // Get whether we want to preserve old numbering of vertices, elements.
    if(options.has_child("merge_tolerance"))
        merge_tolerance = options["merge_tolerance"].to_double();


#if 1
    cout << "partitioner::initialize" << endl;
    cout << "\ttarget=" << target << endl;
    for(size_t i = 0; i < selections.size(); i++)
    {
        cout << "\t";
        selections[i]->print(cout);
        cout << endl;
    }
#endif

    return !selections.empty();
}

//---------------------------------------------------------------------------
void
partitioner::get_largest_selection(int &sel_rank, int &sel_index) const
{
    sel_rank = 0;
    long largest_selection_size = 0;
    for(size_t i = 0; i < selections.size(); i++)
    {
        long ssize = static_cast<long>(selections[i]->length());
        if(ssize > largest_selection_size)
        {
            largest_selection_size = ssize;
            sel_index = static_cast<int>(i);
        }
    }
}

//---------------------------------------------------------------------------
long
partitioner::get_total_selections() const
{
    return static_cast<long>(selections.size());
}

//---------------------------------------------------------------------------
void
partitioner::split_selections()
{
    // Splitting.
    int iteration = 1;
    while(target > get_total_selections())
    {
        // Get the rank with the largest selection and get that local
        // selection index.
        int sel_rank = -1, sel_index = -1;
        get_largest_selection(sel_rank, sel_index);

        if(rank == sel_rank)
        {
            auto ps = selections[sel_index]->partition(*meshes[sel_index]);

            if(!ps.empty())
            {
                const conduit::Node *m = meshes[sel_index];
                meshes.insert(meshes.begin()+sel_index, ps.size()-1, m);
                selections.insert(selections.begin()+sel_index, ps.size()-1, nullptr);
                for(size_t i = 0; i < ps.size(); i++)
                    selections[sel_index + i] = ps[i];

#if 1
                cout << "partitioner::split_selections (after split "
                     << iteration << ")" << endl;
                for(size_t i = 0; i < selections.size(); i++)
                {
                    cout << "\t";
                    selections[i]->print(cout);
                    cout << endl;
                }
                iteration++;
#endif
            }
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::copy_fields(index_t domain, const std::string &topology,
    const std::vector<index_t> &vertex_ids,
    const std::vector<index_t> &element_ids,
    const conduit::Node &n_mesh,
    conduit::Node &n_output) const
{
    if(n_mesh.has_child("fields"))
    {
        const conduit::Node &n_fields = n_mesh["fields"];
        if(!vertex_ids.empty())
        {
            conduit::Node &n_output_fields = n_output["fields"];
            for(index_t i = 0; i < n_fields.number_of_children(); i++)
            {
                const conduit::Node &n_field = n_fields[i];
                if(n_field.has_child("association"))
                {
                    auto association = n_field["association"].as_string();
                    if(association == "vertex")
                    {
                        copy_field(n_field, vertex_ids, n_output_fields);
                    }
                }
            }

            if(mapping)
            {
                // Save the vertex_ids as a new MC field.
                conduit::Node &n_field = n_output_fields["original_vertex_ids"];
                n_field["association"] = "vertex";
                if(!topology.empty())
                    n_field["topology"] = topology;
                std::vector<index_t> domain_ids(vertex_ids.size(), domain);
                conduit::Node &n_values = n_field["values"];
                n_values["domains"].set(domain_ids);
                n_values["ids"].set(vertex_ids);
            }
        }

        if(!element_ids.empty())
        {
            conduit::Node &n_output_fields = n_output["fields"];
            for(index_t i = 0; i < n_fields.number_of_children(); i++)
            {
                const conduit::Node &n_field = n_fields[i];
                if(n_field.has_child("association"))
                {
                    auto association = n_field["association"].as_string();
                    if(association == "element")
                    {
                        copy_field(n_field, element_ids, n_output_fields);
                    }
                }
            }

            if(mapping)
            {
                // Save the element_ids as a new MC field.
                conduit::Node &n_field = n_output_fields["original_element_ids"];
                n_field["association"] = "element";
                if(!topology.empty())
                    n_field["topology"] = topology;
                std::vector<index_t> domain_ids(vertex_ids.size(), domain);
                conduit::Node &n_values = n_field["values"];
                n_values["domains"].set(domain_ids);
                n_values["ids"].set(element_ids);
            }
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::copy_field(const conduit::Node &n_field,
    const std::vector<index_t> &ids, Node &n_output_fields) const
{
    static const std::vector<std::string> keys{"association", "grid_function",
        "volume_dependent", "topology"};

    // If we're subselecting the list of fields, check whether the field we
    // want to copy is in selected_fields.
    if(!selected_fields.empty() &&
       std::find(selected_fields.begin(), selected_fields.end(), n_field.name()) == selected_fields.end())
    {
        return;
    }

// TODO: What about matsets and mixed fields?...
// https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#fields

    // Copy common field attributes from the old field into the new one.
    conduit::Node &n_new_field = n_output_fields[n_field.name()];
    for(const auto &key : keys)
    {
        if(n_field.has_child(key))
            n_new_field[key] = n_field[key];
    }

    const conduit::Node &n_values = n_field["values"];
    conduit::Node &new_values = n_new_field["values"];
    if(n_values.dtype().is_compact()) 
    {
        if(n_values.number_of_children() > 0)
        {

// The vel data must be interleaved. We need to use the DataArray element methods for access.


            // mcarray.
            for(index_t i = 0; i < n_values.number_of_children(); i++)
            {
                const conduit::Node &n_vals = n_values[i];
                slice_array(n_vals, ids, new_values[n_vals.name()]);
            }
        }
        else
            slice_array(n_values, ids, new_values);
    }
    else
    {
        // otherwise, we need to compact our data first
        conduit::Node n;
        n_values.compact_to(n);
        if(n.number_of_children() > 0)
        {
            // mcarray.
            conduit::Node &new_values = n_new_field["values"];
            for(index_t i = 0; i < n.number_of_children(); i++)
            {
                const conduit::Node &n_vals = n[i];
                slice_array(n_vals, ids, new_values[n_vals.name()]);
            }
        }
        else
            slice_array(n, ids, new_values);
    }
}

//---------------------------------------------------------------------------
// @brief Slice the n_src array using the indices stored in ids. We use the
//        array classes for their [] operators that deal with interleaved
//        and non-interleaved arrays.
template <typename T>
inline void
typed_slice_array(const T &src, const std::vector<index_t> &ids, T &dest)
{
    size_t n = ids.size();
    for(size_t i = 0; i < n; i++)
        dest[i] = src[ids[i]];
}

//---------------------------------------------------------------------------
// @note Should this be part of conduit::Node or DataArray somehow. The number
//       of times I've had to slice an array...
void
partitioner::slice_array(const conduit::Node &n_src_values,
    const std::vector<index_t> &ids, Node &n_dest_values) const
{
    // Copy the DataType of the input conduit::Node but override the number of elements
    // before copying it in so assigning to n_dest_values triggers a memory
    // allocation.
    auto dt = n_src_values.dtype();
    n_dest_values = DataType(n_src_values.dtype().id(), ids.size());

    // Do the slice.
    if(dt.is_int8())
    {
        auto dest(n_dest_values.as_int8_array());
        typed_slice_array(n_src_values.as_int8_array(), ids, dest);
    }
    else if(dt.is_int16())
    {
        auto dest(n_dest_values.as_int16_array());
        typed_slice_array(n_src_values.as_int16_array(), ids, dest);
    }
    else if(dt.is_int32())
    {
        auto dest(n_dest_values.as_int32_array());
        typed_slice_array(n_src_values.as_int32_array(), ids, dest);
    }
    else if(dt.is_int64())
    {
        auto dest(n_dest_values.as_int64_array());
        typed_slice_array(n_src_values.as_int64_array(), ids, dest);
    }
    else if(dt.is_uint8())
    {
        auto dest(n_dest_values.as_uint8_array());
        typed_slice_array(n_src_values.as_uint8_array(), ids, dest);
    }
    else if(dt.is_uint16())
    {
        auto dest(n_dest_values.as_uint16_array());
        typed_slice_array(n_src_values.as_uint16_array(), ids, dest);
    }
    else if(dt.is_uint32())
    {
        auto dest(n_dest_values.as_uint32_array());
        typed_slice_array(n_src_values.as_uint32_array(), ids, dest);
    }
    else if(dt.is_uint64())
    {
        auto dest(n_dest_values.as_uint64_array());
        typed_slice_array(n_src_values.as_uint64_array(), ids, dest);
    }
    else if(dt.is_char())
    {
        auto dest(n_dest_values.as_char_array());
        typed_slice_array(n_src_values.as_char_array(), ids, dest);
    }
    else if(dt.is_short())
    {
        auto dest(n_dest_values.as_short_array());
        typed_slice_array(n_src_values.as_short_array(), ids, dest);
    }
    else if(dt.is_int())
    {
        auto dest(n_dest_values.as_int_array());
        typed_slice_array(n_src_values.as_int_array(), ids, dest);
    }
    else if(dt.is_long())
    {
        auto dest(n_dest_values.as_long_array());
        typed_slice_array(n_src_values.as_long_array(), ids, dest);
    }
    else if(dt.is_unsigned_char())
    {
        auto dest(n_dest_values.as_unsigned_char_array());
        typed_slice_array(n_src_values.as_unsigned_char_array(), ids, dest);
    }
    else if(dt.is_unsigned_short())
    {
        auto dest(n_dest_values.as_unsigned_short_array());
        typed_slice_array(n_src_values.as_unsigned_short_array(), ids, dest);
    }
    else if(dt.is_unsigned_int())
    {
        auto dest(n_dest_values.as_unsigned_int_array());
        typed_slice_array(n_src_values.as_unsigned_int_array(), ids, dest);
    }
    else if(dt.is_unsigned_long())
    {
        auto dest(n_dest_values.as_unsigned_long_array());
        typed_slice_array(n_src_values.as_unsigned_long_array(), ids, dest);
    }
    else if(dt.is_float())
    {
        auto dest(n_dest_values.as_float_array());
        typed_slice_array(n_src_values.as_float_array(), ids, dest);
    }
    else if(dt.is_double())
    {
        auto dest(n_dest_values.as_double_array());
        typed_slice_array(n_src_values.as_double_array(), ids, dest);
    }
}

//---------------------------------------------------------------------------
/**
 @brief Iterates over the cells in the topo that are specified in element_ids
        and adds their vertex ids into vertex_ids so we can build up a set of
        vertices that will need to be pulled from the coordset.
 */
void
partitioner::get_vertex_ids_for_element_ids(const conduit::Node &n_topo,
    const std::vector<index_t> &element_ids,
    std::set<index_t> &vertex_ids) const
{
    bool is_base_rectilinear = n_topo["type"].as_string() == "rectilinear";
    bool is_base_structured = n_topo["type"].as_string() == "structured";
    bool is_base_uniform = n_topo["type"].as_string() == "uniform";

    if(is_base_rectilinear || is_base_structured || is_base_uniform)
    {
        index_t edims[3] = {1,1,1}, dims[3] = {0,0,0};
        auto ndims = topology::dims(n_topo);
        conduit::blueprint::mesh::utils::topology::logical_dims(n_topo, edims, 3);
//cout << "get_vertex_ids_for_element_ids: edims=" << edims[0] << ", " << edims[1] << ", " << edims[2] << endl;
        dims[0] = edims[0] + 1;
        dims[1] = edims[1] + (ndims > 1 ? 1 : 0);
        dims[2] = edims[2] + (ndims > 2 ? 1 : 0);
//cout << "get_vertex_ids_for_element_ids: dims=" << dims[0] << ", " << dims[1] << ", " << dims[2] << endl;

        index_t cell_ijk[3]={0,0,0}, pt_ijk[3] = {0,0,0}, ptid = 0;
        static const index_t offsets[8][3] = {
            {0,0,0},
            {1,0,0},
            {0,1,0},
            {1,1,0},
            {0,0,1},
            {1,0,1},
            {0,1,1},
            {1,1,1}
        };
        int np = (ndims == 2) ? 4 : 8;
        auto n = element_ids.size();
        for(size_t i = 0; i < n; i++)
        {
            // Get the IJK coordinate of the element.
            grid_id_to_ijk(element_ids[i], edims, cell_ijk);
//cout << element_ids[i] << "-> " << cell_ijk[0] << "," << cell_ijk[1] << "," << cell_ijk[2] << endl;
            // Turn the IJK into vertex ids.
            for(int i = 0; i < np; i++)
            {
                pt_ijk[0] = cell_ijk[0] + offsets[i][0];
                pt_ijk[1] = cell_ijk[1] + offsets[i][1];
                pt_ijk[2] = cell_ijk[2] + offsets[i][2];
                grid_ijk_to_id(pt_ijk, dims, ptid);

//cout << "\t" << pt_ijk[0] << "," << pt_ijk[1] << "," << pt_ijk[2] << "-> " << ptid << endl;

                vertex_ids.insert(ptid);
            }
        }
    }
    else
    {
        const conduit::Node &n_conn = n_topo["elements/connectivity"];
// Conduit needs to_index_t_array() and as_index_t_ptr() methods.
        conduit::Node indices;
#ifdef CONDUIT_INDEX_32
        n_conn.to_unsigned_int_array(indices);
        auto iptr = indices.as_unsigned_int_ptr();
#else
        n_conn.to_unsigned_long_array(indices);
        auto iptr = indices.as_unsigned_long_ptr();
#endif
        conduit::blueprint::mesh::utils::ShapeType shape(n_topo);
        if(shape.is_poly())
        {
            // TODO:
        }
        else if(shape.is_polygonal())
        {
            // TODO:
        }
        else if(shape.is_polyhedral())
        {
            // TODO:
        }
        else
        {
            // Shapes are single types one after the next in the connectivity.
            auto nverts_in_shape = conduit::blueprint::mesh::utils::TOPO_SHAPE_INDEX_COUNTS[shape.id];
            for(size_t i = 0; i < element_ids.size(); i++)
            {
                auto elem_conn = iptr + element_ids[i] * nverts_in_shape;
                for(index_t j = 0; j < nverts_in_shape; j++)
                    vertex_ids.insert(elem_conn[j]);
            }
        }
    }
}

//---------------------------------------------------------------------------
inline void
index_t_set_to_vector(const std::set<index_t> &src, std::vector<index_t> &dest)
{
    dest.reserve(src.size());
    for(auto it = src.begin(); it != src.end(); it++)
        dest.push_back(*it);
}

//---------------------------------------------------------------------------
conduit::Node *
partitioner::extract(size_t idx, const conduit::Node &n_mesh) const
{
    if(idx >= selections.size())
        return nullptr;

    conduit::Node *retval = nullptr;
    try
    {
        // Get the appropriate topology and coordset nodes.
        const conduit::Node &n_topo = selections[idx]->selected_topology(n_mesh);
        const conduit::Node &n_coordsets = n_mesh["coordsets"];
        std::string csname(n_topo["coordset"].as_string());
        const conduit::Node &n_coordset = n_coordsets[csname];

        // Make output.
        retval = new conduit::Node;
        conduit::Node &n_output = *retval;
        conduit::Node &n_new_coordsets = n_output["coordsets"];
        conduit::Node &n_new_topos = n_output["topologies"];

        // Copy state.
        if(n_mesh.has_child("state"))
        {
            const conduit::Node &n_state = n_mesh["state"];
            n_output["state"].set(n_state);
        }

        // Get the indices of the selected elements.
        std::vector<index_t> element_ids, vertex_ids;
        selections[idx]->get_element_ids(n_mesh, element_ids);

        // Try special case for logical selections. We do this so we can make
        // logically structured outputs.
        auto log_sel = dynamic_cast<selection_logical *>(selections[idx].get());
        if(log_sel != nullptr)
        {
            index_t start[3], end[3];
            log_sel->get_start(start[0], start[1], start[2]);
            log_sel->get_end(end[0], end[1], end[2]);

            // Get the vertex ids of the selected elements. We can do it like
            // this faster than using a set. (needed for field extraction)
            log_sel->get_vertex_ids(n_mesh, vertex_ids);

            if(n_coordset["type"].as_string() == "uniform")
                create_new_uniform_coordset(n_coordset, start, end, n_new_coordsets[csname]);
            else if(n_coordset["type"].as_string() == "rectilinear")
                create_new_rectilinear_coordset(n_coordset, start, end, n_new_coordsets[csname]);
            else
                create_new_explicit_coordset(n_coordset, vertex_ids, n_new_coordsets[csname]);

            // Now, create new topologies.
            if(n_topo["type"].as_string() == "uniform")
                create_new_uniform_topo(n_topo, csname, start, n_new_topos[n_topo.name()]);
            else if(n_topo["type"].as_string() == "rectilinear")
                create_new_rectilinear_topo(n_topo, csname, start, n_new_topos[n_topo.name()]);
            else if(n_topo["type"].as_string() == "structured")
                create_new_structured_topo(n_topo, csname, start, end, n_new_topos[n_topo.name()]);
        }
        else
        {
            // Get the unique set of vertex ids used by the elements.
            std::set<index_t> vertex_ids_set;
            get_vertex_ids_for_element_ids(n_topo, element_ids, vertex_ids_set);
            index_t_set_to_vector(vertex_ids_set, vertex_ids);

#if 0
cout << "vertex_ids=";
for(size_t i = 0; i < vertex_ids.size(); i++)
    cout << vertex_ids[i] << ", ";
cout << endl;
#endif

            // Create a new coordset consisting of the selected vertex ids.
            create_new_explicit_coordset(n_coordset, vertex_ids, n_new_coordsets[csname]);

            // Create a new topology consisting of the selected element ids.
            create_new_unstructured_topo(n_topo, csname, 
                element_ids, vertex_ids, n_new_topos[n_topo.name()]);
        }

        // Create new fields.
        copy_fields(selections[idx]->get_domain(), n_topo.name(),
            vertex_ids, element_ids, n_mesh, n_output);
    }
    catch(conduit::Error &)
    {
        delete retval;
        retval = nullptr;
    }

    return retval;
}

//---------------------------------------------------------------------------
void
partitioner::create_new_uniform_coordset(const conduit::Node &n_coordset,
    const index_t start[3], const index_t end[3], conduit::Node &n_new_coordset) const
{
    // Set coordset dimensions from element start/end.
    index_t ndims = conduit::blueprint::mesh::utils::coordset::dims(n_coordset);
    n_new_coordset["type"] = "uniform";
    n_new_coordset["dims/i"] = end[0] - start[0] + 2;
    n_new_coordset["dims/j"] = end[1] - start[1] + 2;
    if(ndims > 2)
        n_new_coordset["dims/k"] = end[2] - start[2] + 2;

    // Adjust the origins to the right start values.
    auto axes(conduit::blueprint::mesh::utils::coordset::axes(n_coordset));
    const conduit::Node &n_origin = n_coordset["origin"];
    const conduit::Node &n_spacing = n_coordset["spacing"];
    conduit::Node &n_new_origin = n_new_coordset["origin"];
    for(index_t i = 0; i < ndims; i++)
    {
        conduit::Node &origin_i = n_new_origin[n_origin[i].name()];
        double org = n_origin[i].to_double() + n_spacing[i].to_double() * start[i];
        origin_i.set(org);
    }
    // Copy all the spacings
    n_new_coordset["spacing"].set(n_coordset["spacing"]);
}

//---------------------------------------------------------------------------
void
partitioner::create_new_rectilinear_coordset(const conduit::Node &n_coordset,
    const index_t start[3], const index_t end[3], conduit::Node &n_new_coordset) const
{
    const conduit::Node &n_values = n_coordset["values"];
    conduit::Node &n_new_values = n_new_coordset["values"];
    n_new_coordset["type"] = "rectilinear";
    // Slice each axis logically.
    index_t ndims = conduit::blueprint::mesh::utils::coordset::dims(n_coordset);
    for(index_t d = 0; d < ndims; d++)
    {
        std::vector<index_t> indices;
        auto nend = end[d] + 1;
        for(index_t i = start[d]; i <= nend; i++)
            indices.push_back(i);

        const conduit::Node &src = n_values[d];
        slice_array(src, indices, n_new_values[src.name()]);
    }
}

//---------------------------------------------------------------------------
void
partitioner::create_new_explicit_coordset(const conduit::Node &n_coordset,
    const std::vector<index_t> &vertex_ids, conduit::Node &n_new_coordset) const
{
    conduit::Node n_explicit;
    n_new_coordset["type"] = "explicit";
    if(n_coordset["type"].as_string() == "uniform")
    {
        conduit::blueprint::mesh::coordset::uniform::to_explicit(n_coordset, n_explicit);

        auto axes = conduit::blueprint::mesh::utils::coordset::axes(n_explicit);
        const conduit::Node &n_values = n_explicit["values"];
        conduit::Node &n_new_values = n_new_coordset["values"];
        for(size_t i = 0; i < axes.size(); i++)
        {
            const conduit::Node &n_axis_values = n_values[axes[i]];
            conduit::Node &n_new_axis_values = n_new_values[axes[i]];
            slice_array(n_axis_values, vertex_ids, n_new_axis_values);
        }
    }
    else if(n_coordset["type"].as_string() == "rectilinear")
    {
        conduit::blueprint::mesh::coordset::rectilinear::to_explicit(n_coordset, n_explicit);

        auto axes = conduit::blueprint::mesh::utils::coordset::axes(n_explicit);
        const conduit::Node &n_values = n_explicit["values"];
        conduit::Node &n_new_values = n_new_coordset["values"];
        for(size_t i = 0; i < axes.size(); i++)
        {
            const conduit::Node &n_axis_values = n_values[axes[i]];
            conduit::Node &n_new_axis_values = n_new_values[axes[i]];
            slice_array(n_axis_values, vertex_ids, n_new_axis_values);
        }
    }
    else if(n_coordset["type"].as_string() == "explicit")
    {
        auto axes = conduit::blueprint::mesh::utils::coordset::axes(n_coordset);
        const conduit::Node &n_values = n_coordset["values"];
        conduit::Node &n_new_values = n_new_coordset["values"];
        for(size_t i = 0; i < axes.size(); i++)
        {
            const conduit::Node &n_axis_values = n_values[axes[i]];
            conduit::Node &n_new_axis_values = n_new_values[axes[i]];
            slice_array(n_axis_values, vertex_ids, n_new_axis_values);
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::create_new_uniform_topo(const conduit::Node &n_topo,
    const std::string &csname, const index_t start[3],
    conduit::Node &n_new_topo) const
{
    n_new_topo["type"] = "uniform";
    n_new_topo["coordset"] = csname;
    const char *keys[] = {"elements/origin/i",
        "elements/origin/j",
        "elements/origin/k"};
    for(int i = 0; i < 3; i++)
    {
        if(n_topo.has_path(keys[i]))
        {
            const conduit::Node &value = n_topo[keys[i]];
            n_new_topo[keys[i]].set(static_cast<conduit::int64>(value.to_uint64() + start[i]));
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::create_new_rectilinear_topo(const conduit::Node &n_topo,
    const std::string &csname, const index_t start[3],
    conduit::Node &n_new_topo) const
{
    n_new_topo["type"] = "rectilinear";
    n_new_topo["coordset"] = csname;
    const char *keys[] = {"elements/origin/i",
        "elements/origin/j",
        "elements/origin/k"};
    for(int i = 0; i < 3; i++)
    {
        if(n_topo.has_path(keys[i]))
        {
            const conduit::Node &value = n_topo[keys[i]];
            n_new_topo[keys[i]].set(static_cast<conduit::int64>(value.to_uint64() + start[i]));
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::create_new_structured_topo(const conduit::Node &n_topo,
    const std::string &csname, const index_t start[3], const index_t end[3],
    conduit::Node &n_new_topo) const
{
    n_new_topo["type"] = "structured";
    n_new_topo["coordset"] = csname;
    conduit::Node &n_dims = n_new_topo["elements/dims"];
    n_dims["i"].set(static_cast<conduit::int64>(end[0] - start[0] + 1));
    n_dims["j"].set(static_cast<conduit::int64>(end[1] - start[1] + 1));
    if(n_topo.has_path("elements/dims/k"))
        n_dims["k"].set(static_cast<conduit::int64>(end[2] - start[2] + 1));
    const char *keys[] = {"elements/origin/i0",
        "elements/origin/j0",
        "elements/origin/k0"};
    for(int i = 0; i < 3; i++)
    {
        if(n_topo.has_child(keys[i]))
        {
            const conduit::Node &value = n_topo[keys[i]];
            n_new_topo[keys[i]].set(static_cast<conduit::int64>(value.to_uint64() + start[i]));
        }
    }
}

//---------------------------------------------------------------------------
/**
 @note We pass in the coordset name because the conversion routines in here
       result in topologies that do not set a coordset name.
 */
void
partitioner::create_new_unstructured_topo(const conduit::Node &n_topo,
    const std::string &csname,
    const std::vector<index_t> &element_ids,
    const std::vector<index_t> &vertex_ids,
    conduit::Node &n_new_topo) const
{
    if(n_topo["type"].as_string() == "uniform")
    {
        conduit::Node n_uns, cdest; // what is cdest?
        conduit::blueprint::mesh::topology::uniform::to_unstructured(n_topo, n_uns, cdest);
        unstructured_topo_from_unstructured(n_uns, csname, element_ids, vertex_ids, n_new_topo);
    }
    else if(n_topo["type"].as_string() == "rectilinear")
    {
        conduit::Node n_uns, cdest; // what is cdest?
        conduit::blueprint::mesh::topology::rectilinear::to_unstructured(n_topo, n_uns, cdest);
        unstructured_topo_from_unstructured(n_uns, csname, element_ids, vertex_ids, n_new_topo);
    }
    else if(n_topo["type"].as_string() == "structured")
    {
        conduit::Node n_uns, cdest; // what is cdest?
        conduit::blueprint::mesh::topology::structured::to_unstructured(n_topo, n_uns, cdest);
        unstructured_topo_from_unstructured(n_uns, csname, element_ids, vertex_ids, n_new_topo);
    }
    else if(n_topo["type"].as_string() == "unstructured")
    {
        unstructured_topo_from_unstructured(n_topo, csname, element_ids, vertex_ids, n_new_topo);
    }
}

//---------------------------------------------------------------------------
void
partitioner::unstructured_topo_from_unstructured(const conduit::Node &n_topo,
    const std::string &csname,
    const std::vector<index_t> &element_ids,
    const std::vector<index_t> &vertex_ids,
    conduit::Node &n_new_topo) const
{
    n_new_topo["type"].set("unstructured");
    n_new_topo["coordset"].set(csname);

    // vertex_ids contains the list of old vertex ids that our selection uses
    // from the old coordset. It can serve as a new to old map.

    std::map<index_t,index_t> old2new;
//cout << "old2new=" << endl;
    for(size_t i = 0; i < vertex_ids.size(); i++)
{
//cout << "  " << vertex_ids[i] << "-> " << i << endl;
        old2new[vertex_ids[i]] = static_cast<index_t>(i);
}
    const conduit::Node &n_conn = n_topo["elements/connectivity"];
// Conduit needs to_index_t_array() and as_index_t_ptr() methods.
    conduit::Node indices;
#ifdef CONDUIT_INDEX_32
    n_conn.to_unsigned_int_array(indices);
    auto iptr = indices.as_unsigned_int_ptr();
#else
    n_conn.to_unsigned_long_array(indices);
    auto iptr = indices.as_unsigned_long_ptr();
#endif
    conduit::blueprint::mesh::utils::ShapeType shape(n_topo);
    std::vector<index_t> new_conn;
    if(shape.is_poly())
    {
        // TODO:
    }
    else if(shape.is_polygonal())
    {
        // TODO:
    }
    else if(shape.is_polyhedral())
    {
        // TODO:
    }
    else
    {
        // Shapes are single types one after the next in the connectivity.
        auto nverts_in_shape = conduit::blueprint::mesh::utils::TOPO_SHAPE_INDEX_COUNTS[shape.id];
        for(size_t i = 0; i < element_ids.size(); i++)
        {
            auto elem_conn = iptr + element_ids[i] * nverts_in_shape;
#if 0
            cout << "cell " << element_ids[i] << ":  old(";
            for(index_t j = 0; j < nverts_in_shape; j++)
                cout << elem_conn[j] << ", ";
            cout << "), new(";
            for(index_t j = 0; j < nverts_in_shape; j++)
            {
                auto it = old2new.find(elem_conn[j]);
                if(it == old2new.end())
                    cout << "ERROR" << ", ";
                else
                    cout << it->second << ", ";
            }
            cout << ")" << endl;
#endif
            for(index_t j = 0; j < nverts_in_shape; j++)
                new_conn.push_back(old2new[elem_conn[j]]);
        }
    }

    n_new_topo["elements/shape"].set(n_topo["elements/shape"]);
    // TODO: Is there a better way to get the data into the node?
    n_new_topo["elements/connectivity"].set(new_conn);
}

//---------------------------------------------------------------------------
void
partitioner::execute(conduit::Node &output)
{
    // By this stage, we will have at least target selections spread across
    // the participating ranks. Now, we need to process the selections to
    // make chunks.
    std::vector<chunk> chunks;
    for(size_t i = 0; i < selections.size(); i++)
    {
        if(selections[i]->get_whole(*meshes[i]))
        {
// FIXME: This is not what I want to do because it bypasses the splitting process.

            // We had a "null" selection so we'll take the whole mesh.
            chunks.push_back(chunk(meshes[i], false));
        }
        else
        {
            conduit::Node *c = extract(i, *meshes[i]);
            chunks.push_back(chunk(c, true));
        }
    }

    // We need to figure out ownership and make sure each rank has the parts
    // that it needs to achieve "target" overall domains over all ranks. We
    // probably want to send/recv the data as binary blobs. Then deserialize
    // and assemble into combined grids.

    // Compute the destination rank and destination domain of each input
    // chunk present on this rank.
    std::vector<int> dest_rank, dest_domain;
    map_chunks(chunks, dest_rank, dest_domain);
#if 0
    cout << "dest_rank = {" << endl;
    for(size_t i = 0; i < dest_rank.size(); i++)
        cout << dest_rank[i] << ", ";
    cout << "}" << endl;
    cout << "dest_domain = {" << endl;
    for(size_t i = 0; i < dest_domain.size(); i++)
        cout << dest_domain[i] << ", ";
    cout << "}" << endl;
#endif

    // Communicate chunks to the right destination ranks
    std::vector<chunk> chunks_to_assemble;
    std::vector<int> chunks_to_assemble_domains;
    communicate_chunks(chunks, dest_rank, dest_domain,
        chunks_to_assemble, chunks_to_assemble_domains);

    // Now that we have all the parts we need in chunks_to_assemble, combine
    // the chunks.
    std::set<int> unique_doms;
    for(size_t i = 0; i < chunks_to_assemble_domains.size(); i++)
        unique_doms.insert(chunks_to_assemble_domains[i]);

    if(!chunks_to_assemble.empty())
    {
        output.reset();
        for(auto dom = unique_doms.begin(); dom != unique_doms.end(); dom++)
        {
            // Get the chunks for this output domain.
            std::vector<const Node *> this_dom_chunks;
            for(size_t i = 0; i < chunks_to_assemble_domains.size(); i++)
            {
                if(chunks_to_assemble_domains[i] == *dom)
                    this_dom_chunks.push_back(chunks_to_assemble[i].mesh);
            }

            if(this_dom_chunks.size() == 1)
            {
                if(unique_doms.size() > 1)
                {
                    // There are multiple domains in the output.
                    conduit::Node &n = output.append();
                    n.set(*this_dom_chunks[0]); // Could we transfer ownership if we own the chunk?
                }
                else
                {
                    // There is one domain in the output.
                    output.set(*this_dom_chunks[0]); // Could we transfer ownership if we own the chunk?
                }
            }
            else if(this_dom_chunks.size() > 1)
            {
                // Combine the chunks for this domain and add to output or to
                // a list in output.
                if(unique_doms.size() > 1)
                {
                    // There are multiple domains in the output.
                    combine(*dom, this_dom_chunks, output.append());
                }
                else
                {
                    // There is one domain in the output.
                    combine(*dom, this_dom_chunks, output);
                }
            }
        }
    }

    // Clean up
    for(size_t i = 0; i < chunks.size(); i++)
        chunks[i].free();
    for(size_t i = 0; i < chunks_to_assemble.size(); i++)
        chunks_to_assemble[i].free();
}

//-------------------------------------------------------------------------
unsigned int
partitioner::starting_index(const std::vector<partitioner::chunk> &/*chunks*/)
{
    return 0;
}

//-------------------------------------------------------------------------
void
partitioner::map_chunks(const std::vector<partitioner::chunk> &chunks,
    std::vector<int> &dest_ranks,
    std::vector<int> &dest_domain)
{
    // All data stays on this rank in serial.
    dest_ranks.resize(chunks.size());
    for(size_t i = 0; i < chunks.size(); i++)
        dest_ranks[i] = rank;
#if 0
    cout << "map_chunks:" << endl;
    for(size_t i = 0; i < chunks.size(); i++)
        chunks[i].mesh->print();
#endif

    // Determine average chunk size.
    std::vector<index_t> chunk_sizes;
    index_t total_len = 0;
    for(size_t i =0 ; i < chunks.size(); i++)
    {
        const conduit::Node &n_topos = chunks[i].mesh->operator[]("topologies");
        index_t len = 0;
        for(index_t j = 0; j < n_topos.number_of_children(); j++)
            len += conduit::blueprint::mesh::topology::length(n_topos[j]);
        total_len += len;
        chunk_sizes.push_back(len);
    }
    index_t len_per_target = total_len / target;
#if 0
    cout << "map_chunks: total_len = " << total_len
         << ", target=" << target
         << ", len_per_target=" << len_per_target << endl;
#endif
    int start_index = starting_index(chunks);
    if(chunks.size() == static_cast<size_t>(target))
    {
        // The number of chunks is the same as the target.
        for(size_t i =0 ; i < chunks.size(); i++)
            dest_domain.push_back(start_index + static_cast<int>(i));
    }
    else if(chunks.size() > static_cast<size_t>(target))
    {
        // NOTE: We are just grouping adjacent chunks in the overall list
        //       while trying to target a certain number of cells per domain.
        //       We may also want to consider the bounding boxes so we
        //       group chunks that are close spatially.
        unsigned int domid = start_index;
        index_t total_len = 0;
        for(size_t i = 0; i < chunks.size(); i++)
        {
            total_len += chunk_sizes[i];
            if(total_len >= len_per_target && domid < target)
            {
                // Advance to the next domain index.
                total_len = 0;
                domid++;
            }

            dest_domain.push_back(domid);
        }
    }
    else
    {
        // The number of chunks is less than the target. Something is wrong!
        CONDUIT_ERROR("The number of chunks (" << chunks.size()
                      << ") is smaller than requested (" << target << ").");
    }
}

//-------------------------------------------------------------------------
void
partitioner::communicate_chunks(const std::vector<partitioner::chunk> &chunks,
    const std::vector<int> &/*dest_rank*/,
    const std::vector<int> &dest_domain,
    std::vector<partitioner::chunk> &chunks_to_assemble,
    std::vector<int> &chunks_to_assemble_domains)
{
    // In serial, communicating the chunks among ranks means passing them
    // back in the output arguments. We mark them as not-owned so we do not
    // double-free.
    for(size_t i = 0; i < chunks.size(); i++)
    {
        chunks_to_assemble.push_back(chunk(chunks[i].mesh, false));
        chunks_to_assemble_domains.push_back(dest_domain[i]);
    }
}

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::coordset --
//-----------------------------------------------------------------------------
namespace coordset
{

/**
 @brief The implmentation of conduit::blueprint::mesh::coordset::merge
*/
class point_merge
{
public:
    void execute(const std::vector<const conduit::Node *> &coordsets, 
                 double tolerance,
                 Node &output);

private:
    enum class coord_system
    {
        cartesian,
        cylindrical,
        spherical
    };

    int examine_extents(std::vector<std::vector<float64>> &extents) const;

    /**
    @brief Useful when none of the coordsets overlap. Combines all coordinates
        to one array.
    */
    void append_data(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension);

    /**
    @brief Used when coordsets overlap. Combines all coordinates into
        one array; merging points within tolerance.
    */
    void merge_data(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t, double tolerance);

    void create_output(index_t dimension, Node &output) const;

    /**
    @brief Iterates the coordinates in the given coordinate set,
        invoking the given lambda function with the signature
        void (float64 *point, index_t dim)
    NOTE: It will always be valid to index the "point" as if its
        "dim" was 3. For example if the original data was only 2D then
        point[2] will be 0 and dim will be 2.
    */
    template<typename Func>
    void iterate_coordinates(const Node &coordset, Func &&func);

    /**
    @brief Determines how many points there are and reserves space in member vectors
        new_points and old_to_new_ids
    @return npoints*dimension
    */
    index_t reserve_vectors(const std::vector<Node> &coordsets, index_t dimension);

    /**
    @brief The simple (slow) approach to merging the data based off distance.
    */
    void simple_merge_data(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension,
        double tolerance);

    void spatial_search_merge(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension,
        double tolerance);

    void truncate_merge(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension, double tolerance);

    static void xyz_to_rtp(double x, double y, double z, double &out_r, double &out_t, double &out_p);
    // TODO
    static void xyz_to_rz (double x, double y, double z, double &out_r, double &out_z);

    // TODO
    static void rz_to_xyz (double r, double z, double &out_x, double &out_y, double &out_z);
    // TODO
    static void rz_to_rtp (double r, double z, double &out_r, double &out_t, double &out_p);

    static void rtp_to_xyz(double r, double t, double p, double &out_x, double &out_y, double &out_z);
    // TODO
    static void rtp_to_rz (double r, double t, double p, double &out_r, double &out_z);

    static void translate_system(coord_system in_system, coord_system out_system,
        float64 p0, float64 p1, float64 p2, float64 &out_p1, float64 &out_p2, float64 &out_p3);

    /**
    @brief Returns the axis names for the given coordinate system
    */
    static const std::vector<std::string> &get_axes_for_system(coord_system);

    coord_system out_system;

    // Outputs
    std::vector<std::vector<index_t>> old_to_new_ids;
    std::vector<float64> new_coords;
};

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::coordset::utils --
//-----------------------------------------------------------------------------
namespace utils
{

/**
 @brief A simple vector struct to be used by the kdtree
*/
template<typename T, size_t Size>
struct vector
{
    using this_type = vector<T, Size>;
    using data_type = std::array<T, Size>;
    using value_type = T;
private:
    // Used to alias vector data
    template<size_t Index>
    struct accessor
    {
        data_type data;

        constexpr operator T() const
        {
            static_assert(Index < Size, "Invalid access into data.");
            return data[Index];
        }

        T operator=(T v)
        {
            static_assert(Index < Size, "Invalid access into data.");
            return data[Index] = v;
        }
    };

public:
    // Possible to access vector data with x/y/z
    union
    {
        data_type    v;
        accessor<0>  x;
        accessor<1>  y;
        accessor<2>  z;
    };

    constexpr size_t size() const
    {
        return Size;
    }

    T operator[](size_t index) const
    {
        return v[index];
    }

    T &operator[](size_t index)
    {
        return v[index];
    }

    void zero()
    {
        set_all(0);
    }

    void set_all(T val)
    {
        for(size_t i = 0u; i < size(); i++)
        {
            v[i] = val;
        }
    }

    // NOTE: Defining operator= makes this non-POD type
    // this_type operator=(const this_type &other)
    // {
    //     for(size_t i = 0u; i < size(); i++)
    //     {
    //         v[i] = other[i];
    //     }
    //     return *this;
    // }

    void copy(const this_type &other)
    {
        for(auto i = 0u; i < size(); i++)
            other.v[i] = v[i];
    }

    bool operator<=(const this_type &other) const
    {
        bool retval = true;
        for(size_t i = 0u; i < size(); i++)
            retval &= v[i] <= other[i];
        return retval;
    }

    bool operator>=(const this_type &other) const
    {
        bool retval = true;
        for(size_t i = 0u; i < size(); i++)
            retval &= v[i] >= other[i];
        return retval;
    }

    this_type operator+(T scalar) const
    {
        this_type retval;
        for(size_t i = 0u; i < size(); i++)
        {
            retval[i] = v[i] + scalar;
        }
        return retval;
    }

    this_type operator-(T scalar) const
    {
        this_type retval;
        for(size_t i = 0u; i < size(); i++)
        {
            retval[i] = v[i] - scalar;
        }
        return retval;
    }

    double distance2(const this_type &other) const
    {
        double d2 = 0.;
        for(size_t i = 0u; i < size(); i++)
        {
            const auto diff = other[i] - v[i];
            d2 += (diff*diff);
        }
        return d2;
    }

    double distance(const this_type &other) const
    {
        return std::sqrt(distance2(other));
    }
};

/**
 @brief A simple bounding box struct to be used by the kdtree
*/
template<typename VectorType>
struct bounding_box
{
    using value_type = typename VectorType::value_type;
    using data_type = typename VectorType::data_type;
    using T = value_type;
    VectorType min;
    VectorType max;

    bool contains(const VectorType &point) const
    {
        return (point >= min && point <= max);
    }

    bool contains(const VectorType &point, double tolerance) const
    {
        return (point >= (min - tolerance) && point <= (max + tolerance));
    }

    void expand(const VectorType &point)
    {
        for(size_t i = 0u; i < min.size(); i++)
        {
            min[i] = std::min(min[i], point[i]);
            max[i] = std::max(max[i], point[i]);
        }
    }
};

using vec2f = vector<float,2>;
using vec3f = vector<float,3>;
using vec2  = vector<double,2>;
using vec3  = vector<double,3>;

/**
 @brief A spatial search structure used to merge points within a given tolerance
*/
template<typename VectorType, typename DataType>
class kdtree
{
private:
    using Float = typename VectorType::value_type;
    // using IndexType = conduit_index_t;
public:
    constexpr static auto dimension = std::tuple_size<typename VectorType::data_type>::value;
    using vector_type = VectorType;
    using data_type = DataType;
    using IndexType = size_t;

    template<typename Precision, size_t D>
    struct kdnode
    {
        using node = kdnode<Precision, D>;
        std::vector<VectorType> points;
        std::vector<DataType> data;
        bounding_box<VectorType> bb;
        node *left{nullptr};
        node *right{nullptr};
        Float split{0.0};
        unsigned int dim;
        bool has_split{false};
    };
    using node = kdnode<Float, dimension>;
    using pair = std::pair<std::pair<node*,DataType&>, bool>;

    kdtree() = default;
    ~kdtree()
    {
        const auto lambda = [](node *node, unsigned int)
        {
            delete node;
        };
        if(root) { traverse_lrn(lambda, root); }
    }

    /**
    @brief Searches the tree for the given point using
            tolerance as an acceptable distance to merge like-points.
    */
    DataType *find_point(const VectorType &point, Float tolerance)
    {
        DataType *retval = nullptr;
        if(!root)
        {
            retval = nullptr;
        }
        else if(root->bb.contains(point, tolerance))
        {
            retval = find_point(root, 0, point, tolerance);
        }
        return retval;
    }

    void insert(const VectorType &point, const DataType &r)
    {
        scratch.reserve(point_vec_size*2);
        if(!root)
        {
            root = create_node(point, r);
            npoints++;
        }
        else
        {
            insert(root, 0, point, r);
        }
    }

    IndexType size()  const { return npoints; }
    IndexType nodes() const { return nnodes; }
    IndexType depth() const { return tree_depth; }

    void set_bucket_size(IndexType n) { point_vec_size = n; }
    IndexType get_bucket_size() const { return point_vec_size; }

    template<typename Func>
    void iterate_points(Func &&func)
    {
        IndexType point_id = 0u;
        const auto lambda = [&](node *n, unsigned int) {
            for(IndexType i = 0u; i < n->points.size(); i++)
            {
                func(point_id, n->points[i], n->data[i]);
                point_id++;
            }
        };
        if(root) { traverse_lnr(lambda, root); }
    }

    template<typename Func>
    void traverse_nodes(Func &&func)
    {
        if(root) { traverse_lnr(func, root); }
    }

private:
    // Create an empty node
    node *create_node()
    {
        node *newnode = new node;
        newnode->points.reserve(point_vec_size);
        newnode->data.reserve(point_vec_size);
        newnode->bb.min[0] = std::numeric_limits<Float>::max();
        newnode->bb.min[1] = std::numeric_limits<Float>::max();
        newnode->bb.min[2] = std::numeric_limits<Float>::max();
        newnode->bb.max[0] = std::numeric_limits<Float>::lowest();
        newnode->bb.max[1] = std::numeric_limits<Float>::lowest();
        newnode->bb.max[2] = std::numeric_limits<Float>::lowest();
        newnode->left = nullptr;
        newnode->right = nullptr;
        newnode->split = 0;
        newnode->dim = 0;
        newnode->has_split = false;
        nnodes++;
        return newnode;
    }

    // Create a node with initial values inserted
    node *create_node(VectorType loc, const DataType &r)
    {
        node *newnode = create_node();
        node_add_data(newnode, loc, r);
        return newnode;
    }

    static void node_add_data(node *n, const VectorType &p, const DataType &d)
    {
        n->bb.expand(p);
        n->points.push_back(p);
        n->data.push_back(d);
    }

    /**
    @brief Splits the given node and inserts point/data into the proper child
    */
    void node_split(node *n, const VectorType &point, const DataType &data)
    {
        // Determine which dim to split on
        IndexType dim = 0;
        {
            Float longest_dim = std::numeric_limits<Float>::lowest();
            for(IndexType i = 0; i < n->bb.min.size(); i++)
            {
                const Float dim_len = n->bb.max[i] - n->bb.min[i];
                if(longest_dim < dim_len)
                {
                    dim = i;
                    longest_dim = dim_len;
                }
            }
            n->dim = dim;
        }

        // Determine what value on the dim to split on
        {
            scratch.clear();
            for(IndexType i = 0; i < point_vec_size; i++)
            {
                scratch.push_back(i);
            }
            std::sort(scratch.begin(), scratch.end(), [=](IndexType i0, IndexType i1) {
                return n->points[i0][dim] < n->points[i1][dim];
            });

            // If the index stored in scratch is point_vec_size
            const IndexType scratch_idx = scratch.size() / 2;
            const IndexType median_idx = scratch[scratch_idx];
            Float median = n->points[median_idx][dim];
            // Check if the new point is our actual median
            if(point[dim] > n->points[scratch[scratch_idx-1]][dim] && point[dim] < median)
            {
                median = point[dim];
            }

            n->split = median;
            n->left = create_node();
            n->right = create_node();
            n->has_split = true;

            for(IndexType i = 0; i < point_vec_size; i++)
            {
                const Float temp = n->points[i][dim];
                if(temp < median)
                {
                    node_add_data(n->left, n->points[i], n->data[i]);
                }
                else
                {
                    node_add_data(n->right, n->points[i], n->data[i]);
                }
            }

            if(point[dim] < median)
            {
                node_add_data(n->left, point, data);
            }
            else
            {
                node_add_data(n->right, point, data);
            }

            // Clear the data from the parent node
            std::vector<VectorType>{}.swap(n->points);
            std::vector<DataType>{}.swap(n->data);
        }
    }

    DataType *find_point(node *current, unsigned int depth, const VectorType &point, Float tolerance)
    {
        // If we got here we know that the point was in this node's bounding box
        DataType *retval = nullptr;

        // This node has children
        if(current->has_split)
        {
            const bool left_contains = current->left->bb.contains(point, tolerance);
            const bool right_contains = current->right->bb.contains(point, tolerance);
            if(!left_contains && !right_contains)
            {
                // ERROR! This shouldn't happen, the tree must've been built improperly
                retval = nullptr;
            }
            else if(left_contains)
            {
                // Traverse left
                retval = find_point(current->left, depth+1, point, tolerance);
            }
            else if(right_contains)
            {
                // Traverse right
                retval = find_point(current->right, depth+1, point, tolerance);
            }
            else // (left_contains && right_contains)
            {
                // Rare, but possible due to tolerance.
                // Check if the left side has the point without tolerance
                const bool pref_left = current->left->bb.contains(point);
                retval = (pref_left)
                    ? find_point(current->left, depth+1, point, tolerance)
                    : find_point(current->right, depth+1, point, tolerance);
                // We tried the preferred side but it didn't contain the point
                if(retval == nullptr)
                {
                    retval = (pref_left)
                        ? find_point(current->right, depth+1, point, tolerance)
                        : find_point(current->left, depth+1, point, tolerance);
                }
            }
        }
        else
        {
            // This is a leaf node.
            const auto t2 = tolerance * tolerance;
            const IndexType N = current->points.size();
            IndexType idx = 0;
            for(idx = 0; idx < N; idx++)
            {
                const auto &p = current->points[idx];
                const auto dist2 = point.distance2(p);
                if(dist2 <= t2)
                {
                    break;
                }
            }

            // Did not find point
            if(idx == N)
            {
                retval = nullptr;
            }
            else
            {
                retval = &current->data[idx];
            }
        }
        return retval;
    }

    void insert(node *current, unsigned int depth, const VectorType &loc, const DataType &r)
    {
        // No matter what we need to add this point to the current bounding box
        current->bb.expand(loc);

        // This node has children
        if(current->has_split)
        {
            const auto dim = current->dim;
            if(loc[dim] < current->split)
            {
                // Go left
                insert(current->left, depth+1, loc, r);
            }
            else // (loc[dim] >= current->split)
            {
                // Go right
                insert(current->right, depth+1, loc, r);
            }
        }
        else
        {
            // This is a leaf node
            // Determine if the node needs to be split
            if((current->points.size()) == point_vec_size)
            {
                // This will add the point and data to the correct child
                node_split(current, loc, r);
                tree_depth = std::max(tree_depth, (IndexType)depth+1);
                npoints++;
            }
            else
            {
                // This node does not need to be split
                node_add_data(current, loc, r);
                npoints++;
            }
        }
    }

    template<typename Func>
    void traverse_lnr(Func &&func, node *node, unsigned int depth = 0)
    {
        if(node->left) { traverse_lnr(func, node->left, depth + 1); }
        func(node, depth);
        if(node->right) { traverse_lnr(func, node->right, depth + 1); }
    }

    template<typename Func>
    void traverse_lrn(Func &&func, node *node, unsigned int depth = 0)
    {
        if(node->left) { traverse_lrn(func, node->left, depth + 1); }
        if(node->right) { traverse_lrn(func, node->right, depth + 1); }
        func(node, depth);
    }

    // Keep track of tree performance
    IndexType npoints{0u};
    IndexType nnodes{0u};
    IndexType tree_depth{0u};

    node *root{nullptr};
    IndexType point_vec_size{32};
    std::vector<IndexType> scratch;
};

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::coordset::utils --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
point_merge::execute(const std::vector<const Node *> &coordsets,
                     double tolerance,
                     Node &output)
{
    if(coordsets.empty())
        return;

    if(coordsets.size() == 1)
    {
        if(coordsets[0] != nullptr)
        {
            output.reset();
            output["coordsets/coords"] = *coordsets[0];
        }
        return;
    }

    // What do we need to know about the coordsets before we execute the algorithm
    //  - Systems
    //  - Types
    //  - Dimension

    std::vector<Node> working_sets;
    std::vector<coord_system> systems;
    std::vector<std::vector<float64>> extents;
    index_t ncartesian = 0, ncylindrical = 0, nspherical = 0;
    index_t dimension = 0;
    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const Node *cset = coordsets[i];
        if(!cset)
        {
            // ERROR! You passed me a nullptr for a coordset!
            continue;
        }

        if(!cset->has_child("type"))
        {
            // Error! Invalid coordset!
            continue;
        }
        const std::string type = cset->child("type").as_string();

        // Track some information about each set
        dimension = std::max(dimension, mesh::utils::coordset::dims(*cset));
        extents.push_back(mesh::utils::coordset::extents(*cset));

        // Translate coordsystem string to enum
        std::string system = mesh::utils::coordset::coordsys(*cset);
        if(system == "cylindrical")
        {
            ncylindrical++;
            systems.push_back(coord_system::cylindrical);
        }
        else if(system == "spherical")
        {
            nspherical++;
            systems.push_back(coord_system::spherical);
        }
        else // system == cartesian
        {
            ncartesian++;
            systems.push_back(coord_system::cartesian);
        }

        // We only work on explicit sets
        working_sets.emplace_back();
        if(type == "uniform")
        {
            mesh::coordset::uniform::to_explicit(*cset, working_sets.back());
        }
        else if(type == "rectilinear")
        {
            mesh::coordset::rectilinear::to_explicit(*cset, working_sets.back());
        }
        else // type == "explicit"
        {
            working_sets.back() = *cset;
        }
    }

    // Determine best output coordinate system
    // Prefer cartesian, if they are all the same use that coordinate system
    if(ncartesian > 0 || (ncylindrical > 0 && nspherical > 0))
    {
        out_system = coord_system::cartesian;
    }
    else if(nspherical > 0)
    {
        out_system = coord_system::spherical;
    }
    else if(ncylindrical > 0)
    {
        out_system = coord_system::cylindrical;
    }
    else
    {
        // Error! Unhandled case!
        std::cerr << "UNHANDLED CASE " << ncartesian << " " << ncylindrical << " " << nspherical << std::endl;
        return;
    }

    int noverlapping_sets = examine_extents(extents);
    PM_DEBUG_PRINT("noverlapping sets: " << noverlapping_sets << std::endl);
    if(noverlapping_sets == 0)
    {
        // We don't need to do any kind of merging
        append_data(working_sets, systems, dimension);
    }
    else
    {
        merge_data(working_sets, systems, dimension, tolerance);
    }

    create_output(dimension, output);
}

//-----------------------------------------------------------------------------
int
point_merge::examine_extents(std::vector<std::vector<float64>> &extents) const
{
    const auto overlap = [](const float64 box1[6], const float64 box2[6]) -> bool {
        bool retval = true;
        for(auto i = 0u; i < 3u; i++)
        {
            const auto idx  = i * 2;
            const auto min1 = box1[idx];
            const auto max1 = box1[idx+1];
            const auto min2 = box2[idx];
            const auto max2 = box2[idx+1];
            retval &= (max1 >= min2 && max2 >= min1);
        }
        return retval;
    };

    int retval = 0;
    for(auto i = 0u; i < extents.size(); i++)
    {
        float64 box1[6] = {0.,0.,0.,0.,0.,0.};
        const auto &ext1 = extents[i];
        for(auto j = 0u; j < ext1.size(); j++)
        {
            box1[j] = ext1[j];
        }

        for(auto j = 0u; j < extents.size(); j++)
        {
            if(i == j) { continue; }
            const auto ext2 = extents[j];
            float64 box2[6] = {0.,0.,0.,0.,0.,0.};
            for(auto k = 0u; k < ext2.size(); k++)
            {
                box2[k] = ext2[k];
            }

            retval += overlap(box1, box2);
        }
    }
    return retval;
}

//-----------------------------------------------------------------------------
void
point_merge::append_data(const std::vector<Node> &coordsets,
    const std::vector<coord_system> &systems, index_t dimension)
{
    reserve_vectors(coordsets, dimension);

    index_t newid = 0;
    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const auto append = [&](float64 *p, index_t)
        {
            old_to_new_ids[i].push_back(newid);
            for(auto i = 0; i < dimension; i++)
            {
                new_coords.push_back(p[i]);
            }
            newid++;
        };

        const auto translate_append = [&](float64 *p, index_t d) {
            translate_system(systems[i], out_system,
                p[0], p[1], p[2], p[0], p[1], p[2]);
            append(p, d);
        };


        if(systems[i] == out_system)
        {
            iterate_coordinates(coordsets[i], append);
        }
        else
        {
            iterate_coordinates(coordsets[i], translate_append);
        }
    }
}

//-----------------------------------------------------------------------------
void
point_merge::merge_data(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension, double tolerance)
{
#define USE_SPATIAL_SEARCH_MERGE
#if   defined(USE_TRUNCATE_PRECISION_MERGE)
    truncate_merge(coordsets, systems, dimension, tolerance);
#elif defined(USE_SPATIAL_SEARCH_MERGE)
    spatial_search_merge(coordsets, systems, dimension, tolerance);
#else
    simple_merge_data(coordsets, systems, dimension, tolerance);
#endif
#undef USE_SPATIAL_SEARCH_MERGE
}

//-----------------------------------------------------------------------------
void
point_merge::create_output(index_t dimension, Node &output) const
{
    if(dimension < 0 || dimension > 3)
    {
        // ERROR! Invalid dimension!
        return;
    }

    output.reset();

    // Add the new coordset
    {
        output["type"] = "explicit";
        auto &values = output.add_child("values");

        // Define the node schema for coords
        Schema s;
        const auto npoints = new_coords.size() / dimension;
        const index_t stride = sizeof(float64) * dimension;
        const index_t size = sizeof(float64);
        const auto &axes = get_axes_for_system(out_system);
        for(auto i = 0; i < dimension; i++)
        {
            s[axes[i]].set(DataType::float64(npoints,i*size,stride));
        }

        // Copy out coordinate values
        values.set(s);
        float64_array coord_arrays[3];
        for(auto i = 0; i < dimension; i++)
        {
            coord_arrays[i] = values[axes[i]].value();
        }

        index_t point_id = 0;
        for(auto itr = new_coords.begin(); itr != new_coords.end();)
        {
            for(auto d = 0; d < dimension; d++)
            {
                coord_arrays[d][point_id] = *itr++;
            }
            point_id++;
        }
    }

    // Add the pointmaps
    {
        auto &pointmaps = output["pointmaps"];
        for(const auto &idmap : old_to_new_ids)
        {
            const auto size = idmap.size();
            // Create the list entry
            auto &ids = pointmaps.append();
            ids.set(DataType::index_t(size));
            // Copy the contents into the node
            DataArray<index_t> ids_data = ids.value();
            for(size_t i = 0u; i < size; i++)
            {
                ids_data[i] = idmap[i];
            }
        }
    }
}

//-----------------------------------------------------------------------------
template<typename Func>
void
point_merge::iterate_coordinates(const Node &coordset, Func &&func)
{
    if(!coordset.has_child("type"))
        return;

    if(coordset["type"].as_string() != "explicit")
        return;

    if(!coordset.has_child("values"))
        return;

    const Node &coords = coordset["values"];

    // Fetch the nodes for the coordinate values
    const Node *xnode = coords.fetch_ptr("x");
    const Node *ynode = nullptr, *znode = nullptr;
    if(xnode)
    {
        // Cartesian
        ynode = coords.fetch_ptr("y");
        znode = coords.fetch_ptr("z");
    }
    else if((xnode = coords.fetch_ptr("r")))
    {
        if((ynode = coords.fetch_ptr("z")))
        {
            // Cylindrical
        }
        else if((ynode = coords.fetch_ptr("theta")))
        {
            // Spherical
            znode = coords.fetch_ptr("phi");
        }
    }

    // Iterate accordingly
    float64 p[3] {0., 0., 0.};
    if(xnode && ynode && znode)
    {
        // 3D
        const auto xtype = xnode->dtype();
        const auto ytype = ynode->dtype();
        const auto ztype = znode->dtype();
        // TODO: Handle different types
        auto xarray = xnode->as_double_array();
        auto yarray = ynode->as_double_array();
        auto zarray = znode->as_double_array();
        const index_t N = xarray.number_of_elements();
        for(index_t i = 0; i < N; i++)
        {
            p[0] = xarray[i]; p[1] = yarray[i]; p[2] = zarray[i];
            func(p, 3);
        }
    }
    else if(xnode && ynode)
    {
        // 2D
        const auto xtype = xnode->dtype();
        const auto ytype = ynode->dtype();
        // TODO: Handle different types
        auto xarray = xnode->as_double_array();
        auto yarray = ynode->as_double_array();
        const index_t N = xarray.number_of_elements();
        for(index_t i = 0; i < N; i++)
        {
            p[0] = xarray[i]; p[1] = yarray[i]; p[2] = 0.;
            func(p, 2);
        }
    }
    else if(xnode)
    {
        // 1D
        const auto xtype = xnode->dtype();
        // TODO: Handle different types
        auto xarray = xnode->as_double_array();
        const index_t N = xarray.number_of_elements();
        for(index_t i = 0; i < N; i++)
        {
            p[0] = xarray[i]; p[1] = 0.; p[2] = 0.;
            func(p, 1);
        }
    }
    else
    {
        // ERROR! No valid nodes passed.
    }
}

//-----------------------------------------------------------------------------
index_t
point_merge::reserve_vectors(const std::vector<Node> &coordsets, index_t dimension)
{
    old_to_new_ids.reserve(coordsets.size());
    index_t new_size = 0;
    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const Node *values = coordsets[i].fetch_ptr("values");
        index_t npts = 0;
        if(values)
        {
            const Node *xnode = values->fetch_ptr("x");
            if(!xnode)
            {
                xnode = values->fetch_ptr("r");
            }

            if(xnode)
            {
                npts = xnode->dtype().number_of_elements();
            }
        }

        old_to_new_ids.push_back({});
        old_to_new_ids.back().reserve(npts);
        new_size += npts*dimension;
    }
    new_coords.reserve(new_size);
    return new_size;
}

//-----------------------------------------------------------------------------
void
point_merge::simple_merge_data(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension, double tolerance)
{
    PM_DEBUG_PRINT("Simple merging!" << std::endl);
    reserve_vectors(coordsets, dimension);

    const auto t2 = (tolerance*tolerance);
    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const index_t end_check = (index_t)new_coords.size();
        const Node &coordset = coordsets[i];
        auto &idmap = old_to_new_ids[i];

        // To be invoked on each coordinate
        const auto merge = [&](float64 *p, index_t) {
            for(index_t idx = 0; idx < end_check; idx += dimension)
            {
                float64 dist2 = 0.;
                for(index_t d = 0; d < dimension; d++)
                {
                    const auto diff = p[d] - new_coords[idx+d];
                    dist2 += (diff*diff);
                }

                // Within tolerance!
                if(dist2 < t2)
                {
                    // idx / dim should truncate to the proper id
                    idmap.push_back(idx / dimension);
                    return;
                }
            }

            // Did not find another point within tolerance
            const auto newid = (index_t)(new_coords.size() / dimension);
            idmap.push_back(newid);
            for(index_t d = 0; d < dimension; d++)
            {
                new_coords.push_back(p[d]);
            }
        };

        const auto translate_merge = [&](float64 *p, index_t d) {
            translate_system(systems[i], coord_system::cartesian,
                p[0], p[1], p[2], p[0], p[1], p[2]);
            merge(p, d);
        };

        // Invoke the proper lambda on each coordinate
        if(systems[i] != coord_system::cartesian)
        {
            iterate_coordinates(coordset, translate_merge);
        }
        else
        {
            iterate_coordinates(coordset, merge);
        }
    }
}

//-----------------------------------------------------------------------------
void
point_merge::spatial_search_merge(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension,
        double tolerance)
{
    PM_DEBUG_PRINT("Spatial search merging!" << std::endl);
    reserve_vectors(coordsets, dimension);

    using namespace utils;
    kdtree<vec3, index_t> point_records;
    point_records.set_bucket_size(32);
    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const auto &coordset = coordsets[i];

        // To be invoked on every coordinate
        const auto merge = [&](float64 *p, index_t) {
            vec3 key;
            key.v[0] = p[0]; key.v[1] = p[1]; key.v[2] = p[2];
            const auto potential_id = new_coords.size() / dimension;
            const index_t *existing_id = point_records.find_point(key, tolerance);
            // Point wasn't already in the tree, insert it
            if(!existing_id)
            {
                old_to_new_ids[i].push_back(potential_id);
                for(index_t d = 0; d < dimension; d++)
                {
                    new_coords.push_back(p[d]);
                }
                // Store the point in the tree so we can look it up later
                point_records.insert(key, potential_id);
            }
            else
            {
                // Point already existed in the tree, reference the known id
                old_to_new_ids[i].push_back(*existing_id);
            }
        };

        const auto translate_merge = [&](float64 *p, index_t d) {
            translate_system(systems[i], coord_system::cartesian,
                p[0], p[1], p[2], p[0], p[1], p[2]);
            merge(p, d);
        };

        // Invoke the proper lambda on each coordinate
        if(systems[i] != coord_system::cartesian)
        {
            iterate_coordinates(coordset, translate_merge);
        }
        else
        {
            iterate_coordinates(coordset, merge);
        }
    }

    PM_DEBUG_PRINT("Number of points in tree " << point_records.size()
        << ", depth of tree " << point_records.depth()
        << ", nodes in tree " << point_records.nodes() << std::endl);
}

//-----------------------------------------------------------------------------
void
point_merge::truncate_merge(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension, double tolerance)
{
    PM_DEBUG_PRINT("Truncate merging!" << std::endl);
    // Determine what to scale each value by
    // TODO: Be dynamic
    (void)tolerance;
    double scale = 0.;
    {
        auto decimal_places = 4u;
        static const std::array<double, 7u> lookup = {
            1.,
            (2u << 4),
            (2u << 7),
            (2u << 10),
            (2u << 14),
            (2u << 17),
            (2u << 20)
        };
        if(decimal_places < lookup.size())
        {
            scale = lookup[decimal_places];
        }
        else
        {
            scale = lookup[6];
        }
    }

    /*index_t size = */reserve_vectors(coordsets, dimension);

    // Iterate each of the coordinate sets
    using fp_type = int64;
    using tup = std::tuple<fp_type, fp_type, fp_type>;
    std::map<tup, index_t> point_records;

    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const auto &coordset = coordsets[i];

        // To be invoked on every coordinate
        const auto merge = [&](float64 *p, index_t) {
            tup key = std::make_tuple(
                static_cast<fp_type>(std::round(p[0] * scale)),
                static_cast<fp_type>(std::round(p[1] * scale)),
                static_cast<fp_type>(std::round(p[2] * scale)));
            auto res = point_records.insert({key, {}});
            if(res.second)
            {
                const index_t id = (index_t)(new_coords.size() / dimension);
                res.first->second = id;
                old_to_new_ids[i].push_back(id);
                for(index_t j = 0; j < dimension; j++)
                {
                    new_coords.push_back(p[j]);
                }
            }
            else
            {
                old_to_new_ids[i].push_back(res.first->second);
            }
        };

        const auto translate_merge = [&](float64 *p, index_t d) {
            translate_system(systems[i], out_system,
                p[0], p[1], p[2], p[0], p[1], p[2]);
            merge(p, d);
        };

        // Invoke the proper lambda on each coordinate
        if(systems[i] != out_system)
        {
            iterate_coordinates(coordset, translate_merge);
        }
        else
        {
            iterate_coordinates(coordset, merge);
        }
    }
}

//-----------------------------------------------------------------------------
void
point_merge::xyz_to_rtp(double x, double y, double z, double &out_r, double &out_t, double &out_p)
{
    const auto r = std::sqrt(x*x + y*y + z*z);
    out_r = r;
    out_t = std::acos(r / z);
    out_p = std::atan(y / x);
}

//-----------------------------------------------------------------------------
void
point_merge::rtp_to_xyz(double r, double t, double p, double &out_x, double &out_y, double &out_z)
{
    out_x = r * std::cos(p) * std::sin(t);
    out_y = r * std::sin(p) * std::sin(t);
    out_z = r * std::cos(t);
}

//-----------------------------------------------------------------------------
void
point_merge::translate_system(coord_system in_system, coord_system out_system,
        float64 p0, float64 p1, float64 p2,
        float64 &out_p0, float64 &out_p1, float64 &out_p2)
{
    // TODO: Handle rz
    switch(out_system)
    {
    case coord_system::cartesian:
        switch(in_system)
        {
        case coord_system::cylindrical:
            // TODO
            break;
        case coord_system::spherical:
            rtp_to_xyz(p0,p1,p2, out_p0,out_p1,out_p2);
            break;
        default:    // case coord_system:: cartesian Can't happen
            out_p0 = p0;
            out_p1 = p1;
            out_p2 = p2;
            break;
        }
        break;
    case coord_system::cylindrical:
        switch(in_system)
        {
        case coord_system::cartesian:
            // TODO
            break;
        case coord_system::spherical:
            // TODO
            break;
        default:
            out_p0 = p0;
            out_p1 = p1;
            out_p2 = p2;
            break;
        }
        break;
    case coord_system::spherical:
        switch(in_system)
        {
        case coord_system::cartesian:
            xyz_to_rtp(p0,p1,p2, out_p0,out_p1,out_p2);
            break;
        case coord_system::cylindrical:
            // TODO
            break;
        default:
            out_p0 = p0;
            out_p1 = p1;
            out_p2 = p2;
            break;
        }
        break;
    }
}

//-----------------------------------------------------------------------------
const std::vector<std::string> &
point_merge::get_axes_for_system(coord_system cs)
{
    const std::vector<std::string> *retval = &mesh::utils::CARTESIAN_AXES;
    switch(cs)
    {
    case coord_system::cylindrical:
        retval = &mesh::utils::CYLINDRICAL_AXES;
        break;
    case coord_system::spherical:
        retval = &mesh::utils::SPHERICAL_AXES;
        break;
    case coord_system::cartesian: // Do nothing
        break;
    }
    return *retval;
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::coordset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::topology --
//-----------------------------------------------------------------------------
namespace topology
{

struct entity
{
    utils::ShapeType                  shape;
    // utils::ShapeCascade               cascade; // User can make the cascade if they want using the shape
    std::vector<index_t>              element_ids;
    std::vector<std::vector<index_t>> subelement_ids;
    index_t                           entity_id; // Local entity id.
};

// static const std::vector<std::string> TOPO_SHAPES = {"point", "line", "tri", "quad", "tet", "hex", "polygonal", "polyhedral"};
// Q: Why doesn't this exist in conduit_blueprint_mesh_utils.hpp ?
enum class ShapeId : index_t
{
    Vertex     = 0,
    Line       = 1,
    Tri        = 2,
    Quad       = 3,
    Tet        = 4,
    Hex        = 5,
    Polygonal  = 6,
    Polyhedral = 7
};

template<typename Func>
static void iterate_elements(const Node &topo, Func &&func)
{
/*
Multiple topology formats
0. Single shape topology - Fixed celltype, polygonal celltype, polyhedral celltype
  topo:
    coordset: (String name of coordset)
    type: unstructured
    elements:
      shape: (String name of shape)
      connectivity: (Integer array of vertex ids)
1. Mixed shape topology 1 - Fixed celltype, polygonal celltype, polyhedral celltype
  topo:
    coordset: (String name of coordset)
    type: unstructured
    elements:
      -
        shape: (String name of shape)
        connectivity: (Integer array of vertex ids)
2. Mixed shape topology 2 - Fixed celltype, polygonal celltype, polyhedral celltype
  topo:
    coordset: (String name of coordset)
    type: unstructured
    elements:
      (String name):
        shape: (String name of shape)
        connectivity: (Integer array of vertex ids)
3. Stream based toplogy 1
  mesh: 
    type: "unstructured"
    coordset: "coords"
    elements: 
      element_types: 
        (String name): (Q: Could this also be a list entry?) 
          stream_id: (index_t id)
          shape: (String name of shape)
      element_index: 
        stream_ids: (index_t array of stream ids, must be one of the ids listed in element_types)
        element_counts: (index_t array of element counts at a given index (IE 2 would mean 2 of the associated stream ID))
      stream: (index_t array of vertex ids)
4. Stream based topology 2
  topo: 
    type: unstructured
    coordset: (String name of coordset)
    elements: 
      element_types: 
        (String name): (Q: Could this also be a list entry?)
          stream_id: (index_t id)
          shape: (String name of shape)
      element_index: 
        stream_ids: (index_t array of stream ids, must be one of the ids listed in element_types)
        offsets: (index_t array of offsets into stream for each element)
      stream: (index_t array of vertex ids)
*/

    int case_num = -1;
    // Determine case number
    {
        const Node *shape = topo.fetch_ptr("elements/shape");
        if(shape)
        {
            // This is a single shape topology
            const utils::ShapeType st(shape->as_string());
            if(!st.is_valid())
            {
                CONDUIT_ERROR("Invalid topology passed to iterate_elements.");
                return;
            }
            else
            {
                case_num = 0;
            }
        }
        else
        {
            const Node *elements = topo.fetch_ptr("elements");
            if(!elements)
            {
                CONDUIT_ERROR("Invalid topology passed to iterate elements, no \"elements\" node.");
                return;
            }

            const Node *etypes = elements->fetch_ptr("element_types");
            const Node *eindex = elements->fetch_ptr("element_index");
            const Node *estream = elements->fetch_ptr("stream");
            if(!etypes && !eindex && !estream)
            {
                // Not a stream based toplogy, either a list or object of element buckets
                if(elements->dtype().is_list())
                {
                    case_num = 1;
                }
                else if(elements->dtype().is_object())
                {
                    case_num = 2;
                }
            }
            else if(etypes && eindex && estream)
            {
                // Stream based topology, either offsets or counts
                const Node *eoffsets = eindex->fetch_ptr("offsets");
                const Node *ecounts  = eindex->fetch_ptr("element_counts");
                if(ecounts)
                {
                    case_num = 3;
                }
                else if(eoffsets)
                {
                    case_num = 4;
                }
            }
        }
    }
    if(case_num < 0)
    {
        CONDUIT_ERROR("Could not figure out the type of toplogy passed to iterate elements.");
        return;
    }

    // Define the lambda functions to be invoked for each topology type
    const auto traverse_fixed_elements = [&func](const Node &eles, const utils::ShapeType &shape, index_t &ent_id) {
        // Single celltype
        entity e;
        e.shape = shape;
        const auto ent_size = e.shape.indices;
        e.element_ids.resize(ent_size, 0);

        const Node &conn = eles["connectivity"];
        const auto &conn_dtype = conn.dtype();
        const auto &id_dtype = DataType(conn_dtype.id(), 1);
        const index_t nents = conn_dtype.number_of_elements() / ent_size;
        Node temp;
        index_t ei = 0;
        for(index_t i = 0; i < nents; i++)
        {
            e.entity_id = ent_id;
            for(index_t j = 0; j < ent_size; j++)
            {
                // Pull out vertex id at ei then cast to index_t
                temp.set_external(id_dtype, const_cast<void*>(conn.element_ptr(ei)));
                e.element_ids[j] = temp.to_index_t();
                ei++;
            }

            func(e);
            ent_id++;
        }
    };

    const auto traverse_polygonal_elements = [&func](const Node &elements, index_t &ent_id) {
        entity e;
        e.shape = utils::ShapeType((index_t)ShapeId::Polygonal);
        const Node &conn = elements["connectivity"];
        const Node &sizes = elements["sizes"];
        const auto &sizes_dtype = sizes.dtype();
        const DataType size_dtype(sizes_dtype.id(), 1);
        const DataType id_dtype(sizes.dtype().id(), 1);
        const index_t nents = sizes_dtype.number_of_elements();
        Node temp;
        index_t ei = 0;
        for(index_t i = 0; i < nents; i++)
        {
            e.entity_id = ent_id;
            temp.set_external(size_dtype, const_cast<void*>(sizes.element_ptr(i)));
            const index_t sz = temp.to_index_t();
            e.element_ids.resize(sz);
            for(index_t j = 0; j < sz; j++)
            {
                // Pull out vertex id at ei then cast to index_t
                temp.set_external(id_dtype, const_cast<void*>(conn.element_ptr(ei)));
                e.element_ids[j] = temp.to_index_t();
                ei++;
            }

            func(e);
            ent_id++;
        }
    };

    const auto traverse_polyhedral_elements = [&func](const Node &elements, const Node &subelements, index_t &ent_id) {
        entity e;
        e.shape = utils::ShapeType((index_t)ShapeId::Polyhedral);
        const Node &conn = elements["connectivity"];
        const Node &sizes = elements["sizes"];
        const Node &subconn = subelements["connectivity"];
        const Node &subsizes = subelements["sizes"];
        const Node &suboffsets = subelements["offsets"];
        const auto &sizes_dtype = sizes.dtype();
        const DataType size_dtype(sizes_dtype.id(), 1);
        const DataType id_dtype(sizes.dtype().id(), 1);
        const DataType subid_dtype(subconn.dtype().id(), 1);
        const DataType suboff_dtype(suboffsets.dtype().id(), 1);
        const DataType subsize_dtype(subsizes.dtype().id(), 1);
        const index_t nents = sizes_dtype.number_of_elements();
        Node temp;
        index_t ei = 0;
        for(index_t i = 0; i < nents; i++)
        {
            e.entity_id = ent_id;
            temp.set_external(size_dtype, const_cast<void*>(sizes.element_ptr(i)));
            const index_t sz = temp.to_index_t();
            e.element_ids.resize(sz);
            for(index_t j = 0; j < sz; j++)
            {
                // Pull out vertex id at ei then cast to index_t
                temp.set_external(id_dtype, const_cast<void*>(conn.element_ptr(ei)));
                e.element_ids[j] = temp.to_index_t();
                ei++;
            }

            e.subelement_ids.resize(sz);
            for(index_t j = 0; j < sz; j++)
            {
                // Get the size of the subelement so we can define it in the proper index of subelement_ids
                auto &subele = e.subelement_ids[j];
                temp.set_external(subsize_dtype, const_cast<void*>(subsizes.element_ptr(e.element_ids[j])));
                const index_t subsz = temp.to_index_t();
                subele.resize(subsz);

                // Find the offset of the face definition so we can write the vertex ids
                temp.set_external(suboff_dtype, const_cast<void*>(suboffsets.element_ptr(e.element_ids[j])));
                index_t offset = temp.to_index_t();
                for(index_t k = 0; k < subsz; k++)
                {
                    temp.set_external(subid_dtype, const_cast<void*>(subconn.element_ptr(offset)));
                    subele[k] = temp.to_index_t();
                    offset++;
                }
            }

            func(e);
            ent_id++;
        }
    };

    using id_elem_pair =  std::pair<index_t, entity>;
    const auto build_element_vector = [](const Node &element_types, std::vector<id_elem_pair> &eles)
    {
    /*
      element_types: 
        (String name): (Q: Could this also be a list entry?) 
          stream_id: (index_t id)
          shape: (String name of shape)
    */
        eles.clear();
        auto itr = element_types.children();
        while(itr.has_next())
        {
            const Node &n = itr.next();
            const index_t id = n["stream_id"].to_index_t();
            const utils::ShapeType shape(n["shape"].as_string());
            eles.push_back({{id}, {}});
            eles.back().second.shape = shape;
            if(!shape.is_poly())
            {
                eles.back().second.element_ids.resize(shape.indices);
            }
            else
            {
                CONDUIT_ERROR("I cannot handle a stream of polygonal/polyhedral elements!");
                return;
            }
        }
    };

    index_t ent_id = 0;
    switch(case_num)
    {
    case 0:
    {
        utils::ShapeType shape(topo);
        if(shape.is_polyhedral())
        {
            traverse_polyhedral_elements(topo["elements"], topo["subelements"], ent_id);
        }
        else if(shape.is_polygonal())
        {
            traverse_polygonal_elements(topo["elements"], ent_id);
        }
        else // (known celltype case)
        {
            traverse_fixed_elements(topo["elements"], shape, ent_id);
        }
        break;
    }
    case 1: /* Fallthrough */
    case 2:
    {
        // Mixed celltype
        const Node &elements = topo["elements"];
        auto ele_itr = elements.children();
        while(ele_itr.has_next())
        {
            const Node &bucket = ele_itr.next();
            const std::string &shape_name = bucket["shape"].as_string();
            utils::ShapeType shape(shape_name);

            if(shape.is_polyhedral())
            {
                // Need to find corresponding subelements
                const std::string bucket_name = bucket.name();
                if(!topo.has_child("subelements"))
                {
                    CONDUIT_ERROR("Invalid toplogy, shape == polygonal but no subelements node present.");
                    return;
                }
                const Node &subelements = topo["subelements"];
                if(!subelements.has_child(bucket_name))
                {
                    CONDUIT_ERROR("Invalid toplogy, shape == polygonal but no matching subelements node present.");
                    return;
                }
                const Node &subbucket = subelements[bucket_name];
                traverse_polyhedral_elements(bucket, subbucket, ent_id);
            }
            else if(shape.is_polygonal())
            {
                traverse_polygonal_elements(bucket, ent_id);
            }
            else
            {
                traverse_fixed_elements(bucket, shape, ent_id);
            }
        }
        break;
    }
    case 3: /* Fallthrough */
    case 4:
    {
        // Stream with element counts or offsets
        const Node &elements = topo["elements"];
        std::vector<id_elem_pair> etypes;
        build_element_vector(elements["element_types"], etypes);
        const Node &eindex = elements["element_index"];
        const Node &stream = elements["stream"];
        const Node &stream_ids = eindex["stream_ids"];
        const Node *stream_offs = eindex.fetch_ptr("offsets");
        const Node *stream_counts = eindex.fetch_ptr("element_counts");
        const index_t nstream = stream_ids.dtype().number_of_elements();
        const DataType sid_dtype(stream_ids.dtype().id(), 1);
        const DataType stream_dtype(stream.dtype().id(), 1);
        index_t ent_id = 0;
        // For count based this number just keeps rising, for offset based it gets overwritten
        //   by what is stored in the offsets node.
        index_t idx = 0;
        Node temp;
        for(index_t i = 0; i < nstream; i++)
        {
            // Determine which shape we are working with
            temp.set_external(sid_dtype, const_cast<void*>(stream_ids.element_ptr(i)));
            const index_t stream_id = temp.to_index_t();
            auto itr = std::find_if(etypes.begin(), etypes.end(), [=](const id_elem_pair &p){
                return p.first == stream_id;
            });
            entity &e = itr->second;

            // Determine how many elements are in this section of the stream
            index_t start = 0, end = 0;
            if(stream_offs)
            {
                const DataType dt(stream_offs->dtype().id(), 1);
                temp.set_external(dt, const_cast<void*>(stream_offs->element_ptr(i)));
                start = temp.to_index_t();
                if(i == nstream - 1)
                {
                    end = stream_offs->dtype().number_of_elements();
                }
                else
                {
                    temp.set_external(dt, const_cast<void*>(stream_offs->element_ptr(i+1)));
                    end = temp.to_index_t();
                }
            }
            else if(stream_counts)
            {
                const DataType dt(stream_counts->dtype().id(), 1);
                temp.set_external(dt, const_cast<void*>(stream_counts->element_ptr(i)));
                start = idx;
                end   = start + (temp.to_index_t() * e.shape.indices);
            }

            // Iterate the elements in this section
            idx = start;
            while(idx < end)
            {
                const index_t sz = e.shape.indices;
                for(index_t j = 0; j < sz; j++)
                {
                    temp.set_external(stream_dtype, const_cast<void*>(stream.element_ptr(idx)));
                    e.element_ids[j] = temp.to_index_t();
                    idx++;
                }
                e.entity_id = ent_id;
                func(e);
                ent_id++;
            }
            idx = end;
        }
        break;
    }
    default:
        CONDUIT_ERROR("Unsupported topology passed to iterate_elements")
        return;
    }
}

//-----------------------------------------------------------------------------
template<typename T, typename Func>
static void iterate_int_data_impl(const conduit::Node &node, Func &&func)
{
    conduit::DataArray<T> int_da = node.value();
    const index_t nele = int_da.number_of_elements();
    for(index_t i = 0; i < nele; i++)
    {
        func((index_t)int_da[i]);
    }
}

//-----------------------------------------------------------------------------
template<typename Func>
static void iterate_int_data(const conduit::Node &node, Func &&func)
{
    const auto id = node.dtype().id();
    switch(id)
    {
    case conduit::DataType::INT8_ID:
        iterate_int_data_impl<int8>(node, func);
        break;
    case conduit::DataType::INT16_ID:
        iterate_int_data_impl<int16>(node, func);
        break;
    case conduit::DataType::INT32_ID:
        iterate_int_data_impl<int32>(node, func);
        break;
    case conduit::DataType::INT64_ID:
        iterate_int_data_impl<int64>(node, func);
        break;
    case conduit::DataType::UINT8_ID:
        iterate_int_data_impl<uint8>(node, func);
        break;
    case conduit::DataType::UINT16_ID:
        iterate_int_data_impl<uint16>(node, func);
        break;
    case conduit::DataType::UINT32_ID:
        iterate_int_data_impl<uint32>(node, func);
        break;
    case conduit::DataType::UINT64_ID:
        iterate_int_data_impl<uint64>(node, func);
        break;
    default:
        CONDUIT_ERROR("Tried to iterate " << conduit::DataType::id_to_name(id) << " as integer data!");
        break;
    }
}

//-----------------------------------------------------------------------------
static void
append_remapped_ids(const Node &connectivity, const DataArray<index_t> map, std::vector<index_t> &out)
{
    iterate_int_data(connectivity, [&](index_t id) {
        out.push_back(map[id]);
    });
}

//-----------------------------------------------------------------------------
static void
build_unstructured_output(const std::vector<const Node*> &topologies,
                          const Node &pointmaps,
                          const std::string &cset_name,
                          Node &output)
{
    std::cout << "Building unstructured output!" << std::endl;
    output.reset();
    output["type"].set("unstructured");
    output["coordset"].set(cset_name);
    std::vector<std::string>          shape_types;
    std::vector<std::vector<index_t>> out_connectivity;
    const index_t ntopos = (index_t)topologies.size();
    for(index_t i = 0; i < ntopos; i++)
    {
        const Node *topo = topologies[i];
        const Node *elements = topo->fetch_ptr("elements");
        if(!elements)
        {
            // ERROR
            continue;
        }

        const Node *pointmap = pointmaps.child_ptr(i);
        if(!pointmap)
        {
            // ERROR
            continue;
        }
        DataArray<index_t> pmap_da = pointmap->value();
#if 1
        iterate_elements(*topo, [&](const entity &e) {
            // See if we already have a bucket for this shape in our output
            const std::string &shape_string = e.shape.type;
            // Find the index for this shape's bucket
            const auto itr = std::find(shape_types.begin(), shape_types.end(), shape_string);
            index_t idx = (index_t)(itr - shape_types.begin());
            if(itr == shape_types.end())
            {
                idx = shape_types.size();
                shape_types.push_back(shape_string);
                out_connectivity.emplace_back();
            }

            // Translate the point ids using the pointmap.
            std::vector<index_t> &out_conn = out_connectivity[idx];
            for(const index_t id : e.element_ids)
            {
                out_conn.push_back(pmap_da[id]);
            }
        });
#else
        // Build a vector of all the "elements" buckets
        std::vector<const Node*> elements_vec;

        // Single shape topology
        if(elements->has_child("shape"))
        {
            elements_vec.push_back(elements);
        }
        else if(elements->dtype().is_list()
                || elements->dtype().is_object())
        {
            // It is a collection of single element topologies
            // Q: Should we preserve the names when they exist?
            auto itr = elements->children();
            while(itr.has_next())
            {
                elements_vec.push_back(&itr.next());
            }
        }

        // Iterate the buckets of elements
        for(const Node *bucket : elements_vec)
        {
            const Node *shape = bucket->fetch_ptr("shape");
            const Node *connectivity = bucket->fetch_ptr("connectivity");
            if(!shape || !connectivity)
            {
                // ERROR!
                continue;
            }

            // See if we already have a bucket for this shape in our output
            const std::string &shape_string = shape->as_string();
            // Find the index for this shape's bucket
            const auto itr = std::find(shape_types.begin(), shape_types.end(), shape_string);
            index_t idx = (index_t)(itr - shape_types.begin());
            if(itr == shape_types.end())
            {
                idx = shape_types.size();
                shape_types.push_back(shape_string);
                out_connectivity.emplace_back();
            }

            // Translate the point ids using the pointmap.
            std::vector<index_t> &out_conn = out_connectivity[idx];
            append_remapped_ids(*connectivity, pmap_da, out_conn);
        }
#endif
    }

    if(shape_types.size() == 1)
    {
        output["elements/shape"].set(shape_types[0]);
        output["elements/connectivity"].set(out_connectivity[0]);
    }
    else if(shape_types.size() > 1)
    {
        const index_t nshapes = (index_t)shape_types.size();
        for(index_t i = 0; i < nshapes; i++)
        {
            const std::string name = shape_types[i] + "s";
            Node &bucket = output["elements"].add_child(name);
            bucket["shape"].set(shape_types[i]);
            bucket["connectivity"].set(out_connectivity[i]);
        }
    }
}

//-----------------------------------------------------------------------------
static void
build_polygonal_output(const std::vector<const Node*> &topologies,
                       const Node &pointmaps,
                       const std::string &cset_name,
                       Node &output)
{
    std::cout << "Building polygonal output!" << std::endl;
    output["type"].set("unstructured");
    output["coordset"].set(cset_name);
    output["elements/shape"].set("polygonal");

    std::vector<index_t> out_offsets;
    std::vector<index_t> out_conn;
    std::vector<index_t> out_sizes;
    const index_t ntopos = (index_t)topologies.size();
    for(index_t i = 0; i < ntopos; i++)
    {
        const Node &topo = *topologies[i];
        const Node *pointmap = pointmaps.child_ptr(i);
        if(!pointmap)
        {
            CONDUIT_WARN("Could not merge topology " << i <<  ", no associated pointmap.");
            continue;
        }
        DataArray<index_t> pmap_da = pointmap->value();

        iterate_elements(topo, [&out_offsets, &out_conn, &out_sizes, &pmap_da](const entity &e) {
            // If it is a polygon or 2D/1D the remapping is trivial
            if(e.shape.is_polygonal() || e.shape.dim == 2 || e.shape.dim == 1)
            {
                const index_t sz = (index_t)e.element_ids.size();
                out_offsets.push_back(out_conn.size());
                out_sizes.push_back(sz);
                for(index_t j = 0; j < sz; j++)
                {
                    out_conn.push_back(pmap_da[e.element_ids[j]]);
                }
            }
            else if(e.shape.is_polyhedral())
            {
                // Q: Will this even be a case? I'd imagine if polyhedra are present then
                //  everything would need to be promoted to polyhedra
            }
            else if(e.shape.id == (index_t)ShapeId::Tet || e.shape.id == (index_t)ShapeId::Hex)
            {
                // Q: Will this even be a case? I'd imagine if 3D elements exist then the output
                //  would have to be a 3D topology
                // TODO: If this case needs to exist then we need to track original cells
                const index_t embeded_sz = mesh::utils::TOPO_SHAPE_INDEX_COUNTS[e.shape.embed_id];
                index_t ei = 0;
                for(index_t j = 0; j < e.shape.embed_count; j++)
                {
                    out_offsets.push_back(out_conn.size());
                    out_sizes.push_back(embeded_sz);
                    for(index_t k = 0; k < embeded_sz; k++)
                    {
                        // Use the embedding table to find the correct index into the element_ids
                        //  array. Then map the element_id through the pointmap to get the new id.
                        out_conn.push_back(pmap_da[e.element_ids[e.shape.embedding[ei]]]);
                        ei++;
                    }
                }
            }
            else // (e.shape.dim == 0 || !e.shape.is_valid()) Q: Any other cases caught in here?
            {
                CONDUIT_ERROR("Encountered invalid element! At element " << e.entity_id);
                return;
            }
        });
    }
    output["elements/sizes"].set(out_sizes);
    output["elements/offsets"].set(out_offsets);
    output["elements/connectivity"].set(out_conn);
}

//-----------------------------------------------------------------------------
static void
build_polyhedral_output(const std::vector<const Node*> &topologies,
                       const Node &pointmaps,
                       const std::string &cset_name,
                       Node &output)
{
    std::cout << "Building polyhedra output!" << std::endl;
    output.reset();
    output["type"].set("unstructured");
    output["coordset"].set(cset_name);
    output["elements/shape"].set("polyhedral");
    output["subelements/shape"].set("polygonal");
    std::vector<index_t> out_conn;
    std::vector<index_t> out_sizes;
    std::vector<index_t> out_offsets;
    std::vector<index_t> out_subconn;
    std::vector<index_t> out_subsizes;
    std::vector<index_t> out_suboffsets;
    const index_t ntopos = (index_t)topologies.size();
    for(index_t i = 0; i < ntopos; i++)
    {
        const Node &topo = *topologies[i];
        const Node *pointmap = pointmaps.child_ptr(i);
        if(!pointmap)
        {
            CONDUIT_WARN("Could not merge topology " << i <<  ", no associated pointmap.");
            continue;
        }
        DataArray<index_t> pmap_da = pointmap->value();

        iterate_elements(topo, [&](const entity &e) {
            if(e.shape.is_polyhedral())
            {
                // Copy the subelements to their new offsets
                const index_t sz = e.element_ids.size();
                const index_t offset = out_conn.size();
                out_offsets.push_back(offset);
                out_sizes.push_back(sz);
                for(index_t i = 0; i < sz; i++)
                {
                    const index_t subidx = out_subsizes.size();
                    const index_t subsz = e.subelement_ids[i].size();
                    const index_t suboffset = out_subconn.size();
                    out_conn.push_back(subidx);
                    out_suboffsets.push_back(suboffset);
                    out_subsizes.push_back(subsz);
                    for(index_t j = 0; j < subsz; j++)
                    {
                        out_subconn.push_back(pmap_da[e.subelement_ids[i][j]]);
                    }
                }
            }
            else if(utils::TOPO_SHAPE_IDS[e.shape.id] == "f" || utils::TOPO_SHAPE_IDS[e.shape.id] == "l")
            {
                // 2D faces / 1D lines: Line, Tri, Quad
                // Add the shape to the sub elements
                const index_t subidx = out_subsizes.size();
                const index_t suboffset = out_subconn.size();
                out_suboffsets.push_back(suboffset);
                const index_t subsz = e.element_ids.size();
                out_subsizes.push_back(subsz);
                for(index_t j = 0; j < subsz; j++)
                {
                    out_subconn.push_back(pmap_da[e.element_ids[j]]);
                }
                // Then create a polyhedron with only 1 subelement
                out_offsets.push_back(out_conn.size());
                out_sizes.push_back(1);
                out_conn.push_back(subidx);
            }
            else if(utils::TOPO_SHAPE_IDS[e.shape.id] == "c")
            {
                // 3D cells: Tet, Hex
                const index_t nfaces = e.shape.embed_count;
                const index_t embeded_sz = mesh::utils::TOPO_SHAPE_INDEX_COUNTS[e.shape.embed_id];
                out_offsets.push_back(out_conn.size());
                out_sizes.push_back(nfaces);
                index_t ei = 0;
                for(index_t j = 0; j < nfaces; j++)
                {
                    out_conn.push_back(out_subsizes.size());
                    out_suboffsets.push_back(out_subconn.size());
                    out_subsizes.push_back(embeded_sz);
                    for(index_t k = 0; k < embeded_sz; k++)
                    {
                        // Use the embedding table to find the correct index into the element_ids
                        //  array. Then map the element_id through the pointmap to get the new id.
                        out_subconn.push_back(pmap_da[e.element_ids[e.shape.embedding[ei]]]);
                        ei++;
                    }
                }
            }
            else // (utils::TOPO_SHAPE_IDS[e.shape.id] == "p" || !e.shape.is_valid()) Q: Any other cases caught in here?
            {
                CONDUIT_ERROR("Encountered invalid element! At element " << e.entity_id);
                return;
            }
        });
    }
    output["elements/sizes"].set(out_sizes);
    output["elements/offsets"].set(out_offsets);
    output["elements/connectivity"].set(out_conn);
    output["subelements/sizes"].set(out_subsizes);
    output["subelements/offsets"].set(out_suboffsets);
    output["subelements/connectivity"].set(out_subconn);
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::topology --
//-----------------------------------------------------------------------------



//-------------------------------------------------------------------------
std::string
partitioner::recommended_topology(const std::vector<const Node *> &inputs) const
{
    // TODO: See if the inputs are uniform, rectilinear, etc and could be combined
    //       to form an output of one of those types. For example, uniform meshes
    //       can be combined if they abut and combine into larger bricks that
    //       cover space.

    // For now, recommend unstructured.
    return "unstructured";
}

//-------------------------------------------------------------------------
void
partitioner::combine(int domain,
    const std::vector<const Node *> &inputs,
    Node &output)
{
    // NOTE: Some decisions upstream, for the time being, make all the chunks
    //       unstructured. We will try to relax that so we might end up
    //       trying to combine multiple uniform,rectilinear,structured
    //       topologies.

    std::string rt(recommended_topology(inputs));
    if(rt == "uniform" || rt == "rectilinear")
        combine_as_structured(domain, inputs, output);
    else
        combine_as_unstructured(domain, inputs, output);
}

//-------------------------------------------------------------------------
void
partitioner::combine_as_structured(int domain,
    const std::vector<const Node *> &inputs,
    Node &output)
{
    // TODO: Make combined coordset and new structured topology (uniform,
    //       rectilinear, structured) suitable for the output.

    // Add the combined result to output node.
}

//-------------------------------------------------------------------------
void
partitioner::combine_as_unstructured(int domain,
    const std::vector<const Node *> &inputs,
    Node &output)
{
    // Determine names of all coordsets
    std::vector<std::string> coordset_names;
    Node &output_coordsets = output.add_child("coordsets");
    {
        // Group all the like-named coordsets
        std::vector<std::vector<const Node*>> coordset_groups;
        const index_t ninputs = (index_t)inputs.size();
        for(index_t i = 0; i < ninputs; i++)
        {
            const Node *input = inputs[i];
            if(!input) { continue; }

            const Node *csets = input->fetch_ptr("coordsets");
            if(!csets) { continue; }

            const auto &cset_names = csets->child_names();
            const index_t ncset_names = (index_t)cset_names.size();
            for(index_t j = 0; j < ncset_names; j++)
            {
                const auto &cset_name = cset_names[j];
                const Node *cset = csets->fetch_ptr(cset_name);
                if(!cset) { continue; }

                auto itr = std::find(coordset_names.begin(), coordset_names.end(), cset_name);
                if(itr != coordset_names.end())
                {
                    const auto idx = itr - coordset_names.begin();
                    coordset_groups[idx].push_back(cset);
                }
                else // (itr == names.end())
                {
                    const auto idx = coordset_groups.size();
                    coordset_names.push_back(cset_name);
                    coordset_groups.emplace_back();
                    coordset_groups[idx].push_back(cset);
                }
            }
        }

        const index_t ngroups = (index_t)coordset_groups.size();
#if 0
        // Some debug output to see how coordsets were grouped
        std::cout << "Coordsets:\n";
        for(index_t i = 0; i < ngroups; i++)
        {
            std::cout << "  -\n"
                << "    name: " << coordset_names[i] << "\n"
                << "    size: " << coordset_groups[i].size() << "\n";
        }
        std::cout << std::endl;
#endif
        for(index_t i = 0; i < ngroups; i++)
        {
            const auto &coordset_group = coordset_groups[i];
            coordset::combine(coordset_group, output_coordsets.add_child(coordset_names[i]));
        }
    }

    // Combine mapping information stored in chunks to assemble new field
    // that indicates original domain,pointid values for each point

    // Determine names of all topologies

    // Iterate over all topology names and combine like-named topologies
    // as new unstructured topology.
    Node &output_topologies = output.add_child("topologies");
    {
        // Group all the like-named toplogies
        std::vector<std::string> names;
        std::vector<std::vector<const Node*>> topo_groups;
        const index_t ninputs = (index_t)inputs.size();
        for(index_t i = 0; i < ninputs; i++)
        {
            const Node *input = inputs[i];
            if(!input) { continue; }

            const Node *topos = input->fetch_ptr("topologies");
            if(!topos) { continue; }

            const auto &topo_names = topos->child_names();
            const index_t ntopo_names = (index_t)topo_names.size();
            for(index_t j = 0; j < ntopo_names; j++)
            {
                const auto &topo_name = topo_names[j];
                const Node *topo = topos->fetch_ptr(topo_name);
                if(!topo) { continue; }

                auto itr = std::find(names.begin(), names.end(), topo_name);
                if(itr != names.end())
                {
                    const auto idx = itr - names.begin();
                    // Need to check if the topologies in this group have the same coordset
                    const Node *group_cset = topo_groups[idx][0]->fetch_ptr("coordset");
                    if(!group_cset) { continue; }
                    const Node *this_cset = topo->fetch_ptr("coordset");
                    if(!this_cset) { continue; }
                    if(group_cset->as_string() != this_cset->as_string())
                    {
                        // Error! Cannot merge two topologies that reference different named coordsets
                        continue;
                    }
                    topo_groups[idx].push_back(topo);
                }
                else // (itr == names.end())
                {
                    const auto idx = topo_groups.size();
                    names.push_back(topo_name);
                    topo_groups.emplace_back();
                    topo_groups[idx].push_back(topo);
                }
            }
        }

        const index_t ngroups = topo_groups.size();
        for(index_t i = 0; i < ngroups; i++)
        {
            const auto &topo_group = topo_groups[i];
            // All topologies in the same group must reference a coordset with the same name
            const Node *group_cset = topo_group[0]->fetch_ptr("coordset");
            if(!group_cset) { continue; }

            auto itr = std::find(coordset_names.begin(), coordset_names.end(), group_cset->as_string());
            if(itr == coordset_names.end())
            {
                // Error invalid coordset name!
                continue;
            }

            const Node *pointmaps = output_coordsets[*itr].fetch_ptr("pointmaps");
            if(!pointmaps) { continue; }

            topology::combine(topo_group, *pointmaps, output_topologies.add_child(names[i]));
        }
    }

    // Combine mapping info stored in chunks to assemble new field that
    // indicates original domain,cellid values for each cell.

    // Use original point and cell maps to create new fields that combine
    // the fields from each source chunk.

    // Add the combined result to output node.
}

//-------------------------------------------------------------------------
/**
 @brief This class accepts a set of input meshes and repartitions them
        according to input options. This class subclasses the partitioner
        class to add some parallel functionality.
 */
#if 0
class parallel_partitioner : public partitioner
{
public:
    parallel_partitioner(MPI_Comm c);
    virtual ~parallel_partitioner();

    virtual long get_total_selections() const override;

    virtual void get_largest_selection(int &sel_rank, int &sel_index) const override;

protected:
    virtual void map_chunks(const std::vector<chunk> &chunks,
                            std::vector<int> &dest_ranks,
                            std::vector<int> &dest_domain) override;

    virtual void communicate_chunks(const std::vector<chunk> &chunks,
                                    const std::vector<int> &dest_rank,
                                    const std::vector<int> &dest_domain,
                                    std::vector<chunk> &chunks_to_assemble,
                                    std::vector<int> &chunks_to_assemble_domains) override;

private:
    MPI_Comm comm;
};

//---------------------------------------------------------------------------
parallel_partitioner::~parallel_partitioner(MPI_Comm c) : partitioner()
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
    long ntotal_selections = nparts;
    MPI_Allreduce(&nselections, 1, MPI_LONG,
                  &ntotal_selections, 1, MPI_LONG, MPI_SUM, comm);

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
    MPI_Allreduce(&largest_selection_size, 1, MPI_LONG,
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
    MPI_Allreduce(&rank_that_matches, 1, MPI_INT,
                  &largest_rank_that_matches, 1, MPI_INT,
                  MPI_MAX, comm);

    sel_rank = largest_rank_that_matches;
    if(sel_rank == rank)
        sel_index = local_index;
}

//-------------------------------------------------------------------------
void
parallel_partitioner::map_chunks(const std::vector<chunk> &chunks,
    std::vector<int> &dest_ranks,
    std::vector<int> &dest_domain)
{
    // TODO: populate dest_ranks, dest_domain
}

//-------------------------------------------------------------------------
void
parallel_partitioner::communicate_chunks(const std::vector<chunk> &chunks,
    const std::vector<int> &dest_rank,
    const std::vector<int> &dest_domain,
    std::vector<chunk> &chunks_to_assemble,
    std::vector<int> &chunks_to_assemble_domains)
{
    // TODO: send chunks to dest_rank if dest_rank[i] != rank.
    //       If dest_rank[i] == rank then the chunk stays on the rank.
    //
    //       Do sends/recvs to send the chunks as blobs among ranks.
    //
    //       Populate chunks_to_assemble, chunks_to_assemble_domains
}

#endif

//-------------------------------------------------------------------------
void
partition(const conduit::Node &n_mesh, const conduit::Node &options,
    conduit::Node &output)
{
    partitioner P;
    if(P.initialize(n_mesh, options))
    {
        P.split_selections();
        output.reset();
        P.execute(output);
    }
}

namespace coordset
{

void CONDUIT_BLUEPRINT_API combine(const std::vector<const conduit::Node *> &coordsets,
                                 conduit::Node &output,
                                 double tolerance)
{
    point_merge pm;
    pm.execute(coordsets, tolerance, output);
}

}

namespace topology
{

void CONDUIT_BLUEPRINT_API combine(const std::vector<const conduit::Node *> &topologies,
                                   const conduit::Node &pointmaps,
                                   conduit::Node &output,
                                   conduit::Node *options)
{
    if(topologies.size() == 0)
    {
        return;
    }

    bool force_polyhedral = false;
    bool force_polygonal = false;
    if(options)
    {
        if(options->has_child("force_polyhedral"))
        {
            force_polyhedral = true;
        }

        if(options->has_child("force_polygonal"))
        {
            force_polygonal = true;
        }
    }

    const std::string &cset_name = topologies[0]->child("coordset").as_string();
    std::vector<const Node*> working_topologies;
    std::vector<Node> temporary_nodes;
    temporary_nodes.reserve(topologies.size());

    // Validate / translate inputs
    {
        const index_t ntopos = topologies.size();
        for(index_t i = 0; i < ntopos; i++)
        {
            const Node *topo = topologies[i];
            Node temp;
            if(!topo) { continue; }

            const Node *type = topo->fetch_ptr("type");
            if(!type) { continue; }

            const Node *cset = topo->fetch_ptr("coordset");
            if(!cset) { continue; }

            const std::string &t = type->as_string();
            if(t == "points")
            {
                temporary_nodes.emplace_back();
                // topology::points::to_explicit(topo, temporary_nodes.back());
                working_topologies.push_back(&temporary_nodes.back());
            }
            else if(t == "uniform")
            {
                temporary_nodes.emplace_back();
                topology::uniform::to_unstructured(*topo, temporary_nodes.back(), temp);
                working_topologies.push_back(&temporary_nodes.back());
            }
            else if(t == "rectilinear")
            {
                temporary_nodes.emplace_back();
                topology::rectilinear::to_unstructured(*topo, temporary_nodes.back(), temp);
                working_topologies.push_back(&temporary_nodes.back());
            }
            else if(t == "structured")
            {
                temporary_nodes.emplace_back();
                topology::structured::to_unstructured(*topo, temporary_nodes.back(), temp);
                working_topologies.push_back(&temporary_nodes.back());
            }
            else if(t == "unstructured")
            {
                working_topologies.push_back(topo);
            }
            else
            {
                //  ERROR!
                continue;
            }
        }

        // Make sure we have the correct number of point maps
        if(pointmaps.number_of_children() != (index_t)working_topologies.size())
        {
            std::cerr << "ERROR: Number of input pointmaps and number of input topologies do not match!" << std::endl;
            return;
        }
    }

    // Start building the output
    output.reset();
    if(working_topologies.size())
    {
        bool do_polyhedral = force_polyhedral;
        bool do_polygonal  = force_polygonal;
        index_t dim = 0;
        for(const Node *topo : topologies)
        {
            const Node *shape = topo->fetch_ptr("elements/shape");
            if(!shape) { continue; }
            const utils::ShapeType s(shape->as_string());
            if(s.is_polyhedral())
            {
                do_polyhedral = true;
            }
            else if(s.is_polygonal())
            {
                do_polygonal = true;
            }
            dim = std::max(dim, s.dim);
            if(do_polyhedral) break;
        }

        if(do_polyhedral || (do_polygonal && dim > 2))
        {
            build_polyhedral_output(working_topologies, pointmaps, cset_name, output);
        }
        else if(do_polygonal)
        {
            build_polygonal_output(working_topologies, pointmaps, cset_name, output);
        }
        else // Single shape & multi shape topologies
        {
            build_unstructured_output(working_topologies, pointmaps, cset_name, output);
        }
    }
}

}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
