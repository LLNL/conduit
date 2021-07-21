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

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mcarray.hpp"
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_blueprint_mesh.hpp"
#include "conduit_log.hpp"

//#ifdef CONDUIT_PARALLEL_PARTITION
//#include <mpi.h>
//#endif

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
    // TODO:

    // Determine names of all coordsets

    // Iterate over all coordsets and combine like-named coordsets into
    // new explicit coordset. Pass all points through an algorithm that
    // can combine the same points should they exist in multiple coordsets.
    //
    // for each coordset
    //     for each point in coordset
    //         new_pt_id = pointmap.get_id(point, tolerance)
    //              

    // Combine mapping information stored in chunks to assemble new field
    // that indicates original domain,pointid values for each point

    // Determine names of all topologies

    // Iterate over all topology names and combine like-named topologies
    // as new unstructured topology.

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
