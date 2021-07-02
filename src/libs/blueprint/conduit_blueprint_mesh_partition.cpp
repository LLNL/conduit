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
  root: 0
  mpi_comm: 11223


# Should we have separate lists of cell ids per topology?
selections:
   explicit:
      -
       domain: 0
       topologies:
          volume:
              indices: [0,1,2,3,4,5,6,7,8,9]
          faces:
              indices: [100,101,102,103]

Or, should we combine them but preserve the topology structure in the output?
(I like this one...)
selections:
   explicit:
      -
       domain: 0
       indices: [0,1,2,3,4,5,6,7,8,9,100,101,102,103]


# The target is how many meshes are desired in the output
target: 1
# Whether to include information that indicates how the output
# is related to the input.
mapping: true
# Point merging options
merging:
    enabled: true
    tolerance: 0.0001
# Parallel options
replicate: true
root: 0

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

size_t
determine_highest_topology(const std::vector<const conduit::Node *> &domains)
{
    size_t retval = 0; // points
    for(size_t i = 0; i < domains.size(); i++)
    {
        auto mt = domains[i]["topologies/mesh/type"].as_string();
        for(size_t j = 1; j < utils::TOPO_TYPES.size(); j++)
        {
            if(utils::TOPO_TYPES[j] == mt && j > retval)
            {
                retval = j;
                break;
            }
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Base class for selections that identify regions of interest that will
        be extracted from a mesh.
 */
class selection
{
public:
    selection() : n_options_ptr(nullptr) { }
    virtual ~selection() { }

    virtual bool init(const conduit::Node *n_opt_ptr) = 0;

    /**
     @brief Determines whether the selection can be applied to the supplied mesh.
     @param n_mesh A Conduit conduit::Node containing the mesh.
     @return True if the selection can be applied to the mesh; False otherwise.
     */
    virtual bool applicable(const conduit::Node &n_mesh) = 0;

    /**
     @brief Return the number of cells in the selection.
     @return The number of cells in the selection.
     */
    virtual index_t length() const { return 0; }

    /**
     @brief Partitions the selection into smaller selections.
     @param n_mesh A Conduit conduit::Node containing the mesh.
     @return A vector of selection pointers that cover the input selection.
     */
    virtual std::vector<std::shared_ptr<selection> > partition(const conduit::Node &n_mesh) const = 0;

    /**
     @brief Return the domain index to which the selection is being applied.
     @return The domain index or 0. This value is 0 by default.
     */
    index_t get_domain() const
    {
        index_t n = 0;
        if(n_options_ptr && n_options_ptr->has_child(DOMAIN_KEY))
            n = n_options_ptr->child(DOMAIN_KEY).to_index_t();
        return n;
    }

    /**
     @brief Return whether element and vertex mapping will be preserved in the output.
     @return True if mapping information is preserved, false otherwise.
     */
    bool preserve_mapping() const
    {
        bool mapping = true;
        if(n_options_ptr && n_options_ptr->has_child(MAPPING_KEY))
            mapping = n_options_ptr->child(MAPPING_KEY).as_bool();
        return mapping;
    }

    /**
     @brief Returns the cells in this selection that are contained in the
            supplied topology. Such cells will have cell ranges in erange,
            inclusive. The element ids are returned in element_ids.
     */
    virtual void get_element_ids_for_topo(const conduit::Node &n_topo,
                                          const index_t erange[2],
                                          std::vector<index_t> &element_ids) const = 0;

protected:
    static const std::string DOMAIN_KEY;
    static const std::string MAPPING_KEY;

    const conduit::Node *n_options_ptr;
};

const std::string selection::DOMAIN_KEY("domain");
const std::string selection::MAPPING_KEY("mapping");


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

    // Initializes the selection from a conduit::Node.
    virtual bool init(const conduit::Node *n_opt_ptr) override;

    virtual bool applicable(const conduit::Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length() const override
    {
        return cells_for_axis(0) * 
               cells_for_axis(1) * 
               cells_for_axis(2);
    }

    virtual std::vector<std::shared_ptr<selection> > partition(const conduit::Node &n_mesh) const override;

    virtual void extract(const conduit::Node &n_mesh, conduit::Node &n_output) const override;

    void set_start(index_t s0, index_t s1, index_t s2)
    {
        start[0] = s0;
        start[1] = s1;
        start[2] = s2;
    }

    void set_end(index_t e0, index_t e1, index_t e2)
    {
        end[0] = e0;
        end[1] = e1;
        end[2] = e2;
    }

    virtual void get_element_ids_for_topo(const conduit::Node &n_topo,
                                          const index_t erange[2],
                                          std::vector<index_t> &element_ids) const override;
private:
    index_t cells_for_axis(int axis) const
    {
        return (axis >= 0 && axis <= 2) ? std::max(end[axis] - start[axis] + 1, 1) : 0;
    }

    index_t start[3];
    index_t end[3];
};

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
selection_logical::init(const conduit::Node *n_opt_ptr)
{
    bool ok = false;
    n_options_ptr = n_opt_ptr;
    if(n_options_ptr->has_child("start") && n_options_ptr->has_child("end"))
    {
        unsigned_int_array s = n_options_ptr->child("start").value();
        unsigned_int_array e = n_options_ptr->child("end").value();
        if(s.number_of_elements() == 3 &&
           e.number_of_elements() == 3)
        {
            for(int i = 0; i < 3; i++)
            {
                start[i] = static_cast<index_t>(s[i]);
                end[i] = static_cast<index_t>(e[i]);
            }
            ok = true;
        }
    }
    return ok;
}

//---------------------------------------------------------------------------
/**
 @brief Returns whether the logical selection applies to the input mesh.
 */
bool
selection_logical::applicable(const conduit::Node &n_mesh)
{
    bool retval = false;

    const conduit::Node &n_coords = n_mesh["coordsets"][0];
    const conduit::Node &n_topo = n_mesh["topologies"][0];
    bool is_uniform = n_coords["type"].as_string() == "uniform";
    bool is_rectilinear = n_coords["type"].as_string() == "rectilinear";
    bool is_structured = n_coords["type"].as_string() == "explicit" && 
                         n_topo["type"].as_string() == "structured");
    if(is_uniform || is_rectilinear || is_structured)
    {
        index_t dims[3] = {1,1,1};
        const conduit::Node &n_topo = n_mesh["topologies"][0];
        topology::logical_dims(n_topo, dims, 3);

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

    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Partitions along the longest axis and returns a vector containing 2
        logical selections.
 */
std::vector<std::shared_ptr<selection> >
selection_logical::partition(const conduit::Node &n_mesh) const
{
    int la = 0;
    if(cells_for_axis(1) > cells_for_axis(longest_axis))
        la = 1;
    if(cells_for_axis(2) > cells_for_axis(longest_axis))
        la = 2;
    auto n = cells_for_axis(la);

    auto p0 = std::shared_ptr<selection_logical>();
    auto p1 = std::shared_ptr<selection_logical>();
    if(la == 0)
    {
        p0->set_start(start[0],       start[1],       start[2]);
        p0->set_end(start[0]+n/2,     end[1],         end[2]);
        p1->set_start(start[0]+n/2+1, start[1],       start[2]);
        p1->set_end(end[0],           end[1],         end[2]);
    }
    else if(la == 1)
    {
        p0->set_start(start[0],       start[1],       start[2]);
        p0->set_end(start[0],         end[1]+n/2,     end[2]);
        p1->set_start(start[0],       start[1]+n/2+1, start[2]);
        p1->set_end(end[0],           end[1],         end[2]);
    }
    else
    {
        p0->set_start(start[0],       start[1],       start[2]);
        p0->set_end(start[0],         end[1],         end[2]+n/2);
        p1->set_start(start[0],       start[1],       start[2]+n/2+1);
        p1->set_end(end[0],           end[1],         end[2]);
    }

    std::vector<std::shared_ptr<selection> > parts;
    parts.push_back(p0);
    parts.push_back(p1);

    return parts;
}

//---------------------------------------------------------------------------
/*void
selection_logical::get_vertex_ids(const conduit::Node &n_mesh,
    std::vector<index_t> &ids) const
{
    index_t dims[3] = {1,1,1};
    const conduit::Node &n_topo = n_mesh["topologies"][0];
    topology::logical_dims(n_topo, dims, 3);

    ids.clear();
    ids.reserve(dims[0] * dims[1] * dims[2]);
    auto mesh_NXNY = dims[0] * dims[1];
    auto mesh_NX   = dims[0];  
    index_t n_end[3];
    n_end[0] = end[0] + 1;
    n_end[1] = end[1] + 1;
    n_end[2] = end[2] + 1;
    for(index_t k = start[2]; k <= n_end[2]; k++)
    for(index_t j = start[1]; j <= n_end[1]; j++)
    for(index_t i = start[0]; i <= n_end[0]; i++)
    {
        ids.push_back(k*mesh_NXNY + j*mesh_NX + i);
    }
}
*/

//---------------------------------------------------------------------------
void
selection_logical::get_element_ids_for_topo(const conduit::Node &n_topo,
    const index_t erange[2], std::vector<index_t> &element_ids) const
{
    index_t dims[3] = {1,1,1};
    topology::logical_dims(n_topo, dims, 3);

    element_ids.clear();
    element_ids.reserve(length());
    auto mesh_CXCY = dims[0] * dims[1];
    auto mesh_CX   = dims[0];
    for(index_t k = start[2]; k <= end[2]; k++)
    for(index_t j = start[1]; j <= end[1]; j++)
    for(index_t i = start[0]; i <= end[0]; i++)
    {
        auto eid = k*mesh_CXCY + j*mesh_CX + i;
        if(eid >= erange[0] && eid <= erange[1])
            element_ids.push_back(eid);
    }
}

//---------------------------------------------------------------------------
/**
   @brief This selection explicitly defines which cells we're pulling out from
          a mesh, and in which order.
 */
class selection_explicit : public selection
{
public:
    selection_explicit() : selection(), ids_storage(),
        num_cells_in_selection(0), num_cells_in_mesh(0)
    {
    }

    virtual ~selection_explicit()
    {
    }

    virtual bool init(const conduit::Node *n_opt_ptr) override
    {
        bool ok = false;
        n_options_ptr = n_opt_ptr;
        if(n_options_ptr &&
           n_options_ptr->has_child(ELEMENTS_KEY) &&
           n_options_ptr->child(ELEMENTS_KEY).dtype().is_number())
        {
            // Convert to the right type for index_t
#ifdef CONDUIT_INDEX_32
            n_options_ptr->child(ELEMENTS_KEY).to_uint32_array(ids_storage);
#else
            n_options_ptr->child(ELEMENTS_KEY).to_uint64_array(ids_storage);
#endif
            ok = true;
        }
        return ok;
    }

    virtual bool applicable(const conduit::Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length() const override
    {
        return num_cells_in_selection;
    }

    virtual std::vector<std::shared_ptr<selection> > partition(const conduit::Node &n_mesh) const override;

    const index_t *get_indices() const
    {
        // Access the converted data as index_t.
        return reinterpret_cast<const index_t *>(ids_storage.data_ptr());
    }

    virtual void get_element_ids_for_topo(const conduit::Node &n_topo,
                                          const index_t erange[2],
                                          std::vector<index_t> &element_ids) const override;

private:
    static const std::string ELEMENTS_KEY;
    conduit::Node ids_storage;
    index_t num_cells_in_selection;
    index_t num_cells_in_mesh;
};

std::string selection_explicit::ELEMENTS_KEY("elements");

//---------------------------------------------------------------------------
/**
 @brief Returns whether the explicit selection applies to the input mesh.
 */
bool
selection_explicit::applicable(const conduit::Node &n_mesh)
{
    return true;
}

//---------------------------------------------------------------------------
std::vector<std::shared_ptr<selection> >
selection_explicit::partition(const conduit::Node &n_mesh) const
{
    auto num_cells_in_mesh = topology::length(n_mesh);
    auto n = ids_storage.dtype().number_of_elements();
    auto n_2 = n/2;
    auto indices = get_indices();
    std::vector<index_t> ids0, ids1;
    ids0.reserve(n_2);
    ids1.reserve(n_2);
    auto indices = get_indices();
    for(index_t i = 0; i < n; i++)
    {
        if(indices[i] < num_cells_in_mesh)
        {
            if(i < n_2)
                ids0.push_back(indices[i]);
            else
                ids1.push_back(indices[i]);
        }
    }

    auto p0 = std::make_shared<selection_explicit>();
    auto p1 = std::make_shared<selection_explicit>();
    p0->ids_storage.set(ids0);
    p0->num_cells_in_selection = ids0.size(); 
    p0->num_cells_in_mesh = num_cells_in_mesh;

    p1->ids_storage.set(ids1);
    p1->num_cells_in_selection = ids1.size(); 
    p1->num_cells_in_mesh = num_cells_in_mesh;

    std::vector<std::shared_ptr<selection> > parts;
    parts.push_back(p0);
    parts.push_back(p1);

    return parts;
}

//---------------------------------------------------------------------------
void
selection_explicit::get_element_ids_for_topo(const conduit::Node &n_topo,
    const index_t erange[2], std::vector<index_t> &element_ids) const
{
    auto n = ids_storage.dtype().number_of_elements();
    auto indices = get_indices();
    element_ids.reserve(n);
    for(index_t i = 0; i < n; i++)
    {
        auto eid = indices[i];
        if(eid >= erange[0] && eid <= erange[1])
            element_ids.push_back(eid);
    }
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
class selection_ranges : public selection
{
public:
    selection_ranges() : selection(), ranges_storage()
    {
    }

    virtual ~selection_ranges()
    {
    }

    virtual bool init(const conduit::Node *n_opt_ptr) override
    {
        bool ok = false;
        n_options_ptr = n_opt_ptr;
        if(n_options_ptr &&
           n_options_ptr->has_child(RANGES_KEY) &&
           n_options_ptr->child(RANGES_KEY).dtype().is_number())
        {
            // Convert to the right type for index_t
#ifdef CONDUIT_INDEX_32
            n_options_ptr->child(RANGES_KEY).to_uint32_array(ranges_storage);
#else
            n_options_ptr->child(RANGES_KEY).to_uint64_array(ranges_storage);
#endif
            ok = (ranges_storage.dtype().num_elements() % 2 == 0);
        }
        return ok;
    }

    virtual bool applicable(const conduit::Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length() const override;

    virtual std::vector<std::shared_ptr<selection> > partition(const conduit::Node &n_mesh) const override;

    virtual void get_element_ids_for_topo(const conduit::Node &n_topo,
                                          const index_t erange[2],
                                          std::vector<index_t> &element_ids) const override;

    const index_t *get_ranges() const
    {
        // Access the converted data as index_t.
        return reinterpret_cast<const index_t *>(ranges_storage.data_ptr());
    }

private:
    static const std::string RANGES_KEY;
    conduit::Node ranges_storage;
};

std::string selection_ranges::RANGES_KEY("ranges");

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
    auto n = ranges_storage.dtype().number_of_elements() / 2;
    for(index_t i = 0; i < n; i++)
    {
        ncells += ranges[2*i+1] - ranges[2*i] + 1;
    }
    return ncells;
}

//---------------------------------------------------------------------------
std::vector<std::shared_ptr<selection> >
selection_ranges::partition(const conduit::Node &n_mesh) const
{
    index_t ncells = length();
    auto ncells_2 = ncells / 2;
    auto n = ranges_storage.dtype().number_of_elements() / 2;
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
            auto rc = (ranges[2*i+1] - ranges[2*i] + 1;
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

    std::vector<std::shared_ptr<selection> > parts;
    parts.push_back(p0);
    parts.push_back(p1);

    return parts;
}

//---------------------------------------------------------------------------
void
selection_ranges::get_element_ids_for_topo(const conduit::Node &n_topo,
    const index_t erange[2], std::vector<index_t> &element_ids) const
{
    auto n = ids_storage.dtype().number_of_elements();
    auto n_2 = n / 2;
    auto indices = get_ranges();
    for(index_t i = 0; i < n_2; i++)
    {
        index_t start = indices[2*i];
        index_t end = indices[2*i+1];
        for(index_t eid = start; eid <= end; eid++)
        {
            if(eid >= erange[0] && eid <= erange[1])
                element_ids.push_back(eid);
        }
    }
}


//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
/**
 @brief This class can read a set of selections and apply them to a Conduit
        node containing single or multi-domain meshes and produce a new
        Conduit node that refashions the selections into a target number of
        mesh domains.
 */
class partitioner
{
public:
    partitioner();
    virtual ~partitioner();

    bool initialize(const conduit::Node &n_mesh, const conduit::Node &options);

    void split_selections(const conduit::Node &options);

    void execute(const conduit::Node &options, conduit::Node &output);

    virtual long get_total_selections() const;

    virtual void get_largest_selection(int &sel_rank, int &sel_index) const;

protected:
    std::shared_ptr<selection> create_selection(const conduit::Node &n_sel) const;
    void copy_fields(const conduit::Node &n_mesh, conduit::Node &output) const;
    void copy_field(Node &n_output_fields, const conduit::Node &n_field,
                    const std::vector<index_t> &ids) const;
    void slice_array(Node &n_dest_values, const conduit::Node &n_src_values,
                     const std::vector<index_t> &ids) const;

    int rank, size;
    unsigned int target;
    std::vector<Node *>                      meshes;
    std::vector<std::shared_ptr<selection> > selections;
};

//---------------------------------------------------------------------------
partitioner::partitioner() : rank(0), size(1), target(1), meshes(), selections()
{
}

//---------------------------------------------------------------------------
partitioner::~partitioner()
{
}

//---------------------------------------------------------------------------
std::shared_ptr<selection>
partitioner::create_selection(const conduit::Node &n_sel) const
{
    std::shared_ptr<selection> retval;
    if(n_sel["type"].as_string() == "logical")
        retval = std::make_shared<selection_logical>();
    else if(n_sel["type"].as_string() == "explicit")
        retval = std::make_shared<selection_explicit>();
    else if(n_sel["type"].as_string() == "ranges")
        retval = std::make_shared<selection_ranges>();
    return retval;
}

//---------------------------------------------------------------------------
bool
partitioner::initialize(const conduit::Node &n_mesh, const conduit::Node &options)
{
    auto doms = domains(mesh);

    // Iterate over the selections in the options and check them against the
    // domains that were passed in to make a vector of meshes and selections
    // that can be used to partition the meshes.
    if(options.has_child("selections"))
    {
        const conduit::Node &n_selections = options["selections"];
        for(size_t i = 0; i < n_selections.get_num_children(); i++)
        {
            const conduit::Node *n_sel = n_selections.child_ptr(i);
            auto sel = create_selection(*n_sel);
            if(sel != nullptr && sel->init(n_sel))
            {
                // The selection is good. See if it applies to the domains.

                for(size_t domid = 0; domid < doms.size(); domid++)
                {
                    // Q: What is the domain number for this domain?

                    if(domid == sel->get_domain() && sel->applicable(doms[domainid]))
                    {
                        meshes.push_back(doms[domid]);
                        selections.push_back(sel);
                        break;
                    }
                }
            }
        }
    }
    else
    {
        // TODO: Add a selection that means all of each domain.
        //       It ought to be an appropriate selection type for the mesh type.
    }

    // Get the number of target partitions that we're making.
    unsigned int target = 1;
    if(options.has_child("target"))
        target = options.as_unsigned_int();

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
partitioner::split_selections(const conduit::Node &options)
{
    int size = 1, rank = 0;
    long ntotal_selections = get_total_selections();

    // Splitting.
    while(target > ntotal_parts)
    {
        // Get the rank with the largest selection and get that local
        // selection index.
        int sel_rank = -1, sel_index = -1;
        get_largest_selection(sel_rank, sel_index);

        if(rank == sel_rank)
        {
            auto ps = selections[index]->partition(meshes[index]);

            if(!ps.empty())
            {
                const conduit::Node *m = meshes[index];
                meshes.insert(meshes.begin()+index, ps.size()-1, m);
                selections.insert(selections.begin()+index, ps.size()-1, nullptr);
                for(size_t i = 0; i < index; i++)
                    selections[index + i] = ps[i];
            }
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::copy_fields(const std::vector<index_t> &all_selected_vertex_ids,
    const std::vector<index_t> &all_selected_element_ids,
    const conduit::Node &n_mesh, conduit::Node &n_output,
    bool preserve_mapping) const
{
    if(n_mesh.has_child("fields"))
    {
        const conduit::Node &n_fields = n_mesh["fields"];
        if(!all_selected_vertex_ids.empty())
        {
            conduit::Node &n_output_fields = n_output["fields"];
            for(size_t i = 0; i < n_fields.number_of_children(); i++)
            {
                const conduit::Node &n_field = n_fields[i];
                if(n_field.has_child("association"))
                {
                    auto association = n_field["association"].as_string();
                    if(association == "vertex")
                    {
                        copy_field(n_field, all_selected_vertex_ids, n_output_fields);
                    }
                }
            }

            if(preserve_mapping)
            {
                // TODO: save the all_selected_vertex_ids as a new field.
            }
        }

        if(!all_selected_element_ids.empty())
        {
            conduit::Node &n_output_fields = n_output["fields"];
            for(size_t i = 0; i < n_fields.number_of_children(); i++)
            {
                const conduit::Node &n_field = n_fields[i];
                if(n_field.has_child("association"))
                {
                    auto association = n_field["association"].as_string();
                    if(association == "element")
                    {
                        copy_field(n_field, all_selected_element_ids, n_output_fields);
                    }
                }
            }

            if(preserve_mapping)
            {
                // TODO: save the all_selected_element_ids as a new field.
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
    if(n_values.dtype().is_compact()) 
    {
        slice_array(n_values, ids, n_new_field["values"]);
    }
    else
    {
        // otherwise, we need to compact our data first
        conduit::Node n;
        n_values.compact_to(n);
        slice_array(n, ids, n_new_field["values"]);
    }
}

//---------------------------------------------------------------------------
// @brief Slice the n_src array using the indices stored in ids. The 
//        destination memory is already pointed to by n_dest.
template <typename T>
inline T *
slice_array(const conduit::Node &n_src, const std::vector<index_t> &ids, Node &n_dest)
{
    const T *src = reinterpret_cast<const T *>(n_src.data_ptr());
    T *dest = reinterpret_cast<T *>(n_dest.data_ptr());
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
    n_dest_values = DataType(n_src_values.dtype().id(), ids.size());

    // Do the slice.
    if(dt.is_int8())
        slice_array<int8>(n_src_values, ids, n_dest_values);
    else if(dt.is_int16())
        slice_array<int16>(n_src_values, ids, n_dest_values);
    else if(dt.is_int32())
        slice_array<int32>(n_src_values, ids, n_dest_values);
    else if(dt.is_int64())
        slice_array<int64>(n_src_values, ids, n_dest_values);
    else if(dt.is_uint8())
        slice_array<uint8>(n_src_values, ids, n_dest_values);
    else if(dt.is_uint16())
        slice_array<uint16>(n_src_values, ids, n_dest_values);
    else if(dt.is_uint32())
        slice_array<uint32>(n_src_values, ids, n_dest_values);
    else if(dt.is_uint64())
        slice_array<uint64>(n_src_values, ids, n_dest_values);
    else if(dt.is_float32())
        slice_array<float32>(n_src_values, ids, n_dest_values);
    else if(dt.is_float64())
        slice_array<float64>(n_src_values, ids, n_dest_values);
    else if(dt.is_char())
        slice_array<char>(n_src_values, ids, n_dest_values);
    else if(dt.is_short())
        slice_array<short>(n_src_values, ids, n_dest_values);
    else if(dt.is_int())
        slice_array<int>(n_src_values, ids, n_dest_values);
    else if(dt.is_long())
        slice_array<long>(n_src_values, ids, n_dest_values);
    else if(dt.is_unsigned_char())
        slice_array<unsigned char>(n_src_values, ids, n_dest_values);
    else if(dt.is_unsigned_short())
        slice_array<unsigned short>(n_src_values, ids, n_dest_values);
    else if(dt.is_unsigned_int())
        slice_array<unsigned int>(n_src_values, ids, n_dest_values);
    else if(dt.is_unsigned_long())
        slice_array<unsigned long>(n_src_values, ids, n_dest_values);
    else if(dt.is_float())
        slice_array<float>(n_src_values, ids, n_dest_values);
    else if(dt.is_double())
        slice_array<double>(n_src_values, ids, n_dest_values);
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
        index_t edims[3] = {1,1,1};
        auto ndims = topology::dims(n_topo)
        topology::logical_dims(n_topo, edims, 3);

        index_t cell_ijk[3]={0,0,0}, pt_ijk[3] = {0,0,0};
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
        for(index_t i = 0 i < n; i++)
        {
            // Get the IJK coordinate of the element.
            grid_id_to_ijk(element_ids[i], edims, cell_ijk);

            // Turn the IJK into vertex ids.
            for(int i = 0; i < np; i++)
            {
                pt_ijk[0] = cell_ijk[0] + offset[i][0];
                pt_ijk[1] = cell_ijk[1] + offset[i][1];
                pt_ijk[2] = cell_ijk[2] + offset[i][2];
                grid_ijk_to_id(pt_ijk, dims, ptid);

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
        ShapeType shape(n_topo);
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
            auto nverts_in_shape = TOPO_SHAPE_INDEX_COUNTS[shape.id];
            for(size_t i = 0 i < element_ids.size(); i++)
            {
                auto elem_conn = iptr + element_ids[i] * nverts_in_shape;
                for(index_t j = 0; j < nverts_in_shape; j++)
                    vertex_ids.insert(elem_conn[j]);
            }
        }
    }
}

//---------------------------------------------------------------------------
conduit::Node *
partitioner::extract(size_t idx, const conduit::Node &n_mesh) const
{
    if(idx >= selections.size())
        return nullptr;

    const conduit::Node &n_mesh = *meshes[idx];
    const conduit::Node &n_topo = n_mesh["topologies"];
    const conduit::Node &n_coordsets = n_mesh["coordsets"];
    std::map<std::string, std::shared_ptr<std::vector<index_t>>> topo_element_ids;
    std::map<std::string, std::shared_ptr<std::set<index_t>>> coordset_vertex_ids;
    index_t erange[] = {0,0};
    for(size_t i = 0; i < n_topo.get_number_of_children(); i++)
    {
        // Get the current topology.
        const conduit::Node &this_topo = n_topo[i];

        // Get the number of elements in the topology.
        index_t topo_num_elements = topology::length(this_topo);
        erange[1] += topo_num_elements-1;

        // Create a vector to which we'll add the element ids for this topo.
        // We have to keep it around 
        auto eit = topo_element_ids.find(this_topo.name());
        if(eit == topo_element_ids.end())
        {
            topo_element_ids[csname] = std::make_shared<std::vector<index_t>>();
            eit = topo_element_ids.find(this_topo.name());
        }
        // Get the selected element ids that are in this topology.
        std::vector<index_t> &element_ids = eit->second;

// NOTE: we could pass back ranges for the element ids...
        // Get the selected element ids that are in this topology.
        selections[idx]->get_element_ids_for_topo(this_topo, erange, *eit);

        // What's its coordset name?
        std::string csname(this_topo["coordset"].name());
        auto vit = coordset_vertex_ids.find(csname);
        if(vit == coordset_vertex_ids.end())
        {
            coordset_vertex_ids[csname] = std::make_shared<std::set<index_t>>();
            vit = coordset_vertex_ids.find(csname);
        }

// NOTE: we could pass back ranges for the vertex ids...
        // Add the vertex ids for the elements in the selection. This lets us
        // build up a comprehensive set of the vertex ids we need from the
        // coordset, as determined by multiple topologies.
        get_vertex_ids_for_element_ids(this_topo, *eit, *vit)

        erange[0] += topo_num_elements;
    }

    // We now have vectors of element ids that we need to extract from each topo.
    // We also have sets of vertex ids that we need to extract from each coordset.

    conduit::Node *retval = new conduit::Node;
    conduit::Node &n_output = *retval;

// HEY, a concern I have about this way as opposed to making each selection do the
//      extract is that this way makes the output unstructured all the time. With
//      making the logical selection do the extraction, I can still output a logical
//      structured output for those topologies. This way can't really do that.

    // Create new coordsets that include the vertices that are relevant for the
    // selection.
    conduit::Node &n_new_coordsets = n_output["coordsets"];
    index_t vid = 0;
    std::vector<index_t> all_selected_vertex_ids;
    for(size_t i = 0; i < n_coordsets.get_number_of_children(); i++)
    {
        const conduit::Node &n_coordset = n_coordsets[i];
        auto vit = coordset_vertex_ids.find(n_coordset.name());

        // TODO: avoid copying back to std::vector multiple times.
        std::vector<index_t> vertex_ids;
        index_t_set_to_vector(vit->second, vertex_ids);

        // Build up a mapping of old to new vertices over all coordsets that we
        // can use to remap fields.
        for(auto it = vit->second.begin(); it != vit->second.end(); it++)
            all_selected_vertex_ids.push_back(*it);

        // Create the new coordset.
        create_new_explicit_coordset(n_coordset, vertex_ids, n_new_coordsets[n_coordset.name()]);
    }

    // Create new topologies containing the selected cells.
    conduit::Node &n_new_topos = n_output["topologies"];
    index_t eid = 0;
    std::vector<index_t> all_selected_element_ids;
    for(size_t i = 0; i < n_topo.get_number_of_children(); i++)
    {
        const conduit::Node &n_this_topo = n_topo[i];
        auto vit = topo_element_ids.find(n_this_topo.name());
        if(!eit->second.empty())
        {
            const conduit::Node &n_coordset = n_this_topo["coordset"];
            auto vit = coordset_vertex_ids.find(n_coordset.name());
            if(vit != coordset_vertex_ids.end())
            {
                // Build up a mapping of old to new elements over all topos
                // can use to remap fields.
                for(size_t j = 0; j < eit->second.size(); j++)
                    all_selected_element_ids.push_back(eit->second[j]);

                // TODO: avoid copying back to std::vector multiple times.
                std::vector<index_t> vertex_ids;
                index_t_set_to_vector(vit->second, vertex_ids);

                create_new_unstructured_topo(n_this_topo, *eit->second, n_new_topos[n_this_topo.name()]);
            }
        }
    }

    // Now that we've made new coordsets and topologies, make new fields.
    copy_fields(all_selected_vertex_ids, all_selected_element_ids,
                n_mesh, n_output,
                selections[idx]->preserve_mapping());

    return retval;
}

//---------------------------------------------------------------------------
void
partitioner::create_new_explicit_coordset(const conduit::Node &n_coordset,
    const std::vector<index_t> &vertex_ids, conduit::Node &n_new_coordset) const
{
    conduit::Node n_explicit;
    if(n_coordset["type"] == "uniform")
    {
        conduit::blueprint::coordset::uniform::to_explicit(n_coordset, n_explicit);

        auto axes = conduit::blueprint::coordset::axes(n_explicit);
        const conduit::Node &n_values = n_explicit["values"];
        conduit::Node &n_new_values = n_new_coordset["values"];
        for(size_t i = 0; i < axes.size(); i++)
        {
            const conduit::Node &n_axis_values = n_values[axes[i]];
            conduit::Node &n_new_axis_values = n_values[axes[i]];
            slice_array(n_axis_values, vertex_ids, n_new_axis_values);
        }
    }
    else if(n_coordset["type"] == "rectilinear")
    {
        conduit::blueprint::coordset::rectilinear::to_explicit(n_coordset, n_explicit);

        auto axes = conduit::blueprint::coordset::axes(n_explicit);
        const conduit::Node &n_values = n_explicit["values"];
        conduit::Node &n_new_values = n_new_coordset["values"];
        for(size_t i = 0; i < axes.size(); i++)
        {
            const conduit::Node &n_axis_values = n_values[axes[i]];
            conduit::Node &n_new_axis_values = n_values[axes[i]];
            slice_array(n_axis_values, vertex_ids, n_new_axis_values);
        }
    }
    else if(n_coordset["type"] == "explicit")
    {
        auto axes = conduit::blueprint::coordset::axes(n_coordset);
        const conduit::Node &n_values = n_coordset["values"];
        conduit::Node &n_new_values = n_new_coordset["values"];
        for(size_t i = 0; i < axes.size(); i++)
        {
            const conduit::Node &n_axis_values = n_values[axes[i]];
            conduit::Node &n_new_axis_values = n_values[axes[i]];
            slice_array(n_axis_values, vertex_ids, n_new_axis_values);
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::create_new_unstructured_topo(const conduit::Node &n_topo,
    const std::vector<index_t> &element_ids, const std::vector<index_t> &vertex_ids,
    conduit::Node &n_new_topo) const
{
    if(n_topo["type"].as_string() == "uniform")
    {
        conduit::Node n_uns, cdest; // what is cdest?
        conduit::blueprint::mesh::topology::uniform::to_unstructured(n_topo, n_uns, cdest);
        unstructured_topo_from_unstructured(n_uns, element_ids, vertex_ids, n_new_topo);
    }
    else if(n_topo["type"].as_string() == "rectilinear")
    {
        conduit::Node n_uns, cdest; // what is cdest?
        conduit::blueprint::mesh::topology::rectilinear::to_unstructured(n_topo, n_uns, cdest);
        unstructured_topo_from_unstructured(n_uns, element_ids, vertex_ids, n_new_topo);
    }
    else if(n_topo["type"].as_string() == "structured")
    {
        conduit::Node n_uns, cdest; // what is cdest?
        conduit::blueprint::mesh::topology::structured::to_unstructured(n_topo, n_uns, cdest);
        unstructured_topo_from_unstructured(n_uns, element_ids, vertex_ids, n_new_topo);
    }
    else if(n_topo["type"].as_string() == "unstructured")
    {
        unstructured_topo_from_unstructured(n_topo, element_ids, vertex_ids, n_new_topo);
    }
}

//---------------------------------------------------------------------------
void
partitioner::create_unstructured_topo_from_unstructured(const conduit::Node &n_topo,
    const std::vector<index_t> &element_ids, const std::vector<index_t> &vertex_ids,
    conduit::Node &n_new_topo) const
{
    n_new_topo["type"].set("unstructured");
    n_new_topo["coordset"].set(n_topo["coordset"]);
    conduit::Node &n_new_elements = n_new_topo["elements"];

    // vertex_ids contains the list of old vertex ids that our selection uses
    // from the old coordset. It can serve as a new to old map.

    std::map<index_t,index_t> old2new;
    for(index_t i = 0; i < vertex_ids.size(); i)
        old2new[vertex_ids[i]] = i;

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
    ShapeType shape(n_topo);
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
        auto nverts_in_shape = TOPO_SHAPE_INDEX_COUNTS[shape.id];
        for(size_t i = 0 i < element_ids.size(); i++)
        {
            auto elem_conn = iptr + element_ids[i] * nverts_in_shape;
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
partitioner::execute(const conduit::Node &options, conduit::Node &output)
{
    // By this stage, we will have at least target selections spread across
    // the participating ranks. Now, we need to process the selections to
    // make chunks.
    std::vector<Node *> chunks;
    std::vector<bool> own_chunks;
    for(size_t i = 0; i < selections.size(); i++)
    {
        if(selections[i] == nullptr)
        {
            // We had a "null" selection so we'll take the whole mesh.
            chunks.push_back(meshes[i]);
            own_chunks.push_back(false);
        }
        else
        {
            conduit::Node *chunk = extract(i, *meshes[i]);
chunk->print()
            chunks.push_back(chunk);
            own_chunks.push_back(false);
        }
    }


#if 0
    // Though, since Conduit permits datasets with different fields, etc, we
    // probably want to send/recv the data as binary blobs. Then deserialize
    // and assemble into combined grids.
#endif


    // Now we have chunks that are the extracted parts of the
    // meshes that we created.


    // We need to figure out ownership and make sure each rank has the parts
    // that it needs.


    // Now that we have all the parts we need, combine the partitions to
    // arrive at the target number of partitions for this rank. This combination
    // process appends partition meshes together.


    // Store the combined outputs into the output conduit::Node.
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
#endif

//-------------------------------------------------------------------------
void
partition(const conduit::Node &mesh, const conduit::Node &options, conduit::Node &output)
{
    auto doms = domains(mesh);
    if(doms.empty())
        return;

    partitioner P;
    if(P.initialize(n_mesh, options))
    {
        P.adjust_selections();
        P.execute(output);
    }
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh --
//-----------------------------------------------------------------------------

#if 0//def CONDUIT_PARALLEL_PARTITION
// -- consider moving this into conduit_blueprint_mpi_mesh.cpp

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi --
//-----------------------------------------------------------------------------
namespace mpi
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{

//-------------------------------------------------------------------------
void
partition(const conduit::Node &n, const conduit::Node &options,
    conduit::Node &output, MPI_Comm comm)
{
    // Figure out the number of domains in the input mesh.
    auto ndoms = number_of_domains(mesh, comm);


    internal::partition(mesh, ndoms, options, output, comm);
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh --
//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi --
//-----------------------------------------------------------------------------
#endif

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif
