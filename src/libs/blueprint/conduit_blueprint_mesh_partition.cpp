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

#ifdef CONDUIT_PARALLEL_PARTITION
#include <mpi.h>
#endif

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

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::internal --
//-----------------------------------------------------------------------------
namespace internal
{

//-------------------------------------------------------------------------
/**
  options["comm"] = MPI_COMM_WORLD;

  "Information on how to re-decompose the grids will be provided or be easily calculatable."

  To go from 1-N, we need a way to pull the mesh apart. We would be able to
  provide the ways that this is done to select the bits we want so each selection
  makes a piece in the output mesh.

# Selections provide criteria used to split input meshes
selections:
   logical:
      -
       domain: 0
       start: [0,0,0]
       end:   [10,10,10]
      -
       domain: 1
       start: [50,50,50]
       end:   [60,60,60]
   cells:
      -
       domain: 0
       indices: [0,1,2,3,4,5,6,7,8,9]
      -
       domain: 0
       ranges: [0,100, 1000,2000]
      -
       domain: 1
       global_indices: [100,101,102,110,116,119,220]
   spatial:
      -
       box: [0., 0., 0., 11., 12., 22.]
      -
       box: [0., 0., 22., 11., 12., 44.]
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

  options["mapping"] = true;    // If on, preserve original cells, node ids 
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

    virtual bool init(const Node *n_opt_ptr) = 0;

    /**
     @brief Determines whether the selection can be applied to the supplied mesh.
     @param n_mesh A Conduit node containing the mesh.
     @return True if the selection can be applied to the mesh; False otherwise.
     */
    virtual bool applicable(const Node &n_mesh) = 0;

    /**
     @brief Return the number of cells in the selection.
     @return The number of cells in the selection.
     */
    virtual index_t length() const { return 0; }

    /**
     @brief Partitions the selection into smaller selections.
     @param n_mesh A Conduit node containing the mesh.
     @return A vector of selection pointers that cover the input selection.
     */
    virtual std::vector<std::shared_ptr<selection> > partition(const Node &n_mesh) const = 0;

    /**
     @brief Extract the selection from the mesh.
     @param n_mesh A Conduit node representing the mesh.
     @param n_output A Conduit node containing the extracted mesh.
     */
    virtual void extract(const Node &n_mesh, Node &n_output) const = 0;

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
     @brief Returns whether the input mesh has a logical structure.
     @param n_mesh A Conduit node representing the mesh.
     @return True for structured, false for unstructured.
     */
    static bool mesh_has_logical_structure(const Node &n_mesh);

protected:
    void copy_fields(const Node &n_mesh, Node &output) const;
    void copy_field(Node &n_output_fields, const Node &n_field,
                    const std::vector<index_t> &ids) const;
    void slice_array(Node &n_dest_values, const Node &n_src_values,
                     const std::vector<index_t> &ids) const;

    virtual void get_vertex_ids(std::vector<index_t> &vertex_ids) const = 0;
    virtual void get_element_ids(std::vector<index_t> &vertex_ids) const = 0;

    static const std::string DOMAIN_KEY;
    static const std::string MAPPING_KEY;

    const Node *n_options_ptr;
};

const std::string selection::DOMAIN_KEY("domain");
const std::string selection::MAPPING_KEY("mapping");

//---------------------------------------------------------------------------
bool
selection::mesh_has_logical_structure(const Node &n_mesh)
{
    bool structured = false;

    if (n_coords["type"].as_string() == "uniform")
        structured = true;
    else if (n_coords["type"].as_string() == "rectilinear")
        structured = true;
    else if (n_coords["type"].as_string() == "explicit")
    {
        const Node &n_topo = n_mesh["topologies"][0];
        if (n_topo["type"].as_string() == "structured")
            structured = true;
    }
    return structured;
}

//---------------------------------------------------------------------------
void
selection::copy_fields(const Node &n_mesh, Node &n_output) const
{
    if(n_mesh.has_child("fields"))
    {
        const Node &n_fields = n_mesh["fields"];
        int n_vertex_centered = 0, n_element_centered = 0;
        for(size_t i = 0; i < n_fields.number_of_children(); i++)
        {
            if(n_fields[i].has_child("association"))
            {
                auto association = n_fields[i]["association"].as_string();
                if(association == "element")
                    n_element_centered++;
                else if(association == "vertex")
                    n_vertex_centered++;
            }
        }

        if(n_vertex_centered > 0)
        {
            std::vector<index_t> vertex_ids;
            get_vertex_ids(n_mesh, vertex_ids);

            Node &n_output_fields = n_output["fields"];
            for(size_t i = 0; i < n_fields.number_of_children(); i++)
            {
                const Node &n_field = n_fields[i];
                if(n_field.has_child("association"))
                {
                    auto association = n_field["association"].as_string();
                    if(association == "vertex")
                    {
                        copy_field(n_output_fields, n_field, vertex_ids);
                    }
                }
            }

            if(preserve_mapping())
            {
                // TODO: save the vertex_ids as a new field.
            }
        }

        if(n_element_centered > 0)
        {

// NOTE: The cell ids could be used in computing point ids. The point ids would need to be passed in so we can deal with coordinates.

            std::vector<index_t> element_ids;
            get_element_ids(element_ids);

            Node &n_output_fields = n_output["fields"];
            for(size_t i = 0; i < n_fields.number_of_children(); i++)
            {
                const Node &n_field = n_fields[i];
                if(n_field.has_child("association"))
                {
                    auto association = n_field["association"].as_string();
                    if(association == "element")
                    {
                        copy_field(n_output_fields, n_field, element_ids);
                    }
                }
            }

            if(preserve_mapping())
            {
                // TODO: save the element_ids as a new field.
            }
        }
    }
}

//---------------------------------------------------------------------------
void
selection::copy_field(Node &n_output_fields, const Node &n_field,
    const std::vector<index_t> &ids) const
{
    static const std::vector<std::string> keys{"association", "grid_function",
        "volume_dependent", "topology"};

// TODO: What about matsets and mixed fields?...
// https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#fields

    // Copy common field attributes from the old field into the new one.
    Node &n_new_field = n_output_fields[n_field.name()];
    for(const auto &key : keys)
    {
        if(n_field.has_child(key))
            n_new_field[key] = n_field[key];
    }

    const Node &n_values = n_field["values"];
    if(n_values.dtype().is_compact()) 
    {
        slice_array(n_new_field["values"], n_values, ids);
    }
    else
    {
        // otherwise, we need to compact our data first
        Node n;
        n_values.compact_to(n);
        slice_array(n_new_field["values"], n, ids);
    }
}

//---------------------------------------------------------------------------
// @brief Slice the n_src array using the indices stored in ids. The 
//        destination memory is already pointed to by n_dest.
template <typename T>
inline T *
slice_array(Node &n_dest, const Node &n_src, const std::vector<index_t> &ids)
{
    const T *src = reinterpret_cast<const T *>(n_src.data_ptr());
    T *dest = reinterpret_cast<T *>(n_dest.data_ptr());
    size_t n = ids.size();
    for(size_t i = 0; i < n; i++)
        src[i] = dest[ids[i]];
}

//---------------------------------------------------------------------------
// @note Should this be part of conduit::Node or DataArray somehow. The number
//       of times I've had to slice an array...
void
selection::slice_array(Node &n_dest_values, const Node &n_src_values,
    const std::vector<index_t> &ids) const
{
    // Copy the DataType of the input node but override the number of elements
    // before copying it in so assigning to n_dest_values triggers a memory
    // allocation.
    n_dest_values = DataType(n_src_values.dtype().id(), ids.size());

    // Do the slice.
    if(dt.is_int8())
        slice_array<int8>(n_dest_values, n_src_values, ids);
    else if(dt.is_int16())
        slice_array<int16>(n_dest_values, n_src_values, ids);
    else if(dt.is_int32())
        slice_array<int32>(n_dest_values, n_src_values, ids);
    else if(dt.is_int64())
        slice_array<int64>(n_dest_values, n_src_values, ids);
    else if(dt.is_uint8())
        slice_array<uint8>(n_dest_values, n_src_values, ids);
    else if(dt.is_uint16())
        slice_array<uint16>(n_dest_values, n_src_values, ids);
    else if(dt.is_uint32())
        slice_array<uint32>(n_dest_values, n_src_values, ids);
    else if(dt.is_uint64())
        slice_array<uint64>(n_dest_values, n_src_values, ids);
    else if(dt.is_float32())
        slice_array<float32>(n_dest_values, n_src_values, ids);
    else if(dt.is_float64())
        slice_array<float64>(n_dest_values, n_src_values, ids);
    else if(dt.is_char())
        slice_array<char>(n_dest_values, n_src_values, ids);
    else if(dt.is_short())
        slice_array<short>(n_dest_values, n_src_values, ids);
    else if(dt.is_int())
        slice_array<int>(n_dest_values, n_src_values, ids);
    else if(dt.is_long())
        slice_array<long>(n_dest_values, n_src_values, ids);
    else if(dt.is_unsigned_char())
        slice_array<unsigned char>(n_dest_values, n_src_values, ids);
    else if(dt.is_unsigned_short())
        slice_array<unsigned short>(n_dest_values, n_src_values, ids);
    else if(dt.is_unsigned_int())
        slice_array<unsigned int>(n_dest_values, n_src_values, ids);
    else if(dt.is_unsigned_long())
        slice_array<unsigned long>(n_dest_values, n_src_values, ids);
    else if(dt.is_float())
        slice_array<float>(n_dest_values, n_src_values, ids);
    else if(dt.is_double())
        slice_array<double>(n_dest_values, n_src_values, ids);
}

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

    // Initializes the selection from a node.
    virtual bool init(const Node *n_opt_ptr) override;

    virtual bool applicable(const Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length() const override
    {
        return cells_for_axis(0) * 
               cells_for_axis(1) * 
               cells_for_axis(2);
    }

    virtual std::vector<std::shared_ptr<selection> > partition(const Node &n_mesh) const override;

    virtual void extract(const Node &n_mesh, Node &n_output) const override;

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

protected:
    virtual void get_vertex_ids(const Node &n_mesh,
                                std::vector<index_t> &vertex_ids) const override;
    virtual void get_element_ids(const Node &n_mesh,
                                 std::vector<index_t> &vertex_ids) const override;

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
selection_logical::init(const Node *n_opt_ptr)
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
selection_logical::applicable(const Node &n_mesh)
{
    bool retval = false;

    const Node &n_coords = n_mesh["coordsets"][0];
    const Node &n_topo = n_mesh["topologies"][0];
    bool is_uniform = n_coords["type"].as_string() == "uniform";
    bool is_rectilinear = n_coords["type"].as_string() == "rectilinear";
    bool is_structured = n_coords["type"].as_string() == "explicit" && 
                         n_topo["type"].as_string() == "structured");
    if(is_uniform || is_rectilinear || is_structured)
    {
        index_t dims[3] = {1,1,1};
        const Node &n_topo = n_mesh["topologies"][0];
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
selection_logical::partition(const Node &n_mesh) const
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
/**
 @note This method can likely move to selection instead of selection_logical.
 */
void
selection_logical::extract(const Node &n_mesh, Node &n_output) const
{
    const Node &n_coords = n_mesh["coordsets"][0];

    if (n_coords["type"].as_string() == "uniform")
    {
        n_output["coordsets/coords/type"] = "uniform";
        n_output["coordsets/coords/dims/i"] = 

        n_output["topologies/mesh/type"] = "uniform";
        n_output["topologies/mesh/coordset"] = "coords";

        copy_fields(n_mesh, n_output);
    }
    else if (n_coords["type"].as_string() == "rectilinear")
    {
        // TODO: rectilinear
    }
    else if (n_coords["type"].as_string() == "explicit")
    {
        const Node &n_topo = n_mesh["topologies"][0];
        if (n_topo["type"].as_string() == "structured")
        {
            // TODO: structured
        }
        else
        {
            // TODO: unstructured
        }
    }
}

//---------------------------------------------------------------------------
void
selection_logical::get_vertex_ids(const Node &n_mesh,
    std::vector<index_t> &ids) const
{
    index_t dims[3] = {1,1,1};
    const Node &n_topo = n_mesh["topologies"][0];
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

//---------------------------------------------------------------------------
void
selection_logical::get_element_ids(const Node &n_mesh,
    std::vector<index_t> &ids) const
{
    index_t dims[3] = {1,1,1};
    const Node &n_topo = n_mesh["topologies"][0];
    topology::logical_dims(n_topo, dims, 3);

    ids.clear();
    ids.reserve(length());
    auto mesh_CXCY = (dims[0] - 1) * (dims[1] - 1);
    auto mesh_CX   = (dims[0] - 1);
    for(index_t k = start[2]; k <= end[2]; k++)
    for(index_t j = start[1]; j <= end[1]; j++)
    for(index_t i = start[0]; i <= end[0]; i++)
    {
        ids.push_back(k*mesh_CXCY + j*mesh_CX + i);
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

    virtual bool init(const Node *n_opt_ptr) override
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

    virtual bool applicable(const Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length() const override
    {
        return num_cells_in_selection;
    }

    virtual std::vector<std::shared_ptr<selection> > partition(const Node &n_mesh) const override;

    const index_t *get_indices() const
    {
        // Access the converted data as index_t.
        return reinterpret_cast<const index_t *>(ids_storage.data_ptr());
    }

protected:
    virtual void get_vertex_ids(const Node &n_mesh,
                                std::vector<index_t> &ids) const override;
    virtual void get_element_ids(const Node &n_mesh,
                                 std::vector<index_t> &ids) const override;

private:
    static const std::string ELEMENTS_KEY;
    Node ids_storage;
    index_t num_cells_in_selection;
    index_t num_cells_in_mesh;
};

std::string selection_explicit::ELEMENTS_KEY("elements");

//---------------------------------------------------------------------------
/**
 @brief Returns whether the explicit selection applies to the input mesh.
 */
bool
selection_explicit::applicable(const Node &n_mesh)
{
    return true;
}

//---------------------------------------------------------------------------
std::vector<std::shared_ptr<selection> >
selection_explicit::partition(const Node &n_mesh) const
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
selection_explicit::extract(const Node &n_mesh, Node &output) const
{
    const Node &n_coords = n_mesh["coordsets"][0];

    if (n_coords["type"].as_string() == "uniform")
    {
        // TODO: extract from uniform
    }
    else if (n_coords["type"].as_string() == "rectilinear")
    {
        // TODO: extract from rectilinear
    }
    else if (n_coords["type"].as_string() == "explicit")
    {
        const Node &n_topo = n_mesh["topologies"][0];
        if (n_topo["type"].as_string() == "structured")
        {
            // TODO: extract from structured
        }
        else
        {
            // TODO: extract from unstructured
        }
    }
}

//---------------------------------------------------------------------------

/**
Last thought: This is kind of a general problem. Should I pass in the cell ids that we
              already determined since we could be calculating them anyway?

              I could make explicit and ranges handle the ugrid point determination
              with the same code if I passed in a set of cell ids.


 */



void
selection_explicit::get_vertex_ids(const Node &n_mesh,
    std::vector<index_t> &ids) const
{
    std::set<index_t> vertex_ids;
    auto n = ids_storage.dtype().number_of_elements();
    auto indices = get_indices();

    if(mesh_has_logical_structure(n_mesh))
    {
        index_t dims[3] = {1,1,1};
        const Node &n_topo = n_mesh["topologies"][0];
        topology::logical_dims(n_topo, dims, 3);
        index_t celldims[3];
        celldims[0] = std::max(dims[0] - 1, 1);
        celldims[1] = std::max(dims[1] - 1, 1);
        celldims[2] = std::max(dims[2] - 1, 1);

        auto ndims = topology::dims(n_mesh)
        if(ndims == 2)
        {
            celldims[2] = 0;

            // TODO: 
        }
        else if(ndims == 3)
        {
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
            for(index_t i = 0 i < n; i++)
            {
                // Get the IJK coordinate of the cell.
                grid_id_to_ijk(indices[i], celldims, cell_ijk); // We could probably do a much faster version

                // Turn the IJK into vertex ids.
                for(int i = 0; i < 8; i++)
                {
                    pt_ijk[0] = cell_ijk[0] + offset[i][0];
                    pt_ijk[1] = cell_ijk[1] + offset[i][1];
                    pt_ijk[2] = cell_ijk[2] + offset[i][2];
                    grid_ijk_to_id(pt_ijk, dims, ptid);

                    vertex_ids.insert(ptid);
                }
            }
        }
    }
    else
    {
        for(index_t i = 0 i < n; i++)
        {
            // Get the vertices used in the element.

            // Add them to vertex_ids set        
        }
    }

    // Return the sorted vertex ids
    ids.clear();
    ids.reserve(vertex_ids.size());
    for(auto it = vertex_ids.begin(); it != vertex_ids.end(); it++)
        ids.push_back(*it);
}

//---------------------------------------------------------------------------
void
selection_explicit::get_element_ids(const Node &/*n_mesh*/,
    std::vector<index_t> &ids) const
{
    // Making a copy of the ids here is not ideal since they are already
    // explicitly represented in the Conduit node. We might get around it
    // by passing back a DataArray instead of std::vector but that's worse
    // in other ways.
    auto n = ids_storage.dtype().number_of_elements();
    auto indices = get_indices();
    ids.resize(n);
    memcpy(&ids[0], indices, n * sizeof(index_t));
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

    virtual bool init(const Node *n_opt_ptr) override
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

    virtual bool applicable(const Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length() const override;

    virtual std::vector<std::shared_ptr<selection> > partition(const Node &n_mesh) const override;

    const index_t *get_ranges() const
    {
        // Access the converted data as index_t.
        return reinterpret_cast<const index_t *>(ranges_storage.data_ptr());
    }

private:
    static const std::string RANGES_KEY;
    Node ranges_storage;
};

std::string selection_ranges::RANGES_KEY("ranges");

bool
selection_ranges::applicable(const Node &/*n_mesh*/)
{
    return true;
}

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

std::vector<std::shared_ptr<selection> >
selection_ranges::partition(const Node &n_mesh) const
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



// These mesh classes provide some convenience methods for the various 
// Blueprint nodes that represent the meshes.
/*
class mesh
{
public:
    mesh(const Node &n) : node(n) { }
    virtual ~mesh() {}

    virtual get_number_of_cells() const = 0;
    virtual get_number_of_points() const = 0;

    std::vector<index_t> points_for_cells(const std::vector<index_t> &selected_cells) const = 0;

    Node &node;
};

class uniform_mesh : public mesh
{
public:
    uniform_mesh(const Node &n) : mesh(n)
    {
    }

    virtual ~uniform_mesh()
    {
    }

    virtual index_t get_number_of_cells() const override
    {
        return 0;
    }

    virtual index_t get_number_of_points() const override
    {
        return 0;
    }

    // Determine the points that we need given a set of selected cells.
    std::map<index_t> points_for_cells(const std::vector<index_t> &cellids) const override
    {
        std::map<index_t> point_ids;

        for(size_t index = 0; index < cellids.size(); index++)
        {
            index_t cell_ijk[3];
            grid_id_to_ijk(cellids[index], dims, cell_ijk);

            
        }
    }

    index_t dims[3];
};

mesh *
mesh_factory(const Node &n_mesh)
{
    const Node &n_coords = n_mesh["coordsets"][0];

    if (n_coords["type"].as_string() == "uniform")
    {
        return new uniform_mesh(n_mesh);
    }
    else if (n_coords["type"].as_string() == "rectilinear")
    {
        return new rectilinear_mesh(n_mesh);
    }
    else if (n_coords["type"].as_string() == "explicit")
    {
        const Node &n_topo = n_mesh["topologies"][0];
        if (n_topo["type"].as_string() == "structured")
        {
            return new structured_mesh(n_mesh);
        }
        else
        {
            return new unstructured_mesh(n_mesh);
        }
    }
    return nullptr;
}
*/

// NOTE: I might want to put all the logic into a class so I can subclass it in 
//       the MPI version to include methods that do communications.

void
partition(const Node &mesh, const Node &options, Node &output)
{
    auto doms = domains(mesh);
    if(doms.empty())
        return;

    partitioner P;
    if(P.initialize(n_mesh, options))
    {
        P.adjust_partitions();
        P.communicate_partitions();
        P.execute(output);
    }
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
class partitioner
{
public:
    partitioner();
    virtual ~partitioner();

    bool initialize(const Node &n_mesh, const Node &options);

    void split_selections(const Node &options);

    void communicate_partitions(const Node &options);

    void execute(const Node &options, Node &output);

    virtual long get_total_selections() const;

    virtual void get_largest_selection(int &sel_rank, int &sel_index) const;

protected:
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
bool
partitioner::initialize(const Node &n_mesh, const Node &options)
{
    auto doms = domains(mesh);

    // Iterate over the selections in the options and check them against the
    // domains that were passed in to make a vector of meshes and selections
    // that can be used to partition the meshes.
    if(options.has_child("selections"))
    {
        const Node &n_selections = options["selections"];
        if(n_selections.has_child("logical"))
        {
            const Node &n_logical = selections["logical"];
            for(size_t i = 0; i < n_logical.get_num_children(); i++)
            {
                const Node *this_node = n_logical.child_ptr(i);
                auto sel = std::make_shared<selection_logical>();
                if(sel->init(this_node))
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
        else if(n_selections.has_child("explicit"))
        {
            const Node &n_explicit = selections["explicit"];
            for(size_t i = 0; i < n_explicit.get_num_children(); i++)
            {
                const Node *this_node = n_explicit.child_ptr(i);
                if(this_node->has_child("ranges"))
                {
                    auto sel = std::make_shared<selection_ranges>();
                    if(sel->init(this_node))
                    {
                        for(size_t domid = 0; domid < doms.size(); domid++)
                        {
                            if(domid == sel->get_domain() && sel->applicable(doms[domainid]))
                            {
                                meshes.push_back(doms[domid]);
                                selections.push_back(sel);
                                break;
                            }
                        }
                    }
                }
                else if(this_node->has_child("ids"))
                {
                    auto sel = std::make_shared<selection_explicit>();
                    if(sel->init(this_node))
                    {
                        for(size_t domid = 0; domid < doms.size(); domid++)
                        {
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
partitioner::split_selections(const Node &options)
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
                const Node *m = meshes[index];
                meshes.insert(meshes.begin()+index, ps.size()-1, m);
                selections.insert(selections.begin()+index, ps.size()-1, nullptr);
                for(size_t i = 0; i < index; i++)
                    selections[index + i] = ps[i];
            }
        }
    }
}

void
partitioner::communicate_partitions(const Node &options)
{
}

void
partitioner::execute(const Node &options, Node &output)
{
    // By this stage, we will have at least target selections spread across
    // the participating ranks. Now, we need to process the selections to
    // make partitions.
    std::vector<Node *> chunks;
    for(size_t i = 0; i < selections.size(); i++)
    {
        chunks.push_back(selections[i]->extract(meshes[i]);
    }
    

#if 0
    //   -- to do this, we would need to be able to use a selection to extract
    //      parts of an array. This could be done by making a vector<int> of
    //      array indices (lots of memory) or we could handle it internally
    //      in the selections. Producing the vector might be the way to go
    //      since that could be used for both cell and point data.
/*
       sel->copy_cell_field(T *destination, const Node *n_field)

       // This one is hard for all but the logical selection since selections
       // deal with cells.
       sel->copy_point_field(T *destination, const Node *n_field)
*/
    std::vector<index_t> selection::get_cell_ids() const;
    std::vector<index_t> selection::get_point_ids() const;
    // This would let us copy into a temp buffer, send through MPI...
    copy_field(T *dest, const Node *n_field, const std::vector<index_t> &ids);

    // Though, since Conduit permits datasets with different fields, etc, we
    // probably want to send/recv the data as binary blobs. Then deserialize
    // and assemble into combined grids.
#endif


    // Now that we have partitions that are the extracted parts of the
    // meshes that we created.


    // We need to figure out ownership and make sure each rank has the parts
    // that it needs.


    // Now that we have all the parts we need, combine the partitions to
    // arrive at the target number of partitions for this rank. This combination
    // process appends partition meshes together.


    // Store the combined outputs into the output node.
}

//-------------------------------------------------------------------------
/**
 @brief This class accepts a set of input meshes and repartitions them
        according to input options. This class subclasses the partitioner
        class to add some parallel functionality.
 */
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

//-------------------------------------------------------------------------
void
partition(const Node &mesh, const Node &options, Node &output)
{
    auto doms = domains(mesh);
    if(doms.empty())
        return;

    partitioner P;
    if(P.initialize(n_mesh, options))
    {
        P.adjust_selections();
//        P.communicate_partitions();
        P.execute(output);
    }
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::internal --
//-----------------------------------------------------------------------------

//-------------------------------------------------------------------------
#ifndef CONDUIT_PARALLEL_PARTITION
void
partition(const conduit::Node &n, const conduit::Node &options,
    conduit::Node &output)
{
    auto ndoms = number_of_domains(mesh, comm);
    internal::partition(mesh, ndoms, options, output);
}
#endif

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
