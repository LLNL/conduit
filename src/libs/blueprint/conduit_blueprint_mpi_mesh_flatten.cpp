// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh_flatten.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mpi_mesh_flatten.hpp"

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <array>
#include <cmath>
#include <iostream>
#include <unordered_set>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_log.hpp"
#include "conduit_relay_mpi.hpp"

using conduit::utils::log::quote;

// Debug macros
// #define DEBUG_MESH_FLATTEN 1

#ifdef DEBUG_MESH_FLATTEN
#define DEBUG_PRINT(stream)\
do {\
    std::cout << stream;\
} while(0)
#else
#define DEBUG_PRINT(msg)
#endif

//-----------------------------------------------------------------------------
static conduit::index_t
assoc_to_index_t(const std::string &assoc)
{
    return (assoc == "vertex") ? 0 : 1;
}

static bool
is_mc_array(const conduit::Node &n)
{
    const conduit::DataType &dt = n.dtype();
    return dt.is_list() || dt.is_object();
}

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
// -- begin conduit::blueprint::mpi::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{


//-----------------------------------------------------------------------------
ParallelMeshFlattener::ParallelMeshFlattener(MPI_Comm comm)
    : MeshFlattener(), comm(comm)
{
    // Invoke MeshFlattener to set defaults
    root = 0;
    rank = relay::mpi::rank(comm);
    add_rank = false;
}

//-----------------------------------------------------------------------------
ParallelMeshFlattener::~ParallelMeshFlattener()
{
    // Nothing special yet
}

//-----------------------------------------------------------------------------
bool
ParallelMeshFlattener::set_options(const Node &opts)
{
    bool ok = MeshFlattener::set_options(opts);
    // "add_rank", boolean
    if(opts.has_child("add_rank"))
    {
        const Node &n_add_vertex_locations = opts["add_rank"];
        if(n_add_vertex_locations.dtype().is_number())
        {
            this->add_rank = opts["add_rank"].to_int() != 0;
        }
        else
        {
            ok = false;
            CONDUIT_ERROR("options[" << quote("add_rank") <<
                "] must be a number. It will be treated as a boolean (.to_int() != 0).");
        }
    }
    return ok;
}

//-----------------------------------------------------------------------------
ParallelMeshFlattener::FieldInfo::FieldInfo()
    : field_names(), field_ncomps(), field_assocs(),
        field_dtypes(), comp_names()
{

}

//-----------------------------------------------------------------------------
ParallelMeshFlattener::FieldInfo::FieldInfo(const Node &n)
{
    from_node(n);
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::FieldInfo::from_node(const Node &n)
{
    DEBUG_PRINT("Rank " << relay::mpi::rank(MPI_COMM_WORLD) << " - from_node" << std::endl);
    field_names.clear();
    field_ncomps.clear();
    field_assocs.clear();
    field_dtypes.clear();
    comp_names.clear();

    const index_t nfields = n["field_names"].number_of_children();
    const index_t *ncomps = (const index_t*)n["field_ncomps"].element_ptr(0);
    const index_t *assocs = (const index_t*)n["field_assocs"].element_ptr(0);
    const index_t *dtypes = (const index_t*)n["field_dtypes"].element_ptr(0);

    for(index_t i = 0; i < nfields; i++)
    {
        field_names.push_back(n["field_names"][i].as_string());
        field_ncomps.push_back(ncomps[i]);
        field_assocs.push_back(assocs[i]);
        field_dtypes.push_back(dtypes[i]);
    }

    for(index_t i = 0; i < n["comp_names"].number_of_children(); i++)
    {
        comp_names.push_back(n["comp_names"][i].as_string());
    }
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::FieldInfo::to_node(Node &out) const
{
    out.reset();
    out["field_names"].set(DataType::list());
    out["field_ncomps"].set(field_ncomps);
    out["field_assocs"].set(field_assocs);
    out["field_dtypes"].set(field_dtypes);
    out["comp_names"].set(DataType::list());
    for(index_t i = 0; i < (index_t)field_names.size(); i++)
    {
        out["field_names"].append().set(field_names[i]);
    }
    for(index_t i = 0; i < (index_t)comp_names.size(); i++)
    {
        out["comp_names"].append().set(comp_names[i]);
    }
}

//-----------------------------------------------------------------------------
ParallelMeshFlattener::MeshMetaData::MeshMetaData()
    : coord_type(DataType::EMPTY_ID), counts(), nverts(0), nelems(0)
//      field_info()
{

}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::add_mpi_rank(const MeshInfo &my_mesh, Node &output) const
{
    const auto set_rank = [this](index_t, index_t &cval) {
        cval = (index_t)this->rank;
    };

    if(output.has_path("vertex_data/values"))
    {
        Node &vertex_values = output["vertex_data/values"];
        if(vertex_values.has_child("mpi_rank"))
        {
            Node &mpi_rank = vertex_values["mpi_rank"];
            blueprint::mesh::utils::for_each_in_range<index_t>(mpi_rank, 0, my_mesh.nverts, set_rank);
        }
    }

    if(output.has_path("element_data/values"))
    {
        Node &element_values = output["element_data/values"];
        if(element_values.has_child("mpi_rank"))
        {
            Node &mpi_rank = element_values["mpi_rank"];
            blueprint::mesh::utils::for_each_in_range<index_t>(mpi_rank, 0, my_mesh.nelems, set_rank);
        }
    }
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::build_local_field_info(
    const std::vector<std::string> &fields_to_flatten,
    const Node &mesh, const std::string &topo_name, Node &my_fields) const
{
    const index_t nfields = (index_t)fields_to_flatten.size();
    my_fields.reset();
    my_fields["field_names"].set(DataType::list());
    my_fields["field_ncomps"].set(DataType::index_t(nfields));
    my_fields["field_assocs"].set(DataType::index_t(nfields));
    my_fields["field_dtypes"].set(DataType::index_t(nfields));
    my_fields["comp_names"].set(DataType::list());
    index_t *field_ncomps = (index_t*)my_fields["field_ncomps"].element_ptr(0);
    index_t *field_assocs = (index_t*)my_fields["field_assocs"].element_ptr(0);
    index_t *field_dtypes = (index_t*)my_fields["field_dtypes"].element_ptr(0);
    for(index_t i = 0; i < nfields; i++)
    {
        const std::string &field_name = fields_to_flatten[i];
        const Node *field = get_reference_field(mesh, topo_name, field_name);
        if(field)
        {
            const Node &values = field->child("values");
            const index_t ncomps = (is_mc_array(values) ? values.number_of_children() : 0);
            my_fields["field_names"].append().set(field_name);
            field_ncomps[i] = ncomps;
            field_assocs[i] = assoc_to_index_t(field->child("association").as_string());
            field_dtypes[i] = determine_element_dtype(values);
            for(index_t i = 0; i < ncomps; i++)
            {
                my_fields["comp_names"].append().set(values[i].name());
            }
        }
    }
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::calculate_unique_fields(const Node &all_fields,
    Node &out) const
{
    DEBUG_PRINT("Rank " << rank << " - calculate_unique_fields");
    FieldInfo gfi;

    // Build the list of unqiue field names
    std::unordered_set<std::string> field_set;
    for(index_t i = 0; i < all_fields.number_of_children(); i++)
    {
        const FieldInfo fi(all_fields[i]);
        for(index_t j = 0; j < (index_t)fi.field_names.size(); j++)
        {
            const std::string field_name = fi.field_names[j];
            if(field_set.count(field_name) == 0)
            {
                const index_t ncomps = fi.field_ncomps[j];
                field_set.insert(field_name);
                gfi.field_names.push_back(field_name);
                gfi.field_ncomps.push_back(ncomps);
                gfi.field_assocs.push_back(fi.field_assocs[j]);
                gfi.field_dtypes.push_back(fi.field_dtypes[j]);
                if(ncomps > 0)
                {
                    index_t off = 0;
                    for(index_t k = 0; k < j; k++)
                    {
                        off += fi.field_ncomps[k];
                    }

                    for(index_t k = 0; k < ncomps; k++)
                    {
                        gfi.comp_names.push_back(fi.comp_names[off + k]);
                    }
                }
            }
        }
    }

    gfi.to_node(out);
}

//-----------------------------------------------------------------------------
ParallelMeshFlattener::FieldInfo
ParallelMeshFlattener::determine_global_fields(const Node &mesh) const
{
    DEBUG_PRINT("Rank " << rank << " - determine_global_fields" << std::endl);
    FieldInfo info;

    // The user has not provided a list of fields
    const Node &topo = get_topology(mesh[0]);
    const std::string topo_name = topo.name();

    std::vector<std::string> fields_to_flatten;
    try
    {
        get_fields_to_flatten(mesh, topo_name, fields_to_flatten);
    }
    catch(const conduit::Error &e)
    {
        CONDUIT_INFO("Error caught trying to get_fields_to_flatten on rank " << relay::mpi::rank(comm)
            << ", this rank will not contribute to the final table. The error: "
            << e.message());
        fields_to_flatten.clear();
    }

    if(this->field_names.empty())
    {
        // TODO: Optimize for large N ranks.

        // Create the local list of field info
        Node my_fields;
        build_local_field_info(fields_to_flatten, mesh, topo_name, my_fields);

        Node all_fields;
        relay::mpi::gather_using_schema(my_fields, all_fields, root, comm);
        DEBUG_PRINT("Gather using schema worked!" << std::endl);

        Node global_fields;
        if(rank == root)
        {
            calculate_unique_fields(all_fields, global_fields);
        }
        relay::mpi::broadcast_using_schema(global_fields, root, comm);
        DEBUG_PRINT("Broadcast using schema worked!" << std::endl);
        info.from_node(global_fields);
    }
    else
    {
        // The user has provided a list of fields, rely on root's data types and ordering.
        Node global_fields;
        if(rank == root)
        {
            build_local_field_info(fields_to_flatten, mesh, topo_name, global_fields);
        }
        relay::mpi::broadcast_using_schema(global_fields, root, comm);
        info.from_node(global_fields);
    }
    return info;
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::gather_global_mesh_metadata(const MeshInfo &my_info,
    MeshMetaData &out) const
{
    DEBUG_PRINT("Rank " << rank << " - gather_global_mesh_metadata" << std::endl);
    out = MeshMetaData();
    const auto mtype = relay::mpi::conduit_dtype_to_mpi_dtype(DataType::index_t());

    // NOTE: small ints < large ints < small floats < large floats, can compute max
    MPI_Allreduce(&my_info.coord_type, &out.coord_type, 1, mtype, MPI_MAX, comm);

    // Now need per rank nverts / nelems
    const std::array<index_t, 2> my_counts{my_info.nverts, my_info.nelems};
    const int comm_size = relay::mpi::size(comm);
    if(rank == root)
    {
        out.counts.resize(comm_size * 2);
    }
    MPI_Gather(my_counts.data(), my_counts.size(), mtype, out.counts.data(),
        my_counts.size(), mtype, root, comm);

    if(rank == root)
    {
        out.nverts = 0;
        out.nelems = 0;
        auto itr = out.counts.begin();
        while(itr != out.counts.end())
        {
            out.nverts += *itr++;
            out.nelems += *itr++;
        }
    }
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::gather_values(int nrows, int *rank_counts,
    int *rank_offsets, Node &values) const
{
    DEBUG_PRINT("Rank " << rank << " - gather_values" << std::endl);
    for(index_t i = 0; i < values.number_of_children(); i++)
    {
        Node &value = values[i];
        const index_t ncomps = value.number_of_children();
        if(ncomps > 0)
        {
            // mcarray
            for(index_t j = 0; j < ncomps; j++)
            {
                Node &column = value[j];
                void *ptr = column.element_ptr(0);
                const auto mtype = relay::mpi::conduit_dtype_to_mpi_dtype(column.dtype());
                // DEBUG_PRINT("Rank " << rank << " - MPI_Gatherv(" << ptr << ", " << nrows
                //     << ", " << mtype << ", " << ptr << ", " << rank_counts << ", " << rank_offsets
                //     << ", " << mtype << ", " << root << ", " << comm << ");" << std::endl);
                DEBUG_PRINT("Rank " << rank << " - MPI_Gatherv(" << value.name()
                    << "[" << column.name() << "]);" << std::endl);
                MPI_Gatherv(ptr, nrows, mtype,
                    ptr, rank_counts, rank_offsets,
                    mtype, root, comm);
            }
        }
        else
        {
            void *ptr = value.element_ptr(0);
            const auto mtype = relay::mpi::conduit_dtype_to_mpi_dtype(value.dtype());
            // DEBUG_PRINT("Rank " << rank << " - MPI_Gatherv(" << ptr << ", " << nrows
            //     << ", " << mtype << ", " << ptr << ", " << rank_counts << ", " << rank_offsets
            //     << ", " << mtype << ", " << root << ", " << comm << ");" << std::endl);
            DEBUG_PRINT("Rank " << rank << " - MPI_Gatherv(" << value.name() << ");" << std::endl);
            MPI_Gatherv(ptr, nrows, mtype, ptr, rank_counts,
                rank_offsets, mtype, root, comm);
        }
    }
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::gather_results(const MeshInfo &my_info,
    const MeshMetaData &global_meta_data,
    Node &output) const
{
    DEBUG_PRINT("Rank " << rank << " - gather_results" << std::endl);
    // Everyone should have the same exact columns
    const int comm_size = relay::mpi::size(comm);
    std::vector<int> rank_counts;
    std::vector<int> rank_offsets;
    if(rank == root)
    {
        rank_counts.resize(comm_size);
        rank_offsets.resize(comm_size);
        int offset = 0;
        const index_t *c = global_meta_data.counts.data();
        for(int i = 0; i < comm_size; i++, c += 2)
        {
            int count = static_cast<int>(*c);
            rank_counts[i] = count;
            rank_offsets[i] = offset;
            offset += count;
        }
#ifdef DEBUG_MESH_FLATTEN
        std::cout << "Rank counts:\n  ";
        for(int count : rank_counts) std::cout << count << ", ";
        std::cout << "\nRank offsets:\n  ";
        for(int off : rank_offsets) std::cout << off << ", ";
        std::cout << std::endl;
#endif
    }

    gather_values(my_info.nverts, rank_counts.data(), rank_offsets.data(),
        output["vertex_data/values"]);
    DEBUG_PRINT("Rank " << rank << " - done gathering vertex values." << std::endl);

    if(rank == root)
    {
        int offset = 0;
        const index_t *c = global_meta_data.counts.data() + 1;
        for(int i = 0; i < comm_size; i++, c += 2)
        {
            int count = static_cast<int>(*c);
            rank_counts[i] = count;
            rank_offsets[i] = offset;
            offset += count;
        }
#ifdef DEBUG_MESH_FLATTEN
        std::cout << "Rank counts:\n  ";
        for(int count : rank_counts) std::cout << count << ", ";
        std::cout << "\nRank offsets:\n  ";
        for(int off : rank_offsets) std::cout << off << ", ";
        std::cout << std::endl;
#endif
    }

    gather_values(my_info.nelems, rank_counts.data(), rank_offsets.data(),
        output["element_data/values"]);
    DEBUG_PRINT("Rank " << rank << " - done gathering element values." << std::endl);
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::make_local_allocations(const MeshInfo &info,
    const FieldInfo &fi, const index_t coords_dtype, Node &output) const
{
    DEBUG_PRINT("Rank " << rank << " - make_local_allocations" << std::endl);
    output.reset();
    Node &vertex_values = output["vertex_data/values"];
    Node &element_values = output["element_data/values"];
    vertex_values.set(DataType::object());
    vertex_values.set(DataType::object());

    // Allocate cset output
    if(this->add_vertex_locations)
    {
        Node &cset_values = vertex_values[info.cset_name];
        for(index_t d = 0; d < info.dimension; d++)
        {
            allocate_column(cset_values[info.axes[d]],
                info.nverts, coords_dtype);
        }
    }

    // Allocate cell center output
    if(this->add_cell_centers)
    {
        Node &elem_center_values = element_values["element_centers"];
        for(index_t d = 0; d < info.dimension; d++)
        {
            allocate_column(elem_center_values[info.axes[d]],
                info.nelems, this->default_dtype);
        }
    }

    // Allocate fields output
    for(index_t i = 0; i < (index_t)fi.field_names.size(); i++)
    {
        const std::string &name = fi.field_names[i];
        const index_t ncomps = fi.field_ncomps[i];
        const index_t assoc = fi.field_assocs[i];
        const index_t dtype = fi.field_dtypes[i];
        Node &table_values = (assoc == 0) ? vertex_values : element_values;
        const index_t nrows = (assoc == 0) ? info.nverts : info.nelems;
        if(ncomps > 0)
        {
            // mcarray
            Node ref_node;
            index_t off = 0;
            for(index_t j = 0; j < i; j++)
            {
                off += fi.field_ncomps[j];
            }
            for(index_t j = 0; j < ncomps; j++)
            {
                ref_node.add_child(fi.comp_names[off + j]);
            }
            allocate_column(table_values[name], nrows, dtype, &ref_node);
        }
        else
        {
            allocate_column(table_values[name], nrows, dtype);
        }
    }

    // Add domain / rank information to each table
    // Only if the table contains data.
    const DataType dt_index_t = DataType::index_t(1);
    if(vertex_values.number_of_children() > 0)
    {
        if(this->add_domain_info)
        {
            allocate_column(vertex_values["domain_id"], info.nverts, dt_index_t.id());
            allocate_column(vertex_values["vertex_id"], info.nverts, dt_index_t.id());
        }

        if(this->add_rank)
        {
            allocate_column(vertex_values["mpi_rank"], info.nverts, dt_index_t.id());
        }
    }

    if(element_values.number_of_children() > 0)
    {
        if(this->add_domain_info)
        {
            allocate_column(element_values["domain_id"], info.nelems, dt_index_t.id());
            allocate_column(element_values["element_id"], info.nelems, dt_index_t.id());
        }

        if(this->add_rank)
        {
            allocate_column(element_values["mpi_rank"], info.nelems, dt_index_t.id());
        }
    }
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::make_root_allocations(const MeshMetaData &mdata,
    const FieldInfo &fi, const MeshInfo &my_info, Node &output) const
{
    DEBUG_PRINT("Rank " << rank << " - make_root_allocations" << std::endl);
    output.reset();
    Node &vertex_values = output["vertex_data/values"];
    Node &element_values = output["element_data/values"];
    vertex_values.set(DataType::object());
    vertex_values.set(DataType::object());

    // Allocate cset output
    if(this->add_vertex_locations)
    {
        Node &cset_values = vertex_values[my_info.cset_name];
        for(index_t d = 0; d < my_info.dimension; d++)
        {
            allocate_column(cset_values[my_info.axes[d]],
                mdata.nverts, mdata.coord_type);
        }
    }

    // Allocate cell center output
    if(this->add_cell_centers)
    {
        Node &elem_center_values = element_values["element_centers"];
        for(index_t d = 0; d < my_info.dimension; d++)
        {
            allocate_column(elem_center_values[my_info.axes[d]],
                mdata.nelems, this->default_dtype);
        }
    }

    // Allocate fields output
    for(index_t i = 0; i < (index_t)fi.field_names.size(); i++)
    {
        const std::string &name = fi.field_names[i];
        const index_t ncomps = fi.field_ncomps[i];
        const index_t assoc = fi.field_assocs[i];
        const index_t dtype = fi.field_dtypes[i];
        Node &table_values = (assoc == 0) ? vertex_values : element_values;
        const index_t nrows = (assoc == 0) ? mdata.nverts : mdata.nelems;
        if(ncomps > 0)
        {
            // mcarray
            Node ref_node;
            index_t off = 0;
            for(index_t j = 0; j < i; j++)
            {
                off += fi.field_ncomps[j];
            }
            for(index_t j = 0; j < ncomps; j++)
            {
                ref_node.add_child(fi.comp_names[off + j]);
            }
            allocate_column(table_values[name], nrows, dtype, &ref_node);
        }
        else
        {
            allocate_column(table_values[name], nrows, dtype);
        }
    }

    // Add domain / rank information to each table
    // Only if the table contains data.
    const DataType dt_index_t = DataType::index_t(1);
    if(vertex_values.number_of_children() > 0)
    {
        if(this->add_domain_info)
        {
            allocate_column(vertex_values["domain_id"], mdata.nverts, dt_index_t.id());
            allocate_column(vertex_values["vertex_id"], mdata.nverts, dt_index_t.id());
        }

        if(this->add_rank)
        {
            allocate_column(vertex_values["mpi_rank"], mdata.nverts, dt_index_t.id());
        }
    }

    if(element_values.number_of_children() > 0)
    {
        if(this->add_domain_info)
        {
            allocate_column(element_values["domain_id"], mdata.nelems, dt_index_t.id());
            allocate_column(element_values["element_id"], mdata.nelems, dt_index_t.id());
        }

        if(this->add_rank)
        {
            allocate_column(element_values["mpi_rank"], mdata.nelems, dt_index_t.id());
        }
    }
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::cleanup_output(Node &output) const
{
    // On all ranks other than root, remove data.
    if(rank != root)
    {
        output.reset();
    }
    else
    {
        blueprint::mesh::MeshFlattener::cleanup_output(output);
    }
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::flatten_many_domains(const Node &mesh, Node &output) const
{
    DEBUG_PRINT("Rank " << rank << " - flatten_many_domains" << std::endl);
    FieldInfo global_field_info = determine_global_fields(mesh);

    MeshInfo my_mesh_info;
    try
    {
        collect_mesh_info(mesh, my_mesh_info);
    }
    catch(const conduit::Error &e)
    {
        CONDUIT_INFO("Error caught trying to collect_mesh_info on rank " << rank
            << ", this rank will not contribute to the final table. The error: "
            << e.message());
        my_mesh_info = MeshInfo();
    }

    // Only coord_type filled in on all ranks, everything else is only on root.
    MeshMetaData global_metadata;
    gather_global_mesh_metadata(my_mesh_info, global_metadata);

    if(rank != root)
    {
        make_local_allocations(my_mesh_info, global_field_info,
            global_metadata.coord_type, output);
    }
    else
    {
        make_root_allocations(global_metadata, global_field_info,
            my_mesh_info, output);
    }

    DEBUG_PRINT("Rank " << rank << output.schema().to_json() << std::endl);

    // Do local flattening on each rank
    index_t vert_offset = 0;
    index_t elem_offset = 0;
    for(index_t i = 0; i < my_mesh_info.ndomains; i++)
    {
        DEBUG_PRINT("Rank " << rank << " flattening domain " << i << std::endl);
        flatten_single_domain(mesh[i], output, global_field_info.field_names,
            my_mesh_info.domain_ids[i], vert_offset, elem_offset);
        vert_offset += my_mesh_info.verts_per_domain[i];
        elem_offset += my_mesh_info.elems_per_domain[i];
    }

    if(this->add_rank)
    {
        add_mpi_rank(my_mesh_info, output);
    }

    gather_results(my_mesh_info, global_metadata, output);

    cleanup_output(output);

    // TODO: Remove empty tables
    DEBUG_PRINT("Rank " << rank << " done flattening.");
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
// -- end conduit --
//-----------------------------------------------------------------------------
