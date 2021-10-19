// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_flatten.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <iostream>
#include <string>
#include <vector>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_log.hpp"

#define DEBUG_MESH_FLATTEN 1

#ifdef DEBUG_MESH_FLATTEN
#define DEBUG_PRINT(stream)\
do {\
    std::cout << stream;\
} while(0)
#else
#define DEBUG_PRINT(msg)
#endif

using namespace conduit;

class MeshFlattener
{
public:
    MeshFlattener();

    bool set_options(const Node &options);

    void execute(const Node &mesh, Node &output) const;
private:

    const Node &get_coordset(const Node &mesh) const;
    const Node &get_topology(const Node &mesh) const;

    void get_fields_to_flatten(const Node &mesh, std::vector<std::string> &fields_to_flatten) const;
    index_t determine_element_dtype(const Node &data) const;
    void default_initialize_column(Node &column) const;
    void allocate_column(Node &column, index_t nrows, index_t dtype_id,
        const Node *ref_node = nullptr) const;

    template<typename SrcType, typename DestType>
    static void append_data_array_impl2(const DataArray<SrcType> &src, DataArray<DestType> &dest, index_t offset);
    template<typename SrcType>
    static void append_data_array_impl1(const DataArray<SrcType> &src, Node &dest, index_t offset);
    static void append_data_array(const Node &src, Node &dest, index_t offset);
    static void append_mc_data(const Node &src, Node &dest, index_t offset);
    static void append_data(const Node &src, Node &dest, index_t offset);

    /**
    @brief Determines node's actual data type then iterates the elements in range [start, end),
        calling func on each element.
    @param node The input data to be read or modified.
    @param start The starting offset into node's data.
    @param end The end of the range to be iterated.
    @param func A callable object that accepts the arguments (index_t i, AccessType &value)
        where "i" is the current iteration value (from 0 to (end-start-1)), and "value" is a
        temporary reference to the current value in node's data.
    @note The template parameter AccessType is defaulted to float. This type is used to determine
        the value that each element should be cast to before entering func. After func is invoked
        on a given element the data_array is updated by casting the value of "value" back to the node's
        data type.
    */
    template<typename AccessType = float, typename FuncType>
    static void for_each_in_range(Node &node, index_t start, index_t end, FuncType &&func);

    // TODO: Consider what we will do with material based fields, more tables?
    void flatten_single_domain(const Node &mesh, Node &output,
        index_t domain_id = 0, index_t vert_offset = 0, index_t elem_offset = 0) const;
    void flatten_many_domains(const Node &mesh, Node &output) const;

    std::string topology;
    std::vector<std::string> field_names;
    index_t default_dtype; // Must be either float32 or float64
    float64 float_fill_value;
    int64 int_fill_value;
    bool add_cell_centers;
    bool add_domain_info;
    bool add_vertex_locations;
};

//-----------------------------------------------------------------------------
MeshFlattener::MeshFlattener()
    : topology(""), field_names(), default_dtype(blueprint::mesh::utils::DEFAULT_FLOAT_DTYPE.id()),
    float_fill_value(0.), int_fill_value(0), add_cell_centers(true), add_domain_info(true),
    add_vertex_locations(true)
{
    // Construct with defaults
}

//-----------------------------------------------------------------------------
bool
MeshFlattener::set_options(const Node &options)
{
    bool ok = true;
    // "topology" sets the name of the topology to use when flattening.
    if(options.has_child("topology"))
    {
        const Node &topo_name = options["topology"];
        if(topo_name.dtype().is_string())
        {
            this->topology = topo_name.as_string();
        }
        else
        {
            ok = false;
            CONDUIT_ERROR("options[" << utils::log::quote("topology") << "] must be the string name of the desired topology");
        }
    }

    // "field_names" sets the desired fields to output in the table
    if(options.has_child("field_names"))
    {
        const Node &n_field_names = options["field_names"];
        if(n_field_names.dtype().is_list())
        {
            auto itr = n_field_names.children();
            while(itr.has_next())
            {
                const Node &n_name = itr.next();
                if(n_name.dtype().is_string())
                {
                    this->field_names.push_back(n_name.as_string());
                }
                else
                {
                    ok = false;
                    CONDUIT_ERROR("options[" << utils::log::quote("field_names") <<
                        "] entries must be the string names of desired output fields.");
                }
            }
        }
        else
        {
            ok = false;
            CONDUIT_ERROR("options[" << utils::log::quote("field_names")  <<
                "] must be a list containing the string names of desired output fields.");
        }
    }

    // "fill_value" is the default value for every cell in the table
    if(options.has_child("fill_value"))
    {
        const Node &n_fill_value = options["fill_value"];
        // Sets both to the same value
        if(n_fill_value.dtype().is_number())
        {
            this->float_fill_value = n_fill_value.to_float64();
            this->int_fill_value = n_fill_value.to_int64();
        }
        // Per type options
        else if(n_fill_value.dtype().is_object())
        {
            // First make sure the inputs are valid
            if(n_fill_value.has_child("int") && !n_fill_value["int"].dtype().is_number())
            {
                ok = false;
                CONDUIT_ERROR("options[" << utils::log::quote("fill_value/int") <<
                    "] must be a number.");
            }

            if(n_fill_value.has_child("float") && !n_fill_value["float"].dtype().is_number())
            {
                ok = false;
                CONDUIT_ERROR("options[" << utils::log::quote("fill_value/float") <<
                    "] must be a number.");
            }

            // Now set accordingly
            if(n_fill_value.has_child("int") && n_fill_value.has_child("float"))
            {
                this->float_fill_value = n_fill_value["float"].to_float64();
                this->int_fill_value = n_fill_value["int"].to_int64();
            }
            else if(n_fill_value.has_child("int"))
            {
                this->int_fill_value = n_fill_value["int"].to_int64();
            }
            else if(n_fill_value.has_child("float"))
            {
                this->float_fill_value = n_fill_value["float"].to_float64();
            }
        }
        else
        {
            ok = false;
            CONDUIT_ERROR("options[" << utils::log::quote("fill_value") <<
                "] must be a number.")
        }
    }

    // "add_domain_info", determines whether domain_id + vertex/element id should
    //   be added to the tables
    if(options.has_child("add_domain_info"))
    {
        const Node &n_add_domain_info = options["add_domain_info"];
        if(n_add_domain_info.dtype().is_number())
        {
            this->add_domain_info = options["add_domain_info"].to_int() != 0;
        }
        else
        {
            ok = false;
            CONDUIT_ERROR("options[" << utils::log::quote("add_domain_info") <<
                "] must be a number. It will be treated as a boolean (.to_int() != 0).");
        }
    }

    // "add_domain_info", determines whether domain_id + vertex/element id should
    //   be added to the tables
    if(options.has_child("add_cell_centers"))
    {
        const Node &n_add_cell_centers = options["add_cell_centers"];
        if(n_add_cell_centers.dtype().is_number())
        {
            this->add_cell_centers = options["add_cell_centers"].to_int() != 0;
        }
        else
        {
            ok = false;
            CONDUIT_ERROR("options[" << utils::log::quote("add_cell_centers") <<
                "] must be a number. It will be treated as a boolean (.to_int() != 0).");
        }
    }

    // "add_vertex_locations", boolean
    if(options.has_child("add_vertex_locations"))
    {
        const Node &n_add_vertex_locations = options["add_vertex_locations"];
        if(n_add_vertex_locations.dtype().is_number())
        {
            this->add_vertex_locations = options["add_vertex_locations"].to_int() != 0;
        }
        else
        {
            ok = false;
            CONDUIT_ERROR("options[" << utils::log::quote("add_vertex_locations") <<
                "] must be a number. It will be treated as a boolean (.to_int() != 0).");
        }
    }

    return ok;
}

//-----------------------------------------------------------------------------
void
MeshFlattener::execute(const Node &mesh, Node &output) const
{
    output.reset();

    if(blueprint::mesh::is_multi_domain(mesh))
    {
        flatten_many_domains(mesh, output);
    }
    else
    {
        Node temp;
        temp["domain_0"].set_external(mesh);
        flatten_many_domains(temp, output);
    }
}

//-----------------------------------------------------------------------------
const Node &
MeshFlattener::get_coordset(const Node &mesh) const
{
    const std::string cset_name = get_topology(mesh)["coordset"].as_string();
    return mesh["coordsets/" + cset_name];
}

//-----------------------------------------------------------------------------
const Node &
MeshFlattener::get_topology(const Node &mesh) const
{
    return (topology.empty() ? mesh["topologies"][0] : mesh["topologies/" + topology]);
}

//-----------------------------------------------------------------------------
void
MeshFlattener::get_fields_to_flatten(const Node &mesh, std::vector<std::string> &fields_to_flatten) const
{
    fields_to_flatten.clear();
    const Node &topo = get_topology(mesh);
    const std::string &topo_name = topo.name();
    const Node &fields = mesh["fields"];
    if(!this->field_names.empty())
    {
        // Always use the fields provided via options, always check for issues to report to the user.
        for(const std::string &field_name : this->field_names)
        {
            fields_to_flatten.push_back(field_name);

            const Node *field = fields.fetch_ptr(field_name);
            if(field)
            {
                if(!field->has_child("topology"))
                {
                    if(field->has_child("matset"))
                    {
                        CONDUIT_INFO("The field " << utils::log::quote(field->name()) <<
                            " appears to be material-dependent which is currently unsupported by mesh::flatten().");
                        continue;
                    }
                    else
                    {
                        CONDUIT_ERROR("The field " << utils::log::quote(field->name()) <<
                            " does not have an associated topology or matset.");
                        continue;
                    }
                }

                const std::string field_topo_name = field->child("topology").as_string();
                if(field_topo_name != topo_name)
                {
                    CONDUIT_INFO("The selected field " << utils::log::quote(field->name()) <<
                        " does not exist on the active mesh toplogy " << utils::log::quote(topo_name) <<
                        ".");
                }
            }
        }
    }
    else
    {
        // If no fields were provided in the options then use all fields associated with the active topology
        auto itr = fields.children();
        while(itr.has_next())
        {
            const Node &field = itr.next();
            if(!field.has_child("topology"))
            {
                if(field.has_child("matset"))
                {
                    CONDUIT_INFO("The field " + utils::log::quote(field.name()) +
                        " appears to be material-dependent which is currently unsupported by mesh::flatten().");
                    continue;
                }
                else
                {
                    CONDUIT_ERROR("The field " + utils::log::quote(field.name()) +
                        " does not have an associated topology or matset.");
                    continue;
                }
            }

            const std::string field_topo_name = field["topology"].as_string();
            if(field_topo_name == topo_name)
            {
                fields_to_flatten.push_back(field.name());
            }
        }
    }
}

//-----------------------------------------------------------------------------
index_t
MeshFlattener::determine_element_dtype(const Node &data) const
{
    // Turns out blueprint::mesh::utils has a function for this
    return blueprint::mesh::utils::find_widest_dtype(
        data, DataType(default_dtype, 1)).id();
}

//-----------------------------------------------------------------------------
void
MeshFlattener::default_initialize_column(Node &column) const
{
    const bool is_floating_point = column.dtype().is_floating_point();
    const bool is_integer = column.dtype().is_integer();
#define DEFAULT_INITIALIZE_COLUMN_IMPL(Type, default_value)\
{\
    Type *data_ptr = (Type*)column.element_ptr(0);\
    Type *const data_ptr_end = data_ptr + column.dtype().number_of_elements();\
    while(data_ptr != data_ptr_end) {\
        *data_ptr++ = static_cast<Type>(default_value);\
    }\
}
    if(is_floating_point)
    {
        switch(column.dtype().id())
        {
        case DataType::FLOAT32_ID:
            DEFAULT_INITIALIZE_COLUMN_IMPL(float32, this->float_fill_value)
            break;
        case DataType::FLOAT64_ID:
            DEFAULT_INITIALIZE_COLUMN_IMPL(float64, this->float_fill_value)
            break;
        default:
        {
            CONDUIT_ERROR("Unknown floating point type id " << column.dtype().id() << ".");
        }
        }
    }
    else if(is_integer)
    {
        switch(column.dtype().id())
        {
        // Signed types
        case DataType::INT8_ID:
            DEFAULT_INITIALIZE_COLUMN_IMPL(int8, this->int_fill_value)
            break;
        case DataType::INT16_ID:
            DEFAULT_INITIALIZE_COLUMN_IMPL(int16, this->int_fill_value)
            break;
        case DataType::INT32_ID:
            DEFAULT_INITIALIZE_COLUMN_IMPL(int32, this->int_fill_value)
            break;
        case DataType::INT64_ID:
            DEFAULT_INITIALIZE_COLUMN_IMPL(int64, this->int_fill_value)
            break;
        // Unsigned types
        case DataType::UINT8_ID:
            DEFAULT_INITIALIZE_COLUMN_IMPL(uint8, this->int_fill_value)
            break;
        case DataType::UINT16_ID:
            DEFAULT_INITIALIZE_COLUMN_IMPL(uint16, this->int_fill_value)
            break;
        case DataType::UINT32_ID:
            DEFAULT_INITIALIZE_COLUMN_IMPL(uint32, this->int_fill_value)
            break;
        case DataType::UINT64_ID:
            DEFAULT_INITIALIZE_COLUMN_IMPL(uint64, this->int_fill_value)
            break;
        default:
        {
            CONDUIT_ERROR("Unknown integer type id " << column.dtype().id() << ".");
        }
        }
    }
    else
    {
        CONDUIT_ERROR("Node with invalid type passed to default_initialize_column()." <<
            " Must be an integer or floating point number.");
    }
#undef DEFAULT_INITIALIZE_COLUMN_IMPL
}

//-----------------------------------------------------------------------------
void
MeshFlattener::allocate_column(Node &column, index_t nrows, index_t dtype_id,
        const Node *ref_node) const
{
    if(ref_node)
    {
        const DataType &dt = ref_node->dtype();
        if(dt.is_object() || dt.is_list())
        {
            auto itr = ref_node->children();
            while(itr.has_next())
            {
                const Node &n = itr.next();
                column[n.name()].set(DataType(dtype_id, nrows));
            }
        }
        else
        {
            column.set(DataType(dtype_id, nrows));
        }
    }
    else
    {
        column.set(DataType(dtype_id, nrows));
    }
}

//-----------------------------------------------------------------------------
template<typename SrcType, typename DestType>
void
MeshFlattener::append_data_array_impl2(const DataArray<SrcType> &src, DataArray<DestType> &dest, index_t offset)
{
    const index_t N = src.number_of_elements();
    for(index_t i = 0; i < N; i++)
    {
        dest[offset + i] = static_cast<DestType>(src[i]);
    }
}

//-----------------------------------------------------------------------------
template<typename SrcType>
void
MeshFlattener::append_data_array_impl1(const DataArray<SrcType> &src, Node &dest, index_t offset)
{
    const index_t dtype_id = dest.dtype().id();
    switch(dtype_id)
    {
    // Signed int types
    case DataType::INT8_ID:
    {
        DataArray<int8> value = dest.value();
        append_data_array_impl2(src, value, offset);
        break;
    }
    case DataType::INT16_ID:
    {
        DataArray<int16> value = dest.value();
        append_data_array_impl2(src, value, offset);
        break;
    }
    case DataType::INT32_ID:
    {
        DataArray<int32> value = dest.value();
        append_data_array_impl2(src, value, offset);
        break;
    }
    case DataType::INT64_ID:
    {
        DataArray<int64> value = dest.value();
        append_data_array_impl2(src, value, offset);
        break;
    }
    // Unsigned int types
    case DataType::UINT8_ID:
    {
        DataArray<uint8> value = dest.value();
        append_data_array_impl2(src, value, offset);
        break;
    }
    case DataType::UINT16_ID:
    {
        DataArray<uint16> value = dest.value();
        append_data_array_impl2(src, value, offset);
        break;
    }
    case DataType::UINT32_ID:
    {
        DataArray<uint32> value = dest.value();
        append_data_array_impl2(src, value, offset);
        break;
    }
    case DataType::UINT64_ID:
    {
        DataArray<uint64> value = dest.value();
        append_data_array_impl2(src, value, offset);
        break;
    }
    // Floating point types
    case DataType::FLOAT32_ID:
    {
        DataArray<float32> value = dest.value();
        append_data_array_impl2(src, value, offset);
        break;
    }
    case DataType::FLOAT64_ID:
    {
        DataArray<float64> value = dest.value();
        append_data_array_impl2(src, value, offset);
        break;
    }
    default:
    {
        CONDUIT_ERROR("Invalid data type passed to append_data");
    }
    }
}

//-----------------------------------------------------------------------------
void
MeshFlattener::append_data_array(const Node &src, Node &dest, index_t offset)
{
    const index_t dtype_id = src.dtype().id();
    switch(dtype_id)
    {
    // Signed int types
    case DataType::INT8_ID:
    {
        const DataArray<int8> value = src.value();
        append_data_array_impl1(value, dest, offset);
        break;
    }
    case DataType::INT16_ID:
    {
        const DataArray<int16> value = src.value();
        append_data_array_impl1(value, dest, offset);
        break;
    }
    case DataType::INT32_ID:
    {
        const DataArray<int32> value = src.value();
        append_data_array_impl1(value, dest, offset);
        break;
    }
    case DataType::INT64_ID:
    {
        const DataArray<int64> value = src.value();
        append_data_array_impl1(value, dest, offset);
        break;
    }
    // Unsigned int types
    case DataType::UINT8_ID:
    {
        const DataArray<uint8> value = src.value();
        append_data_array_impl1(value, dest, offset);
        break;
    }
    case DataType::UINT16_ID:
    {
        const DataArray<uint16> value = src.value();
        append_data_array_impl1(value, dest, offset);
        break;
    }
    case DataType::UINT32_ID:
    {
        const DataArray<uint32> value = src.value();
        append_data_array_impl1(value, dest, offset);
        break;
    }
    case DataType::UINT64_ID:
    {
        const DataArray<uint64> value = src.value();
        append_data_array_impl1(value, dest, offset);
        break;
    }
    // Floating point types
    case DataType::FLOAT32_ID:
    {
        const DataArray<float32> value = src.value();
        append_data_array_impl1(value, dest, offset);
        break;
    }
    case DataType::FLOAT64_ID:
    {
        const DataArray<float64> value = src.value();
        append_data_array_impl1(value, dest, offset);
        break;
    }
    default:
    {
        CONDUIT_ERROR("Invalid data type passed to append_data");
    }
    }
}

//-----------------------------------------------------------------------------
void
MeshFlattener::append_mc_data(const Node &src, Node &dest, index_t offset)
{
    auto itr = src.children();
    while(itr.has_next())
    {
        const Node &s = itr.next();
        if(!dest.has_child(s.name()))
        {
            CONDUIT_ERROR("Dest does not have a child named " << utils::log::quote(s.name()));
            continue;
        }
        append_data_array(s, dest[s.name()], offset);
    }
}

//-----------------------------------------------------------------------------
void
MeshFlattener::append_data(const Node &src, Node &dest, index_t offset)
{
    const DataType &src_dtype = src.dtype();
    if(src_dtype.is_list() || src_dtype.is_object())
    {
        append_mc_data(src, dest, offset);
    }
    else
    {
        append_data_array(src, dest, offset);
    }
}

//-----------------------------------------------------------------------------
template<typename AccessType, typename FuncType>
void
MeshFlattener::for_each_in_range(Node &node, index_t start, index_t end, FuncType &&func)
{
    const index_t dtype_id = node.dtype().id();
#define FOR_EACH_IN_RANGE_IMPL(ActualType)\
{\
    DataArray<ActualType> value = node.value();\
    index_t offset = start;\
    for(index_t i = 0; offset < end; i++, offset++)\
    {\
        AccessType temp = static_cast<AccessType>(value[offset]);\
        func(i, temp);\
        value[offset] = static_cast<AccessType>(temp);\
    }\
}
    switch(dtype_id)
    {
    // Signed int types
    case DataType::INT8_ID:
    {
        FOR_EACH_IN_RANGE_IMPL(int8)
        break;
    }
    case DataType::INT16_ID:
    {
        FOR_EACH_IN_RANGE_IMPL(int16)
        break;
        break;
    }
    case DataType::INT32_ID:
    {
        FOR_EACH_IN_RANGE_IMPL(int32)
        break;
    }
    case DataType::INT64_ID:
    {
        FOR_EACH_IN_RANGE_IMPL(int64)
        break;
    }
    // Unsigned int types
    case DataType::UINT8_ID:
    {
        FOR_EACH_IN_RANGE_IMPL(uint8)
        break;
    }
    case DataType::UINT16_ID:
    {
        FOR_EACH_IN_RANGE_IMPL(uint16)
        break;
    }
    case DataType::UINT32_ID:
    {
        FOR_EACH_IN_RANGE_IMPL(uint32)
        break;
    }
    case DataType::UINT64_ID:
    {
        FOR_EACH_IN_RANGE_IMPL(uint64)
        break;
    }
    // Floating point types
    case DataType::FLOAT32_ID:
    {
        FOR_EACH_IN_RANGE_IMPL(float32)
        break;
    }
    case DataType::FLOAT64_ID:
    {
        FOR_EACH_IN_RANGE_IMPL(float64)
        break;
    }
    default:
    {
        CONDUIT_ERROR("Invalid data type passed to for_each_in_range");
    }
    }
}

//-----------------------------------------------------------------------------
void
MeshFlattener::flatten_single_domain(const Node &mesh, Node &output,
        index_t domain_id, index_t vert_offset, index_t elem_offset) const
{
    const conduit::Node &topo = get_topology(mesh);
    const conduit::Node &cset = get_coordset(mesh);
    const index_t nelems = blueprint::mesh::topology::length(topo);
    const index_t nverts = blueprint::mesh::coordset::length(cset);
    Node &vert_table = output["vertex_data"];
    Node &elem_table = output["element_data"];

    // Add coordset data to table
    if(this->add_vertex_locations)
    {
        const std::string &cset_type = cset["type"].as_string();
        Node &cset_output = vert_table["values"][0];
        if(cset_type == "uniform" || cset_type == "rectilinear")
        {
            Node temp;
            if(cset_type == "uniform")
            {
                blueprint::mesh::coordset::uniform::to_explicit(cset, temp);
            }
            else
            {
                blueprint::mesh::coordset::rectilinear::to_explicit(cset, temp);
            }
            append_data(temp["values"], cset_output, vert_offset);
        }
        else if(cset_type == "explicit")
        {
            append_data(cset["values"], cset_output, vert_offset);
        }
    }

    // Add cell center information to element table
    if(this->add_cell_centers)
    {
        // TODO
    }

    // Add domain_id + vertex/element ids to their respective tables
    if(this->add_domain_info)
    {
        const auto set_domain_id = [domain_id](index_t, index_t &value) {
            value = domain_id;
        };
        const auto set_id = [](index_t i, index_t &value) {
            value = i;
        };

        // Vertex table values
        for_each_in_range<index_t>(vert_table["values/domain_id"], vert_offset,
            vert_offset + nverts, set_domain_id);
        for_each_in_range<index_t>(vert_table["values/vertex_id"], vert_offset,
            vert_offset + nverts, set_id);

        // Element table values
        for_each_in_range<index_t>(elem_table["values/domain_id"], elem_offset,
            elem_offset + nelems, set_domain_id);
        for_each_in_range<index_t>(elem_table["values/element_id"], elem_offset,
            elem_offset + nelems, set_id);
    }

    // Add fields to their respective tables
    {
        const Node &fields = mesh["fields"];
        std::vector<std::string> fields_to_flatten;
        get_fields_to_flatten(mesh, fields_to_flatten);

        for(const std::string &field_name : fields_to_flatten)
        {
            const Node &field = fields[field_name];
            const std::string association = field["association"].as_string();
            if(association == "vertex")
            {
                append_data(field["values"], vert_table["values/" + field.name()], vert_offset);
            }
            else if(association == "element")
            {
                append_data(field["values"], elem_table["values/" + field.name()], elem_offset);
            }
        }
    }
}

//-----------------------------------------------------------------------------
void
MeshFlattener::flatten_many_domains(const Node &mesh, Node &output) const
{
    std::vector<index_t> verts_per_domain;
    std::vector<index_t> elems_per_domain;
    std::vector<index_t> domain_ids;
    std::vector<std::string> axes;
    std::string cset_name;
    index_t dimension = 0;
    index_t coord_type = DataType::EMPTY_ID;

    // Collect information about the mesh
    const index_t ndomains = mesh.number_of_children();
    {
        bool first = true;
        for(index_t i = 0; i < ndomains; i++)
        {
            const Node &domain = mesh[i];
            const Node &dom_topo = get_topology(domain);
            const Node &dom_cset = get_coordset(domain);
            elems_per_domain.push_back(blueprint::mesh::topology::length(dom_topo));
            verts_per_domain.push_back(blueprint::mesh::coordset::length(dom_cset));

            if(domain.has_child("state") && domain.has_child("domain_id")
                && domain["state/domain_id"].dtype().is_integer())
            {
                domain_ids.push_back(domain["state/domain_id"].to_index_t());
            }

            if(first)
            {
                first = false;
                dimension = blueprint::mesh::coordset::dims(dom_cset);
                axes = blueprint::mesh::utils::coordset::axes(dom_cset);
                const std::string cset_type_name = dom_cset["type"].as_string();
                if(cset_type_name == "uniform" || cset_type_name == "rectilinear")
                {
                    coord_type = determine_element_dtype(dom_cset);
                }
                else if(cset_type_name == "explicit")
                {
                    coord_type = determine_element_dtype(dom_cset["values"]);
                }
                cset_name = dom_cset.name();
            }
        }

        // Check if each domain had a proper domain_id, if not replace domain
        //  ids with their index into the top level mesh
        if((index_t)domain_ids.size() != ndomains)
        {
            domain_ids.resize(ndomains);
            for(index_t i = 0; i < ndomains; i++)
            {
                domain_ids[i] = i;
            }
        }
    }

    // Calculate totals, will be used for number of rows
    index_t nverts = 0;
    index_t nelems = 0;
    for(const auto dom_verts : verts_per_domain) nverts += dom_verts;
    for(const auto dom_elems : elems_per_domain) nelems += dom_elems;
#ifdef DEBUG_MESH_FLATTEN
    std::cout << "Total number of verticies: " << nverts << "\n"
        << "Total number of elements: " << nelems << std::endl;
#endif

    // Make allocations
    Node &vertex_table = output["vertex_data"];
    Node &element_table = output["element_data"];

    // Allocate cset output
    if(this->add_vertex_locations)
    {
        const std::string cset_output_path = "values/" + cset_name;
        for(index_t d = 0; d < dimension; d++)
        {
            allocate_column(vertex_table[cset_output_path + "/" + axes[d]],
                nverts, coord_type);
        }
    }

    // Allocate cell center output
    if(this->add_cell_centers)
    {
        const std::string elem_center_output_path = "values/element_centers";
        for(index_t d = 0; d < dimension; d++)
        {
            allocate_column(element_table[elem_center_output_path + "/" + axes[d]],
                nelems, coord_type);
        }
    }

    // Add domain information to each table
    if(this->add_domain_info)
    {
        const DataType dt_index_t = DataType::index_t(1);
        allocate_column(vertex_table["values/domain_id"], nverts, dt_index_t.id());
        allocate_column(vertex_table["values/vertex_id"], nverts, dt_index_t.id());
        allocate_column(element_table["values/domain_id"], nelems, dt_index_t.id());
        allocate_column(element_table["values/element_id"], nelems, dt_index_t.id());
    }

    // Allocate fields output
    // TODO: Find the field if it doesn't exist in domain 0
    {
        std::vector<std::string> fields_to_flatten;
        get_fields_to_flatten(mesh[0], fields_to_flatten);
        const Node &temp_fields = mesh[0]["fields"];
        for(const std::string &field_name : fields_to_flatten)
        {
            const Node *field = temp_fields.fetch_ptr(field_name);
            if(field)
            {
                // TODO: Support material-based fields
                const Node &field_values = field->child("values");
                const index_t elem_dtype_id = determine_element_dtype(field_values);
                const std::string assoc = field->child("association").as_string();
                if(assoc == "vertex")
                {
                    allocate_column(vertex_table["values/" + field_name], nverts, elem_dtype_id, &field_values);
                }
                else if(assoc == "element")
                {
                    allocate_column(element_table["values/" + field_name], nelems, elem_dtype_id, &field_values);
                }
            }
            else
            {
                CONDUIT_ERROR("Cannot allocate table entry for " << utils::log::quote(field_name) <<
                    " because it doesn't exist on domain 0.");
            }
        }
    }

    index_t vert_offset = 0, elem_offset = 0;
    for(index_t i = 0; i < ndomains; i++)
    {
        flatten_single_domain(mesh[i], output, domain_ids[i], vert_offset, elem_offset);
        vert_offset += verts_per_domain[i];
        elem_offset += elems_per_domain[i];
    }

    // TODO: Clear material tables if they end up existing
    if(output["vertex_data"].dtype().is_empty())
    {
        output.remove_child("vertex_data");
    }

    if(output["element_data"].dtype().is_empty())
    {
        output.remove_child("element_data");
    }
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
// -- begin conduit::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{

void flatten(const conduit::Node &mesh,
             const conduit::Node &options,
             conduit::Node &output)
{
    output.reset();

    MeshFlattener do_flatten;
    do_flatten.set_options(options);
    do_flatten.execute(mesh, output);
    return;
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
// -- end conduit --
//-----------------------------------------------------------------------------
