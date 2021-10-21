// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_flatten.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh_flatten.hpp"

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <array>
#include <iostream>
#include <set>
#include <string>
#include <vector>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_blueprint_mesh_utils_iterate_elements.hpp"
#include "conduit_log.hpp"

using ::conduit::utils::log::quote;

// Debug macros
#define DEBUG_MESH_FLATTEN 1

#ifdef DEBUG_MESH_FLATTEN
#define DEBUG_PRINT(stream)\
do {\
    std::cout << stream;\
} while(0)
#else
#define DEBUG_PRINT(msg)
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
// -- begin conduit::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{

//-----------------------------------------------------------------------------
// -- begin conduit::mesh::utils --
//-----------------------------------------------------------------------------
namespace utils
{

//-----------------------------------------------------------------------------
template<typename SrcType, typename DestType>
static void
append_data_array_impl2(const DataArray<SrcType> &src,
    DataArray<DestType> &dest, index_t offset, index_t nelems)
{
    index_t off = offset;
    for(index_t i = 0; i < nelems; i++)
    {
        dest[off++] = static_cast<DestType>(src[i]);
    }
}

//-----------------------------------------------------------------------------
template<typename SrcType>
static void
append_data_array_impl1(const DataArray<SrcType> &src, Node &dest,
    index_t offset, index_t nelems)
{
    const index_t dtype_id = dest.dtype().id();
    switch(dtype_id)
    {
    // Signed int types
    case DataType::INT8_ID:
    {
        DataArray<int8> value = dest.value();
        append_data_array_impl2(src, value, offset, nelems);
        break;
    }
    case DataType::INT16_ID:
    {
        DataArray<int16> value = dest.value();
        append_data_array_impl2(src, value, offset, nelems);
        break;
    }
    case DataType::INT32_ID:
    {
        DataArray<int32> value = dest.value();
        append_data_array_impl2(src, value, offset, nelems);
        break;
    }
    case DataType::INT64_ID:
    {
        DataArray<int64> value = dest.value();
        append_data_array_impl2(src, value, offset, nelems);
        break;
    }
    // Unsigned int types
    case DataType::UINT8_ID:
    {
        DataArray<uint8> value = dest.value();
        append_data_array_impl2(src, value, offset, nelems);
        break;
    }
    case DataType::UINT16_ID:
    {
        DataArray<uint16> value = dest.value();
        append_data_array_impl2(src, value, offset, nelems);
        break;
    }
    case DataType::UINT32_ID:
    {
        DataArray<uint32> value = dest.value();
        append_data_array_impl2(src, value, offset, nelems);
        break;
    }
    case DataType::UINT64_ID:
    {
        DataArray<uint64> value = dest.value();
        append_data_array_impl2(src, value, offset, nelems);
        break;
    }
    // Floating point types
    case DataType::FLOAT32_ID:
    {
        DataArray<float32> value = dest.value();
        append_data_array_impl2(src, value, offset, nelems);
        break;
    }
    case DataType::FLOAT64_ID:
    {
        DataArray<float64> value = dest.value();
        append_data_array_impl2(src, value, offset, nelems);
        break;
    }
    default:
    {
        CONDUIT_ERROR("Invalid data type passed to append_data");
    }
    }
}

//-----------------------------------------------------------------------------
static void
append_data_array(const Node &src, Node &dest,
    index_t offset, index_t nelems)
{
    DEBUG_PRINT("utils::append_data_array"
        << "\n  src.dtype().number_of_elements(): " << src.dtype().number_of_elements()
        << "\n  dest.dtype().number_of_elements(): " << dest.dtype().number_of_elements()
        << "\n  offset: " << offset << std::endl);
    if(offset + nelems > dest.dtype().number_of_elements())
    {
        CONDUIT_ERROR("Invalid arguments passed to utils::append_data()."
            << "  Trying copy " << nelems << " elements into an array of "
            << dest.dtype().number_of_elements() << " elements starting at "
            << "offset " << offset << ". " << offset << " + " << nelems
            << " > " << dest.dtype().number_of_elements() << ".");
        return;
    }

    const index_t dtype_id = src.dtype().id();
    switch(dtype_id)
    {
    // Signed int types
    case DataType::INT8_ID:
    {
        const DataArray<int8> value = src.value();
        append_data_array_impl1(value, dest, offset, nelems);
        break;
    }
    case DataType::INT16_ID:
    {
        const DataArray<int16> value = src.value();
        append_data_array_impl1(value, dest, offset, nelems);
        break;
    }
    case DataType::INT32_ID:
    {
        const DataArray<int32> value = src.value();
        append_data_array_impl1(value, dest, offset, nelems);
        break;
    }
    case DataType::INT64_ID:
    {
        const DataArray<int64> value = src.value();
        append_data_array_impl1(value, dest, offset, nelems);
        break;
    }
    // Unsigned int types
    case DataType::UINT8_ID:
    {
        const DataArray<uint8> value = src.value();
        append_data_array_impl1(value, dest, offset, nelems);
        break;
    }
    case DataType::UINT16_ID:
    {
        const DataArray<uint16> value = src.value();
        append_data_array_impl1(value, dest, offset, nelems);
        break;
    }
    case DataType::UINT32_ID:
    {
        const DataArray<uint32> value = src.value();
        append_data_array_impl1(value, dest, offset, nelems);
        break;
    }
    case DataType::UINT64_ID:
    {
        const DataArray<uint64> value = src.value();
        append_data_array_impl1(value, dest, offset, nelems);
        break;
    }
    // Floating point types
    case DataType::FLOAT32_ID:
    {
        const DataArray<float32> value = src.value();
        append_data_array_impl1(value, dest, offset, nelems);
        break;
    }
    case DataType::FLOAT64_ID:
    {
        const DataArray<float64> value = src.value();
        append_data_array_impl1(value, dest, offset, nelems);
        break;
    }
    default:
    {
        CONDUIT_ERROR("Invalid data type passed to append_data");
    }
    }
}

//-----------------------------------------------------------------------------
static void
append_mc_data(const Node &src, Node &dest,
    index_t offset, index_t nelems)
{
    auto itr = src.children();
    while(itr.has_next())
    {
        const Node &s = itr.next();
        if(!dest.has_child(s.name()))
        {
            CONDUIT_ERROR("Dest does not have a child named " << quote(s.name()));
            continue;
        }
        append_data_array(s, dest[s.name()], offset, nelems);
    }
}

//-----------------------------------------------------------------------------
// NOTE: Public API
void
append_data(const Node &src, Node &dest, index_t offset,
    index_t nelems)
{
    const DataType &src_dtype = src.dtype();
    if(src_dtype.is_list() || src_dtype.is_object())
    {
        append_mc_data(src, dest, offset, nelems);
    }
    else
    {
        append_data_array(src, dest, offset, nelems);
    }
}

//-----------------------------------------------------------------------------
template<typename CsetType, typename OutputType>
static void
generate_element_centers_impl(const Node &topo, const index_t dimension,
    const DataArray<CsetType> *cset_values, DataArray<OutputType> *output_values,
    const index_t offset)
{
    using conduit::blueprint::mesh::utils::topology::entity;
    index_t output_idx = offset;
    utils::topology::iterate_elements(topo, [&](const entity &e) {
        const index_t nids = static_cast<index_t>(e.element_ids.size());
        for(index_t d = 0; d < dimension; d++)
        {
            OutputType sum = 0;
            for(index_t i = 0; i < nids; i++)
            {
                sum += static_cast<OutputType>(cset_values[d][e.element_ids[i]]);
            }
            output_values[d][output_idx] = sum / static_cast<OutputType>(nids);
        }
        output_idx++;
    });
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
MeshFlattener::MeshFlattener()
    : topology(""), field_names(),
    default_dtype(blueprint::mesh::utils::DEFAULT_FLOAT_DTYPE.id()),
    float_fill_value(0.), int_fill_value(0), add_cell_centers(true),
    add_domain_info(true), add_vertex_locations(true)
{
    // Construct with defaults
}

//-----------------------------------------------------------------------------
MeshFlattener::~MeshFlattener()
{
    // Nothing special
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
            CONDUIT_ERROR("options[" << quote("topology") << "] must be the string name of the desired topology");
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
                    CONDUIT_ERROR("options[" << quote("field_names") <<
                        "] entries must be the string names of desired output fields.");
                }
            }
        }
        else
        {
            ok = false;
            CONDUIT_ERROR("options[" << quote("field_names")  <<
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
                CONDUIT_ERROR("options[" << quote("fill_value/int") <<
                    "] must be a number.");
            }

            if(n_fill_value.has_child("float") && !n_fill_value["float"].dtype().is_number())
            {
                ok = false;
                CONDUIT_ERROR("options[" << quote("fill_value/float") <<
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
            CONDUIT_ERROR("options[" << quote("fill_value") <<
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
            CONDUIT_ERROR("options[" << quote("add_domain_info") <<
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
            CONDUIT_ERROR("options[" << quote("add_cell_centers") <<
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
            CONDUIT_ERROR("options[" << quote("add_vertex_locations") <<
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
MeshFlattener::coordset_to_explicit(const Node &cset, Node &out_cset) const
{
    const std::string &cset_type = cset["type"].as_string();
    if(cset_type == "uniform")
    {
        blueprint::mesh::coordset::uniform::to_explicit(cset, out_cset);
    }
    else if(cset_type == "rectilinear")
    {
        blueprint::mesh::coordset::rectilinear::to_explicit(cset, out_cset);
    }
    else if(cset_type == "explicit")
    {
        out_cset.set_external(cset);
    }
    else
    {
        CONDUIT_ERROR("Unsupported coordset type passed to MeshFlattener::coordset_to_explicit()");
    }
}

//-----------------------------------------------------------------------------
bool
MeshFlattener::check_field_supported(const Node &field,
    const std::string &topo_name, bool report_issues) const
{
    const std::string field_name = field.name();
    if(!field.has_child("topology"))
    {
        if(field.has_child("matset"))
        {
            if(report_issues)
            {
                CONDUIT_INFO("The field " << quote(field_name) <<
                    " appears to be material-dependent which is currently unsupported by mesh::flatten().");
            }
            return false;
        }
        else
        {
            if(report_issues)
            {
                CONDUIT_ERROR("The field " << quote(field_name) <<
                    " does not have an associated topology or matset.");
            }
            return false;
        }
    }

    const std::string field_topo_name = field.child("topology").as_string();
    if(field_topo_name != topo_name)
    {
        if(report_issues)
        {
            CONDUIT_INFO("The selected field " << quote(field_name) <<
                " does not exist on the active mesh toplogy " << quote(topo_name) <<
                ".");
        }
        return false;
    }

    if(!field.has_child("association"))
    {
        if(report_issues)
        {
            CONDUIT_INFO("The selected field " << quote(field_name) <<
                " is not associated with verticies or elements. It will not be present in the output.");
        }
        return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
void
MeshFlattener::get_fields_to_flatten(const Node &mesh,
    const std::string &topo_name,
    std::vector<std::string> &fields_to_flatten) const
{
    fields_to_flatten.clear();
    const index_t ndomains = mesh.number_of_children();
    if(!this->field_names.empty())
    {
        // Find each of the given fields, keep the order given by the user
        std::vector<std::string> field_stack;
        for(auto itr = this->field_names.rbegin(); itr != this->field_names.rend(); itr++)
        {
            field_stack.push_back(*itr);
        }

        while(!field_stack.empty())
        {
            const std::string field_name = field_stack.back();
            field_stack.pop_back();
            bool found = false;
            for(index_t i = 0; i < ndomains; i++)
            {
                const Node &domain = mesh[i];
                const Node &fields = domain["fields"];
                const Node *field = fields.fetch_ptr(field_name);
                if(field)
                {
                    if(check_field_supported(*field, topo_name))
                    {
                        const std::string association = field->child("association").as_string();
                        fields_to_flatten.push_back(field_name);
                        found = true;
                        break;
                    }
                }
            }

            if(!found)
            {
                CONDUIT_INFO("Field name " << field_name << " was provided as an"
                    << " option to mesh::flatten(), but it does not exist on any of"
                    << " the mesh domains. It will not be present in the output.")
            }
        }
    }
    else
    {
        // If no fields were provided in the options then use all fields associated with the active topology
        std::set<std::string> field_set;
        for(index_t i = 0; i < ndomains; i++)
        {
            const Node &domain = mesh[i];
            const Node &fields = domain["fields"];
            auto itr = fields.children();
            while(itr.has_next())
            {
                const Node &field = itr.next();
                const std::string field_name = field.name();
                if(field_set.count(field_name) == 0)
                {
                    if(check_field_supported(field, topo_name))
                    {
                        const std::string association = field["association"].as_string();
                        field_set.insert(field_name); // Make sure we don't add this field again
                        fields_to_flatten.push_back(field_name);
                    }
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
const Node *
MeshFlattener::get_reference_field(const Node &mesh,
    const std::string &topo_name,
    const std::string &field_name) const
{
    const index_t ndomains = mesh.number_of_children();
    for(index_t i = 0; i < ndomains; i++)
    {
        const Node &fields = mesh[i]["fields"];
        if(fields.has_child(field_name))
        {
            const Node &field = fields[field_name];
            if(check_field_supported(field, topo_name, false))
            {
                return &field;
            }
        }
    }
    return nullptr;
}

//-----------------------------------------------------------------------------
index_t
MeshFlattener::determine_element_dtype(const Node &data) const
{
    // Turns out blueprint::mesh::utils has a function for this
    const std::vector<DataType> default_dtypes{
        DataType::float32(),
        DataType::int32(),
        DataType::uint32()
    };
    return blueprint::mesh::utils::find_widest_dtype(
        data, default_dtypes).id();
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
    // Nodes calloc their data, no need to re-set everything to 0.
    // Question: Is this a guarantee of Node::set(DataType)?
    if(is_floating_point && this->float_fill_value != 0.)
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
    else if(is_integer && this->int_fill_value != 0)
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
    else if(!is_floating_point && !is_integer)
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
                default_initialize_column(column[n.name()]);
            }
        }
        else
        {
            column.set(DataType(dtype_id, nrows));
            default_initialize_column(column);
        }
    }
    else
    {
        column.set(DataType(dtype_id, nrows));
        default_initialize_column(column);
    }
}

//-----------------------------------------------------------------------------
void
MeshFlattener::generate_element_centers(const Node &topo,
    const Node &explicit_cset, Node &output, index_t offset) const
{
    using namespace blueprint::mesh::utils::topology;
    const Node &n_cset_values = explicit_cset["values"];
    const index_t dimension = n_cset_values.number_of_children();
    if(!output[0].dtype().is_floating_point())
    {
        CONDUIT_ERROR("Cell center output DataType must be floating point.");
        return;
    }

    // Figure out the type of the input coordinates
    // Need to be sure that they are all the same type
    const index_t cset_type = determine_element_dtype(n_cset_values);
    bool all_same = true;
    for(index_t i = 0; i < dimension; i++)
    {
        if(cset_type != n_cset_values[i].dtype().id())
        {
            all_same = false;
        }
    }

    Node temp;
    if(all_same)
    {
        // Zero copy
        temp.set_external(n_cset_values);
    }
    else
    {
        // Need to convert coords to all the same type
        for(index_t i = 0; i < dimension; i++)
        {
            const Node &n_comp = n_cset_values[i];
            // Only convert what we need to convert
            if(n_comp.dtype().id() != cset_type)
            {
                n_cset_values.to_data_type(cset_type, temp[n_comp.name()]);
            }
            else
            {
                temp[n_comp.name()].set_external(n_comp);
            }
        }
    }

#define GENERATE_ELEMENT_CENTERS_CSET_TYPE(CsetType, OutputType)\
{\
    const std::array<const DataArray<CsetType>, 3> cset_values{\
        (dimension > 0) ? n_cset_values[0].value() : DataArray<CsetType>(),\
        (dimension > 1) ? n_cset_values[1].value() : DataArray<CsetType>(),\
        (dimension > 2) ? n_cset_values[2].value() : DataArray<CsetType>(),\
    };\
    utils::generate_element_centers_impl<CsetType, OutputType>(topo, dimension,\
        cset_values.data(), output_values.data(), offset);\
}

    // Invoke the algorithm with the proper types
    // Output is allocated by us so we know each component has the same type.
    if(output[0].dtype().is_float32())
    {
        std::array<DataArray<float32>, 3> output_values{
            (dimension > 0) ? output[0].value() : DataArray<float32>(),
            (dimension > 1) ? output[1].value() : DataArray<float32>(),
            (dimension > 2) ? output[2].value() : DataArray<float32>(),
        };

        switch(cset_type)
        {
        // Signed int types
        case DataType::INT8_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(int8, float32)
            break;
        case DataType::INT16_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(int16, float32)
            break;
        case DataType::INT32_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(int32, float32)
            break;
        case DataType::INT64_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(int64, float32)
            break;
        // Unsigned int types
        case DataType::UINT8_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(uint8, float32)
            break;
        case DataType::UINT16_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(uint16, float32)
            break;
        case DataType::UINT32_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(uint32, float32)
            break;
        case DataType::UINT64_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(uint64, float32)
            break;
        // Floating point types
        case DataType::FLOAT32_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(float32, float32)
            break;
        case DataType::FLOAT64_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(float64, float32)
            break;
        default:
        {
            CONDUIT_ERROR("Invalid coordinate data type passed to generate_element_centers.");
        }
        }
    }
    else
    {
        std::array<DataArray<float64>, 3> output_values{
            (dimension > 0) ? output[0].value() : DataArray<float64>(),
            (dimension > 1) ? output[1].value() : DataArray<float64>(),
            (dimension > 2) ? output[2].value() : DataArray<float64>(),
        };

        switch(cset_type)
        {
        // Signed int types
        case DataType::INT8_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(int8, float64)
            break;
        case DataType::INT16_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(int16, float64)
            break;
        case DataType::INT32_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(int32, float64)
            break;
        case DataType::INT64_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(int64, float64)
            break;
        // Unsigned int types
        case DataType::UINT8_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(uint8, float64)
            break;
        case DataType::UINT16_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(uint16, float64)
            break;
        case DataType::UINT32_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(uint32, float64)
            break;
        case DataType::UINT64_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(uint64, float64)
            break;
        // Floating point types
        case DataType::FLOAT32_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(float32, float64)
            break;
        case DataType::FLOAT64_ID:
            GENERATE_ELEMENT_CENTERS_CSET_TYPE(float64, float64)
            break;
        default:
        {
            CONDUIT_ERROR("Invalid coordinate data type passed to generate_element_centers.");
        }
        }
    }
#undef GENERATE_ELEMENT_CENTERS_CSET_TYPE
}

//-----------------------------------------------------------------------------
void
MeshFlattener::flatten_single_domain(const Node &mesh, Node &output,
    const std::vector<std::string> &fields_to_flatten, index_t domain_id,
    index_t vert_offset, index_t elem_offset) const
{
    const conduit::Node &topo = get_topology(mesh);
    const conduit::Node &cset = get_coordset(mesh);
    const index_t nelems = blueprint::mesh::topology::length(topo);
    const index_t nverts = blueprint::mesh::coordset::length(cset);
    Node &vert_table = output["vertex_data"];
    Node &elem_table = output["element_data"];

    // Used to store an explicit version of the current cset.
    //  Either holds ownership of data created by coordset::<type>::to_explicit
    //  OR points to the actual coordset data via set_external.
    //  Should be treated as READ ONLY.
    // NOTE: Is there a better way for me todo this? Don't want to have
    //  to copy an entire explicit coordset if it's already explicit.
    Node explicit_cset;

    // Add coordset data to table
    if(this->add_vertex_locations)
    {
        coordset_to_explicit(cset, explicit_cset);
        Node &cset_output = vert_table["values"][0];
        utils::append_data(explicit_cset["values"], cset_output, vert_offset, nverts);
    }

    // Add cell center information to element table
    if(this->add_cell_centers)
    {
        if(explicit_cset.dtype().is_empty())
        {
            coordset_to_explicit(cset, explicit_cset);
        }
        Node &element_center_output = elem_table["values/element_centers"];
        generate_element_centers(topo, explicit_cset, element_center_output, elem_offset);
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
        utils::for_each_in_range<index_t>(vert_table["values/domain_id"], vert_offset,
            vert_offset + nverts, set_domain_id);
        utils::for_each_in_range<index_t>(vert_table["values/vertex_id"], vert_offset,
            vert_offset + nverts, set_id);

        // Element table values
        utils::for_each_in_range<index_t>(elem_table["values/domain_id"], elem_offset,
            elem_offset + nelems, set_domain_id);
        utils::for_each_in_range<index_t>(elem_table["values/element_id"], elem_offset,
            elem_offset + nelems, set_id);
    }

    // Add fields to their respective tables
    const Node &fields = mesh["fields"];
#ifdef DEBUG_MESH_FLATTEN
    std::cout << "Domain[" << domain_id << "] fields to flatten:\n";
    for(index_t i = 0; i < (index_t)fields_to_flatten.size(); i++)
    {
        const std::string &field_name = fields_to_flatten[i];
        std::cout << "  " << quote(field_name) <<
            (i < (index_t)(fields_to_flatten.size() - 1) ? "\n" : "");
    }
    std::cout << std::endl;
#endif
    for(const std::string &field_name : fields_to_flatten)
    {
        const Node *field = fields.fetch_ptr(field_name);
        if(field)
        {
            const std::string association = field->child("association").as_string();
            const Node &field_values = field->child("values");
            if(association == "vertex")
            {
                utils::append_data(field_values, vert_table["values/" + field_name],
                    vert_offset, nverts);
            }
            else if(association == "element")
            {
                utils::append_data(field_values, elem_table["values/" + field_name],
                    elem_offset, nelems);
            }
        }
#ifdef DEBUG_MESH_FLATTEN
        else
        {
            std::cout << "  field " << quote(field_name)
                << "does not exist on domain!" << std::endl;
        }
#endif
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
                nelems, this->default_dtype);
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
    std::vector<std::string> fields_to_flatten;
    get_fields_to_flatten(mesh, get_topology(mesh[0]).name(), fields_to_flatten);
    for(const std::string &field_name : fields_to_flatten)
    {
        const Node *ref_field = get_reference_field(mesh, get_topology(mesh[0]).name(),
            field_name);
        if(ref_field)
        {
            const std::string assoc = ref_field->child("association").as_string();
            const Node &field_values = ref_field->child("values");
            const index_t elem_dtype_id = determine_element_dtype(field_values);
            if(assoc == "vertex")
            {
                allocate_column(vertex_table["values/" + field_name], nverts, elem_dtype_id, &field_values);
            }
            else if(assoc == "element")
            {
                allocate_column(element_table["values/" + field_name], nelems, elem_dtype_id, &field_values);
            }
            else
            {
                CONDUIT_ERROR("Unknown field association type - " << assoc);
            }
        }
        else
        {
            // NOTE: Must be a logic error in get_reference_field or get_fields_to_flatten,
            //  this should never happen.
            CONDUIT_ERROR("Unable to find reference field for " << field_name);
        }
    }

    DEBUG_PRINT("Table allocation:" << output.schema().to_json() << std::endl);

    // Flatten each domain
    index_t vert_offset = 0, elem_offset = 0;
    for(index_t i = 0; i < ndomains; i++)
    {
        flatten_single_domain(mesh[i], output, fields_to_flatten, domain_ids[i], vert_offset, elem_offset);
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
