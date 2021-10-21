// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_flatten.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_FLATTEN_HPP
#define CONDUIT_BLUEPRINT_MESH_FLATTEN_HPP

// Internal utility header

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <vector>
#include <string>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_exports.h"

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

/**
@brief Reads the data stored in src and copies it into dest starting
    at the given offset into dest's memory.
@note Data from src is read as its native type and static_cast is used
    to convert each element into dest's memory.
@param src Input data, must be a leaf node or mcarray
@param dest Output node, must have allocated enough memory to store
    offset + src.dtype().number_of_elements() elements.
@param offset An offset into dest's memory to start copying.
@param nelems The number of elements to copy from src to dest.
*/
void CONDUIT_BLUEPRINT_API append_data(const Node &src, Node &dest,
                                       index_t offset, index_t nelems);

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
void for_each_in_range(Node &node, index_t start, index_t end, FuncType &&func);

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils --
//-----------------------------------------------------------------------------

class CONDUIT_BLUEPRINT_API MeshFlattener
{
public:
    MeshFlattener();
    virtual ~MeshFlattener();

    bool set_options(const Node &options);

    void execute(const Node &mesh, Node &output) const;
protected:
    const Node &get_coordset(const Node &mesh) const;
    const Node &get_topology(const Node &mesh) const;

    /**
    @brief Inspects the type of the given coordset and calls
        the correct to_explicit function. If the given coordset
        is already explicit then out_cset is just - out_cset.set_external(cset).
    @note The data returned through the parameter out_cset is READ ONLY.
    @param cset The input coordset.
    @param[out] values READ ONLY "explicit" coordset.
    */
    void coordset_to_explicit(const Node &cset, Node &out_cset) const;

    /**
    @brief Checks if the given field node is supported by the flatten
        operation. Currently ensures the field exists on the given
        topology and checks that the association field exists
        FUTURE: Add necessary logic for supporting grid_functions
        and matsets.
    */
    bool check_field_supported(const Node &field,
        const std::string &topo_name,
        bool report_issues = true) const;

    /**
    @brief Inspects the given multi-domain mesh and provides a list of
        fields to include in the output along with their associations.
    @param mesh
    */
    void get_fields_to_flatten(const Node &mesh, const std::string &topo_name,
        std::vector<std::string> &fields_to_flatten) const;

    /**
    @brief Inspects the given multi-domain mesh and provides the a reference
        to the first valid field found with the given name.
    @param mesh Mesh node containing all local domains
    @param topo_name Used to check that field exists on the correct topology
    @param field_name The name of the field to find.
    @return Returns a pointer to the first valid field with the given name,
        if none exist then nullptr is returned.
    */
    const Node *get_reference_field(const Node &mesh,
        const std::string &topo_name,
        const std::string &field_name) const;

    index_t determine_element_dtype(const Node &data) const;
    void default_initialize_column(Node &column) const;
    void allocate_column(Node &column, index_t nrows, index_t dtype_id,
        const Node *ref_node = nullptr) const;

    void generate_element_centers(const Node &topo, const Node &explicit_cset,
        Node &output, index_t offset) const;

    // TODO: Consider what we will do with material based fields, more tables? Separate option?
    void flatten_single_domain(const Node &mesh, Node &output,
        const std::vector<std::string> &fields_to_flatten, index_t domain_id,
        index_t vert_offset, index_t elem_offset) const;

    /**
    @brief Handles allocating output table and invoking flatten_single_domain() over
        each domain at the proper offsets, to be overridden by parallel implementation.
    @param mesh Input mesh, must be an object or list of mesh domains.
    @param output The output to be returned by execute()
    */
    virtual void
    flatten_many_domains(const Node &mesh, Node &output) const;

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
// -- begin conduit::mesh::utils --
//-----------------------------------------------------------------------------
namespace utils
{

// NOTE: Putting template function definitions down here so that class definition
//  and function prototypes are easily accessible at the top..

//-----------------------------------------------------------------------------
template<typename AccessType, typename FuncType>
void
for_each_in_range(Node &node, index_t start, index_t end, FuncType &&func)
{
    const index_t dtype_id = node.dtype().id();
#define CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL(ActualType)\
{\
    DataArray<ActualType> value = node.value();\
    index_t offset = start;\
    for(index_t i = 0; offset < end; i++, offset++)\
    {\
        AccessType temp = static_cast<AccessType>(value[offset]);\
        func(i, temp);\
        value[offset] = static_cast<ActualType>(temp);\
    }\
}
    switch(dtype_id)
    {
    // Signed int types
    case DataType::INT8_ID:
    {
        CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL(int8)
        break;
    }
    case DataType::INT16_ID:
    {
        CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL(int16)
        break;
    }
    case DataType::INT32_ID:
    {
        CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL(int32)
        break;
    }
    case DataType::INT64_ID:
    {
        CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL(int64)
        break;
    }
    // Unsigned int types
    case DataType::UINT8_ID:
    {
        CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL(uint8)
        break;
    }
    case DataType::UINT16_ID:
    {
        CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL(uint16)
        break;
    }
    case DataType::UINT32_ID:
    {
        CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL(uint32)
        break;
    }
    case DataType::UINT64_ID:
    {
        CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL(uint64)
        break;
    }
    // Floating point types
    case DataType::FLOAT32_ID:
    {
        CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL(float32)
        break;
    }
    case DataType::FLOAT64_ID:
    {
        CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL(float64)
        break;
    }
    default:
    {
        CONDUIT_ERROR("Invalid data type passed to for_each_in_range");
    }
    }

// Cleanup macro
#undef CONDUIT_BLUEPRINT_MESH_UTILS_FOR_EACH_IN_RANGE_IMPL
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils --
//-----------------------------------------------------------------------------

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

#endif
