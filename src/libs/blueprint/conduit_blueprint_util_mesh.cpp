// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_util_mesh.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// std lib includes
#include <algorithm>
#include <cmath>
#include <string>
#include <map>
#include <vector>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_o2mrelation_iterator.hpp"
#include "conduit_blueprint_util_mesh.hpp"

// FIXME(JRC): The helper functions below are hackily copied over from
// 'conduit_blueprint_mesh.cpp'; these helpers should ultimately be abstracted
// into an internal module and shared.

using namespace conduit;
namespace O2MIndex = conduit::blueprint::o2mrelation;

//-----------------------------------------------------------------------------
namespace conduit { namespace blueprint { namespace mesh {
//-----------------------------------------------------------------------------
    static const DataType default_int_dtype(DataType::INT32_ID, 1);
    static const DataType default_uint_dtype(DataType::UINT32_ID, 1);
    static const DataType default_float_dtype(DataType::FLOAT64_ID, 1);

    static const DataType default_int_dtype_list[2] = {default_int_dtype, default_uint_dtype};
    static const std::vector<DataType> default_int_dtypes(default_int_dtype_list,
        default_int_dtype_list + sizeof(default_int_dtype_list) / sizeof(default_int_dtype_list[0]));

    static const DataType default_number_dtype_list[3] = {default_float_dtype,
        default_int_dtype, default_uint_dtype};
    static const std::vector<DataType> default_number_dtypes(default_number_dtype_list,
        default_number_dtype_list + sizeof(default_number_dtype_list) /
        sizeof(default_number_dtype_list[0]));
} } }

//-----------------------------------------------------------------------------
// -- begin internal helpers --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
DataType find_widest_dtype2(const Node &node,
                            const std::vector<DataType> &default_dtypes)
{
    DataType widest_dtype(default_dtypes[0].id(), 0, 0, 0, 0, default_dtypes[0].endianness());

    std::vector<const Node*> node_bag(1, &node);
    while(!node_bag.empty())
    {
        const Node *curr_node = node_bag.back(); node_bag.pop_back();
        const DataType curr_dtype = curr_node->dtype();
        if( curr_dtype.is_list() || curr_dtype.is_object() )
        {
            NodeConstIterator curr_node_it = curr_node->children();
            while(curr_node_it.has_next())
            {
                node_bag.push_back(&curr_node_it.next());
            }
        }
        else
        {
            for(index_t ti = 0; ti < (index_t)default_dtypes.size(); ti++)
            {
                const DataType &valid_dtype = default_dtypes[ti];
                bool is_valid_dtype =
                    (curr_dtype.is_floating_point() && valid_dtype.is_floating_point()) ||
                    (curr_dtype.is_signed_integer() && valid_dtype.is_signed_integer()) ||
                    (curr_dtype.is_unsigned_integer() && valid_dtype.is_unsigned_integer()) ||
                    (curr_dtype.is_string() && valid_dtype.is_string());
                if(is_valid_dtype && (widest_dtype.element_bytes() < curr_dtype.element_bytes()))
                {
                    widest_dtype.set(DataType(curr_dtype.id(), 1));
                }
            }
        }
    }

    bool no_type_found = widest_dtype.element_bytes() == 0;
    return no_type_found ? default_dtypes[0] : widest_dtype;
}

//-----------------------------------------------------------------------------
DataType find_widest_dtype2(const Node &node,
                            const DataType &default_dtype)
{
    return find_widest_dtype2(node, std::vector<DataType>(1, default_dtype));
}

//-----------------------------------------------------------------------------
// -- end internal helpers --
//-----------------------------------------------------------------------------

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
// -- begin conduit::blueprint::util --
//-----------------------------------------------------------------------------
namespace util
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::util::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::util::mesh::matset --
//-----------------------------------------------------------------------------
namespace matset
{
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
to_silo(const conduit::Node &n,
        conduit::Node &dest,
        const float64 epsilon)
{
    Node temp, data;
    const DataType int_dtype = find_widest_dtype2(n, blueprint::mesh::default_int_dtypes);
    const DataType float_dtype = find_widest_dtype2(n, blueprint::mesh::default_float_dtype);

    // Extract Material Set Metadata //

    const bool mset_is_unibuffer = blueprint::mesh::matset::is_uni_buffer(n);
    const bool mset_is_matdom = blueprint::mesh::matset::is_material_dominant(n);

    Node matset_mat_map;
    if(mset_is_unibuffer)
    {
        matset_mat_map.set_external(n["volume_fractions/material_map"]);
    }
    else // if(!mset_is_unibuffer)
    {
        std::vector<std::string> mat_vec = n["volume_fractions"].child_names();
        std::sort(mat_vec.begin(), mat_vec.end());
        for(int64 mat_index = 0; mat_index < (index_t)mat_vec.size(); mat_index++)
        {
            const std::string &mat_name = mat_vec[mat_index];
            temp.set_external(DataType::int64(1), &mat_index);
            temp.to_data_type(int_dtype.id(), matset_mat_map[mat_name]);
        }
    }
    const Node mset_mat_map(matset_mat_map);

    index_t matset_num_elems = 0;
    if(mset_is_matdom)
    {
        NodeConstIterator eids_iter = n["element_ids"].children();
        while(eids_iter.has_next())
        {
            const Node &eids_node = eids_iter.next();
            const DataType eids_dtype(eids_node.dtype().id(), 1);
            for(index_t ei = 0; ei < eids_node.dtype().number_of_elements(); ei++)
            {
                temp.set_external(eids_dtype, (void*)eids_node.element_ptr(ei));
                const index_t elem_index = temp.to_int();
                matset_num_elems = std::max(matset_num_elems, elem_index + 1);
            }
        }
    }
    else // if(!mset_is_matdom)
    {
        // may need to do a bit of sculping here; embed the base array into
        // something w/ "values" child, as below
        Node mat_vfs;
        if(mset_is_unibuffer)
        {
            mat_vfs["values"].set_external(n["volume_fractions/values"]);
        }
        else
        {
            const Node &temp_vfs = n["volume_fractions"].child(0);
            if(temp_vfs.dtype().is_object())
            {
                mat_vfs.set_external(temp_vfs);
            }
            else // if(temp_vfs.dtype().is_number())
            {
                mat_vfs["values"].set_external(temp_vfs);
            }
        }

        blueprint::o2mrelation::O2MIterator mat_iter(mat_vfs);
        matset_num_elems = mat_iter.elements(O2MIndex::ONE);
    }
    const index_t mset_num_elems = matset_num_elems;

    // Organize Per-Zone Material Data //

    std::vector< std::map<index_t, float64> > elem_mat_maps(mset_num_elems);
    if(mset_is_unibuffer)
    {
        const Node &mat_vfs = n["volume_fractions"];
        const Node &mat_mids = n["material_ids"];
        Node mat_eids;
        if(mset_is_matdom)
        {
            mat_eids.set_external(n["element_ids"]);
        }

        blueprint::o2mrelation::O2MIterator mat_iter(n);
        while(mat_iter.has_next(O2MIndex::DATA))
        {
            const index_t elem_ind_index = mat_iter.next(O2MIndex::ONE);
            if(mset_is_matdom)
            {
                temp.set_external(
                    DataType(mat_eids.dtype().id(), 1),
                    (void*)mat_eids.element_ptr(elem_ind_index));
            }
            const index_t elem_index = mset_is_matdom ? temp.to_index_t() : elem_ind_index;

            mat_iter.to_front(O2MIndex::MANY);
            while(mat_iter.has_next(O2MIndex::MANY))
            {
                mat_iter.next(O2MIndex::MANY);
                const index_t mat_ind_index = mat_iter.index(O2MIndex::DATA);

                temp.set_external(
                    DataType(mat_vfs.dtype().id(), 1),
                    (void*)mat_vfs.element_ptr(mat_ind_index));
                const float64 mat_vf = temp.to_float64();

                temp.set_external(
                    DataType(mat_mids.dtype().id(), 1),
                    (void*)mat_mids.element_ptr(mat_ind_index));
                const index_t mat_id = temp.to_index_t();

                // if this elem has a non-zero (or non-trival) volume fraction for this
                // materal, add it do the map
                if(mat_vf > epsilon)
                {
                    elem_mat_maps[elem_index][mat_id] = mat_vf;
                }
            }
        }
    }
    else // if(!mset_is_unibuffer)
    {
        NodeConstIterator mats_iter = n["volume_fractions"].children();
        while(mats_iter.has_next())
        {
            const Node& mat_node = mats_iter.next();
            const std::string& mat_name = mats_iter.name();
            const index_t mat_id = mset_mat_map[mat_name].to_index_t();

            // NOTE(JRC): This is required because per-material subtrees aren't
            // necessarily 'o2mrelation'-compliant; they can just be raw arrays.
            // To make subsequent processing uniform, we make raw arrays 'o2mrelation's.
            Node mat_vfs;
            if(mat_node.dtype().is_number())
            {
                mat_vfs["values"].set_external(mat_node);
            }
            else
            {
                mat_vfs.set_external(mat_node);
            }

            Node mat_eids;
            if(mset_is_matdom)
            {
                mat_eids.set_external(n["element_ids"][mat_name]);
            }

            Node mat_data; {
                const std::string vf_path =
                    blueprint::o2mrelation::data_paths(mat_vfs).front();
                mat_data.set_external(mat_vfs[vf_path]);
            }

            blueprint::o2mrelation::O2MIterator mat_iter(mat_vfs);
            for(index_t mat_index = 0; mat_iter.has_next(); mat_index++)
            {
                temp.set_external(
                    DataType(mat_data.dtype().id(), 1),
                    (void*)mat_data.element_ptr(mat_iter.next()));
                const float64 mat_vf = temp.to_float64();

                if(mset_is_matdom)
                {
                    temp.set_external(
                        DataType(mat_eids.dtype().id(), 1),
                        (void*)mat_eids.element_ptr(mat_index));
                }
                const index_t mat_elem = mset_is_matdom ? temp.to_index_t() : mat_index;

                // if this elem has a non-zero (or non-trival) volume fraction for this
                // materal, add it do the map
                if(mat_vf > epsilon)
                {
                    elem_mat_maps[mat_elem][mat_id] = mat_vf;
                }
            }
        }
    }

    index_t matset_num_slots = 0;
    for(const std::map<index_t, float64> &elem_mat_map: elem_mat_maps)
    {
        matset_num_slots += (elem_mat_map.size() > 1) ? elem_mat_map.size() : 0;
    }
    const index_t mset_num_slots = matset_num_slots;

    // Generate Silo Data Structures //

    dest.reset();
    dest["topology"].set(n["topology"]);
    dest["matlist"].set(DataType(int_dtype.id(), mset_num_elems));
    dest["mix_next"].set(DataType(int_dtype.id(), mset_num_slots));
    dest["mix_mat"].set(DataType(int_dtype.id(), mset_num_slots));
    dest["mix_vf"].set(DataType(float_dtype.id(), mset_num_slots));

    for(index_t elem_index = 0, slot_index = 0; elem_index < mset_num_elems; elem_index++)
    {
        const std::map<index_t, float64>& elem_mat_map = elem_mat_maps[elem_index];
        if(elem_mat_map.size() == 0)
        {
            temp.set(0);
            data.set_external(int_dtype, dest["matlist"].element_ptr(elem_index));
            temp.to_data_type(int_dtype.id(), data);
        }
        else if(elem_mat_map.size() == 1)
        {
            temp.set(elem_mat_map.begin()->first + 1);
            data.set_external(int_dtype, dest["matlist"].element_ptr(elem_index));
            temp.to_data_type(int_dtype.id(), data);
        }
        else
        {
            index_t next_slot_index = slot_index;
            for(const auto& zone_mix_mat : elem_mat_map)
            {
                temp.set(zone_mix_mat.first);
                data.set_external(int_dtype, dest["mix_mat"].element_ptr(next_slot_index));
                temp.to_data_type(int_dtype.id(), data);

                temp.set(zone_mix_mat.second);
                data.set_external(float_dtype, dest["mix_vf"].element_ptr(next_slot_index));
                temp.to_data_type(float_dtype.id(), data);

                temp.set(next_slot_index + 1 + 1);
                data.set_external(int_dtype, dest["mix_next"].element_ptr(next_slot_index));
                temp.to_data_type(int_dtype.id(), data);

                ++next_slot_index;
            }
            temp.set(0);
            data.set_external(int_dtype, dest["mix_next"].element_ptr(next_slot_index - 1));
            temp.to_data_type(int_dtype.id(), data);

            temp.set(~slot_index);
            data.set_external(int_dtype, dest["matlist"].element_ptr(elem_index));
            temp.to_data_type(int_dtype.id(), data);

            slot_index += elem_mat_map.size();
        }
    }
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::util::mesh::matset --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::util::mesh --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::util --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

