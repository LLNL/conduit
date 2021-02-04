// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_matset_xforms.cpp
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
// -- begin conduit::blueprint::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::matset --
//-----------------------------------------------------------------------------
namespace matset
{
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::matset::detail --
//-----------------------------------------------------------------------------
namespace detail
{
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Single implementation that supports the case where just matset
// is passed, and the case where the field is passed.
//
// This is in the detail name space b/c the calling convention is a little
// strange:
//   empty field  node -- first arg, triggers one path, non empty another
//
// We smooth this out for the API by providing the non detail variants,
// which error when passed empty nodes.
//-----------------------------------------------------------------------------
void
to_silo(const conduit::Node &field,
        const conduit::Node &matset,
        conduit::Node &dest,
        const float64 epsilon)
{
    Node temp, data;
    const DataType int_dtype = find_widest_dtype2(matset,
                                            blueprint::mesh::default_int_dtypes);
    const DataType float_dtype = find_widest_dtype2(matset,
                                            blueprint::mesh::default_float_dtype);
    // if matset_values is not empty, we will
    // apply the same xform to it as we do to the volume fractions.
    const bool xform_matset_values = field.has_child("matset_values");

    // NOTE: matset values are always treated as a float64.
    // we could map to the widest int or float type in the future.

    // Extract Material Set Metadata //
    const bool mset_is_unibuffer = blueprint::mesh::matset::is_uni_buffer(matset);
    const bool mset_is_matdom = blueprint::mesh::matset::is_material_dominant(matset);

    // setup the material map, which provides a map from material names
    // to to material numbers
    Node matset_mat_map;
    
    // mset_is_unibuffer will always have the material_map, other cases
    // it is optional. If not given, the map from material names to ids
    // is implied by the order the materials are presented in the matset node
    if(matset.has_child("material_map") )
    {
        // uni-buffer case provides the map we are looking for
        matset_mat_map.set_external(matset["material_map"]);
    }
    else // if(!mset_is_unibuffer)
    {
        // material_map is implied, construct it here for use and output
        NodeConstIterator vf_itr = matset["volume_fractions"].children();
        while(vf_itr.has_next())
        {
            vf_itr.next();
            std::string curr_mat_name = vf_itr.name();
            temp.set(vf_itr.index());
            temp.to_data_type(int_dtype.id(), matset_mat_map[curr_mat_name]);
        }
    }
    
    const Node mset_mat_map(matset_mat_map);

    // find the number of elements in the matset
    index_t matset_num_elems = 0;
    if(mset_is_matdom)
    {
        NodeConstIterator eids_iter = matset["element_ids"].children();
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
        // may need to do a bit of sculpting here; embed the base array into
        // something w/ "values" child, as below
        Node mat_vfs;
        if(mset_is_unibuffer)
        {
            mat_vfs.set_external(matset);
        }
        else
        {
            const Node &temp_vfs = matset["volume_fractions"].child(0);
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

    // create a sparse map from each zone, to each material and its value.
    std::vector< std::map<index_t, float64> > elem_mat_maps(mset_num_elems);
    std::vector< std::map<index_t, float64> > elem_matset_values_maps(mset_num_elems);
    if(mset_is_unibuffer)
    {
        const Node &mat_vfs = matset["volume_fractions"];
        const Node &mat_mids = matset["material_ids"];

        Node mat_eids;
        if(mset_is_matdom)
        {
            mat_eids.set_external(matset["element_ids"]);
        }

        blueprint::o2mrelation::O2MIterator mat_iter(matset);
        while(mat_iter.has_next(O2MIndex::DATA))
        {
            const index_t elem_ind_index = mat_iter.next(O2MIndex::ONE);
            
            // -- get element id -- //
            // this is either "elem_ind_index" from the o2m, or
            // this index applied to the material-to-elements map
            if(mset_is_matdom)
            {
                temp.set_external(
                    DataType(mat_eids.dtype().id(), 1),
                    (void*)mat_eids.element_ptr(elem_ind_index));
            }

            const index_t elem_index = mset_is_matdom ? temp.to_index_t() : elem_ind_index;

            // we now have the element index, find all material indicies
            // using the o2m-many iter
            mat_iter.to_front(O2MIndex::MANY);
            while(mat_iter.has_next(O2MIndex::MANY))
            {
                mat_iter.next(O2MIndex::MANY);
                const index_t mat_ind_index = mat_iter.index(O2MIndex::DATA);

                // this index now allows us to fetch the
                //  vol frac
                //  matset value
                //  material id
                
                // get the vf and convert it to a float64
                temp.set_external(
                    DataType(mat_vfs.dtype().id(), 1),
                    (void*)mat_vfs.element_ptr(mat_ind_index));
                const float64 mat_vf = temp.to_float64();

                float64 curr_matset_value = 0;
                // process matset values if passed and convert it to a float64
                if(xform_matset_values)
                {
                    const Node matset_values = field["matset_values"];
                    temp.set_external(
                        DataType(matset_values.dtype().id(), 1),
                        (void*)matset_values.element_ptr(mat_ind_index));
                        curr_matset_value = temp.to_float64();
                }

                // get the material id as an index_t
                temp.set_external(
                    DataType(mat_mids.dtype().id(), 1),
                    (void*)mat_mids.element_ptr(mat_ind_index));
                const index_t mat_id = temp.to_index_t();

                // if this elem has a non-zero (or non-trivial) volume fraction for this
                // material, add it do the map
                if(mat_vf > epsilon)
                {
                    elem_mat_maps[elem_index][mat_id] = mat_vf;

                    // process matset values if passed
                    if(xform_matset_values)
                    {
                        elem_matset_values_maps[elem_index][mat_id] = curr_matset_value;
                    }
                }
            }
        }
    }
    else // if(!mset_is_unibuffer)
    {
        NodeConstIterator mats_iter = matset["volume_fractions"].children();
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
                mat_eids.set_external(matset["element_ids"][mat_name]);
            }

            // this is a multi-buffer case, make sure we are pointing
            // to the correct values for this pass
            Node mat_data;
            {
                const std::string vf_path =
                    blueprint::o2mrelation::data_paths(mat_vfs).front();
                mat_data.set_external(mat_vfs[vf_path]);
            }

            blueprint::o2mrelation::O2MIterator mat_iter(mat_vfs);
            for(index_t mat_index = 0; mat_iter.has_next(); mat_index++)
            {
                const index_t mat_itr_index = mat_iter.next();

                // get the current vf value as a float64
                temp.set_external(
                    DataType(mat_data.dtype().id(), 1),
                    (void*)mat_data.element_ptr(mat_itr_index));
                const float64 mat_vf = temp.to_float64();

                // if material dominant:
                //  we use indirection array to find the element index.
                //
                // if element dominant:
                //  the o2m_index is the element index

                if(mset_is_matdom)
                {
                    temp.set_external(
                        DataType(mat_eids.dtype().id(), 1),
                        (void*)mat_eids.element_ptr(mat_index));
                }
                const index_t mat_elem = mset_is_matdom ? temp.to_index_t() : mat_index;
                
                // we now have both the element and material index.

                // if this elem has a non-zero (or non-trivial) volume fraction for this
                // material, add it do the map
                if(mat_vf > epsilon)
                {
                    elem_mat_maps[mat_elem][mat_id] = mat_vf;
                }
            }
        }

        /// handle case where matset_values was passed
        /// this requires another o2m traversal
        if(xform_matset_values)
        {
            NodeConstIterator matset_values_iter = field["matset_values"].children();
            while(matset_values_iter.has_next())
            {
                const Node& curr_node = matset_values_iter.next();
                const std::string& mat_name = matset_values_iter.name();
                const index_t mat_id = mset_mat_map[mat_name].to_index_t();

                // NOTE(JRC): This is required because per-material subtrees aren't
                // necessarily 'o2mrelation'-compliant; they can just be raw arrays.
                // To make subsequent processing uniform, we make raw arrays 'o2mrelation's.

                Node o2m;
                if(curr_node.dtype().is_number())
                {
                    o2m["values"].set_external(curr_node);
                }
                else
                {
                    o2m.set_external(curr_node);
                }

                Node mat_eids;
                if(mset_is_matdom)
                {
                    mat_eids.set_external(matset["element_ids"][mat_name]);
                }

            // this is a multi-buffer case, make sure we are pointing
            // to the correct values for this pass
                Node matset_values_data;
                {
                    const std::string path =
                        blueprint::o2mrelation::data_paths(o2m).front();
                    matset_values_data.set_external(o2m[path]);
                }

                blueprint::o2mrelation::O2MIterator o2m_iter(o2m);
                for(index_t o2m_index = 0; o2m_iter.has_next(); o2m_index++)
                {
                    const index_t o2m_access_index = o2m_iter.next();

                    // if material dominant:
                    //  we use indirection array to find the element index.
                    //
                    // if element dominant:
                    //  the o2m_index is the element index
                    if(mset_is_matdom)
                    {
                        temp.set_external(
                            DataType(mat_eids.dtype().id(), 1),
                            (void*)mat_eids.element_ptr(o2m_index));
                    }


                    const index_t mat_elem = mset_is_matdom ? temp.to_index_t() : o2m_index;

                    // we now have both the element and material index.
                    // check if the volume fractions have an entry for this case,
                    // if so we will add the corresponding mixvar to its map

                    // if elem_mat_maps[mat_elem] has entry mat_id, add entry to
                    // elem_matset_values_maps
                    if( elem_mat_maps[mat_elem].find(mat_id) != elem_mat_maps[mat_elem].end())
                    {
                        temp.set_external(
                          DataType(matset_values_data.dtype().id(), 1),
                          (void*)matset_values_data.element_ptr(o2m_access_index));
                        const float64 curr_matset_value  = temp.to_float64();
                        elem_matset_values_maps[mat_elem][mat_id] = curr_matset_value;
                    }
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
    dest["topology"].set(matset["topology"]);
    // in some cases, this method will sort the material names
    // so always include the material map
    dest["material_map"].set(matset_mat_map);
    dest["matlist"].set(DataType(int_dtype.id(), mset_num_elems));
    dest["mix_next"].set(DataType(int_dtype.id(), mset_num_slots));
    dest["mix_mat"].set(DataType(int_dtype.id(), mset_num_slots));
    dest["mix_vf"].set(DataType(float_dtype.id(), mset_num_slots));

    if(xform_matset_values)
    {
        dest["field_mixvar_values"].set(DataType(float_dtype.id(), mset_num_slots));
        if(field.has_child("values"))
        {
            dest["field_values"].set(field["values"]);
        }
    }

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

                // also do matset_values if passed
                // elem_index ==> element index
                // zone_mix_mat.first ==> material index
                // process matset values if passed
                if(xform_matset_values)
                {
                    temp.set(elem_matset_values_maps[elem_index][zone_mix_mat.first]);
                    data.set_external(float_dtype, dest["field_mixvar_values"].element_ptr(next_slot_index));
                    temp.to_data_type(float_dtype.id(), data);
                }

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

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::matset::detail --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void
to_silo(const conduit::Node &matset,
        conduit::Node &dest,
        const float64 epsilon)
{
    // extra seat belt here b/c we want to avoid folks entering
    // the detail version of to_silo with surprising results. 

    if(!matset.dtype().is_object() )
    {
        CONDUIT_ERROR("blueprint::mesh::matset::to_silo passed matset node"
                      " must be a valid matset tree.");
    }
    
    conduit::Node field;

    detail::to_silo(field,
                    matset,
                    dest,
                    epsilon);
}


//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::matset --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::matset --
//-----------------------------------------------------------------------------
namespace field
{
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void
to_silo(const conduit::Node &field,
        const conduit::Node &matset,
        conduit::Node &dest,
        const float64 epsilon)
{
    // extra seat belts here b/c we want to avoid folks entering
    // the detail version of to_silo with surprising results.

    if(!field.dtype().is_object() )
    {
        CONDUIT_ERROR("blueprint::mesh::field::to_silo passed field node"
                      " must be a valid matset tree.");
    }

    if(!matset.dtype().is_object() )
    {
        CONDUIT_ERROR("blueprint::mesh::matset::to_silo passed matset node"
                      " must be a valid matset tree.");
    }

    conduit::blueprint::mesh::matset::detail::to_silo(field,
                                                      matset,
                                                      dest,
                                                      epsilon);
}

//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::field --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint:::mesh --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

