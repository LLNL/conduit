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
#include <numeric>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_o2mrelation_iterator.hpp"
#include "conduit_blueprint_o2mrelation_index.hpp"

using namespace conduit;
// access conduit blueprint mesh utilities
namespace bputils = conduit::blueprint::mesh::utils;
// access one-to-many index types
namespace o2mrelation = conduit::blueprint::o2mrelation;

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
    const DataType int_dtype = bputils::find_widest_dtype(matset, bputils::DEFAULT_INT_DTYPES);
    const DataType float_dtype = bputils::find_widest_dtype(matset, bputils::DEFAULT_FLOAT_DTYPE);
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
            temp.reset();
            temp.set(vf_itr.index());
            temp.to_data_type(int_dtype.id(), matset_mat_map[curr_mat_name]);
        }
    }

    const Node mset_mat_map(matset_mat_map);

    // find the number of elements in the matset
    index_t matset_num_elems = 0;
    if(mset_is_matdom)
    {
        if(mset_is_unibuffer)
        {
            const DataAccessor<index_t> eids = matset["element_ids"].value();
            const index_t N = eids.number_of_elements();
            for(index_t i = 0; i < N; i++)
            {
                matset_num_elems = std::max(matset_num_elems, eids[i] + 1);
            }
        }
        else
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
        matset_num_elems = mat_iter.elements(o2mrelation::ONE);
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
        while(mat_iter.has_next(o2mrelation::DATA))
        {
            const index_t elem_ind_index = mat_iter.next(o2mrelation::ONE);

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
            mat_iter.to_front(o2mrelation::MANY);
            while(mat_iter.has_next(o2mrelation::MANY))
            {
                mat_iter.next(o2mrelation::MANY);
                const index_t mat_ind_index = mat_iter.index(o2mrelation::DATA);

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
        CONDUIT_ASSERT(elem_mat_map.size() != 0, "A zone has no materials.");
        if (elem_mat_map.size() == 1)
        {
            temp.reset();
            temp.set(elem_mat_map.begin()->first);
            data.set_external(int_dtype, dest["matlist"].element_ptr(elem_index));
            temp.to_data_type(int_dtype.id(), data);
        }
        else
        {
            index_t next_slot_index = slot_index;
            for(const auto& zone_mix_mat : elem_mat_map)
            {
                temp.reset();
                temp.set(zone_mix_mat.first);
                data.set_external(int_dtype, dest["mix_mat"].element_ptr(next_slot_index));
                temp.to_data_type(int_dtype.id(), data);

                // also do matset_values if passed
                // elem_index ==> element index
                // zone_mix_mat.first ==> material index
                // process matset values if passed
                if(xform_matset_values)
                {
                    temp.reset();
                    temp.set(elem_matset_values_maps[elem_index][zone_mix_mat.first]);
                    data.set_external(float_dtype, dest["field_mixvar_values"].element_ptr(next_slot_index));
                    temp.to_data_type(float_dtype.id(), data);
                }

                temp.reset();
                temp.set(zone_mix_mat.second);
                data.set_external(float_dtype, dest["mix_vf"].element_ptr(next_slot_index));
                temp.to_data_type(float_dtype.id(), data);

                temp.reset();
                temp.set(next_slot_index + 1 + 1);
                data.set_external(int_dtype, dest["mix_next"].element_ptr(next_slot_index));
                temp.to_data_type(int_dtype.id(), data);

                ++next_slot_index;
            }

            temp.reset();
            temp.set(0);
            data.set_external(int_dtype, dest["mix_next"].element_ptr(next_slot_index - 1));
            temp.to_data_type(int_dtype.id(), data);


            temp.reset();
            temp.set(~slot_index);
            data.set_external(int_dtype, dest["matlist"].element_ptr(elem_index));
            temp.to_data_type(int_dtype.id(), data);

            slot_index += elem_mat_map.size();
        }
    }
}

//-----------------------------------------------------------------------------
// field copy
void
copy_matset_independent_parts_of_field(const conduit::Node &src_field,
                                       const std::string &dest_matset_name,
                                       conduit::Node &dest_field)
{
    // copy over everything except the matset values and matset name
    auto field_child_itr = src_field.children();
    while (field_child_itr.has_next())
    {
        const Node &n_field_info = field_child_itr.next();
        std::string field_child_name = field_child_itr.name();

        if (field_child_name != "matset_values" &&
            field_child_name != "matset")
        {
            dest_field[field_child_name].set(n_field_info);
        }
    }
    dest_field["matset"] = dest_matset_name;
}

//-----------------------------------------------------------------------------
void
create_reverse_matmap(const conduit::Node &src_matset,
                      std::map<int, std::string> &reverse_matmap)
{
    // fill out map
    auto matmap_itr = src_matset["material_map"].children();
    while (matmap_itr.has_next())
    {
        const Node &matmap_entry = matmap_itr.next();
        const std::string matname = matmap_itr.name();

        reverse_matmap[matmap_entry.to_int()] = matname;
    }
}

//-----------------------------------------------------------------------------
// load the element ids into a set to find out how many there are
int
determine_num_elems_in_multi_buffer_by_material(const conduit::Node &elem_ids)
{
    std::set<int> elem_ids_set;

    auto eid_itr = elem_ids.children();
    while (eid_itr.has_next())
    {
        const Node &mat_elem_ids = eid_itr.next();
        const std::string matname = eid_itr.name();
        int64_accessor mat_elem_ids_vals = mat_elem_ids.value();
        int num_vf = mat_elem_ids_vals.dtype().number_of_elements();
        for (int i = 0; i < num_vf; i ++)
        {
            elem_ids_set.insert(mat_elem_ids_vals[i]);
        }
    }

    return static_cast<int>(elem_ids_set.size());
};

//-----------------------------------------------------------------------------
void
walk_uni_buffer_by_element_to_multi_buffer_by_element(
    const conduit::Node &src_matset,
    std::map<int, std::string> &reverse_matmap,
    float64_accessor &values, // can be either vol fracs or matset vals
    int64_accessor &material_ids,
    std::map<std::string, std::vector<float64>> &new_vals)
{
    auto o2m_idx = o2mrelation::O2MIndex(src_matset);
    int num_elems = o2m_idx.size();

    // initialize sizes
    for (auto & mapitem : reverse_matmap)
    {
        const std::string &matname = mapitem.second;
        new_vals[matname] = std::vector<float64>(num_elems);
    }

    // iterate through matset
    for (int elem_id = 0; elem_id < num_elems; elem_id ++)
    {
        for (int many_id = 0; many_id < o2m_idx.size(elem_id); many_id ++)
        {
            index_t data_index = o2m_idx.index(elem_id, many_id);

            float64 val = values[data_index];
            int mat_id = material_ids[data_index];
            const std::string &matname = reverse_matmap[mat_id];
            new_vals[matname][elem_id] = val;
        }
    }
}

//-----------------------------------------------------------------------------
template<typename T>
void
read_from_map_write_out(std::map<std::string, std::vector<T>> &datamap,
                        conduit::Node &destination)
{
    for (auto & mapitem : datamap)
    {
        const std::string &matname = mapitem.first;
        const std::vector<T> &data_vector = mapitem.second;

        destination[matname].set(data_vector.data(), data_vector.size());
    }
}

//-----------------------------------------------------------------------------
void
create_sbm_rep(const conduit::Node &elem_id_src,
               const conduit::Node &values_src,
               std::map<std::string, std::pair<int64_accessor, float64_accessor>> &sbm_rep)
{
    auto eid_itr = elem_id_src.children();
    while (eid_itr.has_next())
    {
        const Node &mat_elem_ids = eid_itr.next();
        const std::string matname = eid_itr.name();
        sbm_rep[matname].first = mat_elem_ids.value();
    }

    auto val_itr = values_src.children();
    while (val_itr.has_next())
    {
        const Node &values = val_itr.next();
        const std::string matname = val_itr.name();
        sbm_rep[matname].second = values.value();
    }
}

//-----------------------------------------------------------------------------
void
sbm_rep_to_full(const std::map<std::string, std::pair<int64_accessor, float64_accessor>> &sbm_rep,
                const int num_elems,
                conduit::Node &destination)
{
    for (auto &mapitem : sbm_rep)
    {
        std::vector<double> values(num_elems, 0.0);
        
        const std::string &matname = mapitem.first;
        const int64_accessor sbm_eids = mapitem.second.first;
        const float64_accessor sbm_vals = mapitem.second.second;
        
        const int num_vf = sbm_vals.dtype().number_of_elements();
        for (int mat_vf_id = 0; mat_vf_id < num_vf; mat_vf_id ++)
        {
            const int elem_id = sbm_eids[mat_vf_id];
            const double value = sbm_vals[mat_vf_id];

            values[elem_id] = value;
        }

        destination[matname].set(values.data(), values.size());
    }
}

//-----------------------------------------------------------------------------
// venn full -> sparse by element
void
multi_buffer_by_element_to_uni_buffer_by_element_matset(const conduit::Node &src_matset,
                                                        conduit::Node &dest_matset,
                                                        const float64 epsilon)
{
    dest_matset.reset();

    // set the topology
    dest_matset["topology"].set(src_matset["topology"]);

    std::map<int, float64_accessor> full_vol_fracs;
    // create the material map
    auto mat_itr = src_matset["volume_fractions"].children();
    int mat_id = 0;
    while (mat_itr.has_next())
    {
        const Node &mat_vol_fracs = mat_itr.next();
        std::string matname = mat_itr.name();
        full_vol_fracs[mat_id] = mat_vol_fracs.value();
        dest_matset["material_map"][matname] = mat_id;
        mat_id ++;
    }

    const int nmats = dest_matset["material_map"].number_of_children();

    std::vector<double> vol_fracs;
    std::vector<int> mat_ids;
    std::vector<int> sizes;
    std::vector<int> offsets;

    int num_elems = src_matset["volume_fractions"][0].dtype().number_of_elements();
    int offset = 0;

    for (int elem_id = 0; elem_id < num_elems; elem_id ++)
    {
        int size = 0;
        for (int mat_id = 0; mat_id < nmats; mat_id ++)
        {
            float64 vol_frac = full_vol_fracs[mat_id][elem_id];
            if (vol_frac > epsilon)
            {
                vol_fracs.push_back(vol_frac);
                mat_ids.push_back(mat_id);
                size ++;
            }
        }
        sizes.push_back(size);
        offsets.push_back(offset);
        offset += size;
    }

    dest_matset["volume_fractions"].set(vol_fracs.data(), vol_fracs.size());
    dest_matset["material_ids"].set(mat_ids.data(), mat_ids.size());
    dest_matset["sizes"].set(sizes.data(), sizes.size());
    dest_matset["offsets"].set(offsets.data(), offsets.size());
}

//-----------------------------------------------------------------------------
// venn full -> sparse by element
void
multi_buffer_by_element_to_uni_buffer_by_element_field(const conduit::Node &src_matset,
                                                       const conduit::Node &src_field,
                                                       const std::string &dest_matset_name,
                                                       conduit::Node &dest_field,
                                                       const float64 epsilon)
{
    dest_field.reset();

    // if this field is material dependent
    if (src_field.has_child("matset_values"))
    {
        copy_matset_independent_parts_of_field(src_field,
                                               dest_matset_name,
                                               dest_field);

        // map material ids to matset values and volume fractions
        std::map<int, float64_accessor> full_vol_fracs;
        std::map<int, float64_accessor> full_matset_vals;
        std::set<int> mat_ids;

        // create the material map
        auto mat_itr = src_matset["volume_fractions"].children();
        auto fmat_itr = src_field["matset_values"].children();
        int mat_id = 0;
        while (mat_itr.has_next() && fmat_itr.has_next())
        {
            const Node &mat_vol_fracs = mat_itr.next();
            std::string matname = mat_itr.name();

            const Node &mat_field_vals = fmat_itr.next();
            std::string fmatname = fmat_itr.name();

            CONDUIT_ASSERT(matname == fmatname, "Materials must be ordered the same in "
                "material dependent fields and their matsets.");

            full_vol_fracs[mat_id] = mat_vol_fracs.value();
            full_matset_vals[mat_id] = mat_field_vals.value();
            mat_ids.insert(mat_id);
            mat_id ++;
        }

        std::vector<float64> matset_values;

        const int nmats = static_cast<int>(mat_ids.size());
        int num_elems = src_matset["volume_fractions"][0].dtype().number_of_elements();

        for (int elem_id = 0; elem_id < num_elems; elem_id ++)
        {
            int size = 0;
            for (mat_id = 0; mat_id < nmats; mat_id ++)
            {
                float64 matset_val = full_matset_vals[mat_id][elem_id];
                float64 vol_frac = full_vol_fracs[mat_id][elem_id];
                if (vol_frac > epsilon)
                {
                    matset_values.push_back(matset_val);
                    size ++;
                }
            }
        }

        dest_field["matset_values"].set(matset_values.data(), matset_values.size());
    }
    else
    {
        dest_field.set(src_field);
    }
}

//-----------------------------------------------------------------------------
// venn sparse by element -> full
void
uni_buffer_by_element_to_multi_buffer_by_element_matset(const conduit::Node &src_matset,
                                                        conduit::Node &dest_matset)
{
    dest_matset.reset();

    // set the topology
    dest_matset["topology"].set(src_matset["topology"]);

    // map material numbers to material names
    std::map<int, std::string> reverse_matmap;
    create_reverse_matmap(src_matset, reverse_matmap);

    // get ptr to vol fracs and mat ids
    float64_accessor volume_fractions = src_matset["volume_fractions"].value();
    int64_accessor material_ids = src_matset["material_ids"].value();
    
    // create container for new matset vals
    std::map<std::string, std::vector<float64>> new_vol_fracs;

    walk_uni_buffer_by_element_to_multi_buffer_by_element(src_matset,
                                                          reverse_matmap,
                                                          volume_fractions,
                                                          material_ids,
                                                          new_vol_fracs);

    read_from_map_write_out(new_vol_fracs, dest_matset["volume_fractions"]);
}

//-----------------------------------------------------------------------------
// venn sparse by element -> full
void
uni_buffer_by_element_to_multi_buffer_by_element_field(const conduit::Node &src_matset,
                                                       const conduit::Node &src_field,
                                                       const std::string &dest_matset_name,
                                                       conduit::Node &dest_field)
{
    dest_field.reset();

    // if this field is material dependent
    if (src_field.has_child("matset_values"))
    {
        copy_matset_independent_parts_of_field(src_field,
                                               dest_matset_name,
                                               dest_field);

        // map material numbers to material names
        std::map<int, std::string> reverse_matmap;
        create_reverse_matmap(src_matset, reverse_matmap);

        // get ptr to matset values and mat ids
        float64_accessor matset_values = src_field["matset_values"].value();
        int64_accessor material_ids = src_matset["material_ids"].value();

        // create container for new matset vals and initialize sizes
        std::map<std::string, std::vector<float64>> new_matset_vals;

        walk_uni_buffer_by_element_to_multi_buffer_by_element(src_matset,
                                                              reverse_matmap,
                                                              matset_values,
                                                              material_ids,
                                                              new_matset_vals);

        read_from_map_write_out(new_matset_vals, dest_field["matset_values"]);
    }
    else
    {
        dest_field.set(src_field);
    }
}

//-----------------------------------------------------------------------------
// venn sparse by element -> sparse by material
void
uni_buffer_by_element_to_multi_buffer_by_material_matset(const conduit::Node &src_matset,
                                                         conduit::Node &dest_matset)
{
    dest_matset.reset();

    // set the topology
    dest_matset["topology"].set(src_matset["topology"]);

    // map material numbers to material names
    std::map<int, std::string> reverse_matmap;
    create_reverse_matmap(src_matset, reverse_matmap);

    // get ptr to vol fracs and mat ids
    float64_accessor volume_fractions = src_matset["volume_fractions"].value();
    int64_accessor material_ids = src_matset["material_ids"].value();
    
    // create containers for new vol fracs and elem ids
    std::map<std::string, std::vector<float64>> new_vol_fracs;
    std::map<std::string, std::vector<int64>> new_elem_ids;

    // iterate through matset
    auto o2m_idx = o2mrelation::O2MIndex(src_matset);
    for (int elem_id = 0; elem_id < o2m_idx.size(); elem_id ++)
    {
        for (int many_id = 0; many_id < o2m_idx.size(elem_id); many_id ++)
        {
            index_t data_index = o2m_idx.index(elem_id, many_id);

            float64 vol_frac = volume_fractions[data_index];
            int64 mat_id = material_ids[data_index];
            const std::string &matname = reverse_matmap[mat_id];
            
            new_vol_fracs[matname].push_back(vol_frac);
            new_elem_ids[matname].push_back(elem_id);
        }
    }

    read_from_map_write_out(new_vol_fracs, dest_matset["volume_fractions"]);
    read_from_map_write_out(new_elem_ids, dest_matset["element_ids"]);
}

//-----------------------------------------------------------------------------
// venn sparse by element -> sparse by material
void
uni_buffer_by_element_to_multi_buffer_by_material_field(const conduit::Node &src_matset,
                                                        const conduit::Node &src_field,
                                                        const std::string &dest_matset_name,
                                                        conduit::Node &dest_field)
{
    dest_field.reset();

    // if this field is material dependent
    if (src_field.has_child("matset_values"))
    {
        copy_matset_independent_parts_of_field(src_field,
                                               dest_matset_name,
                                               dest_field);

        // map material numbers to material names
        std::map<int, std::string> reverse_matmap;
        create_reverse_matmap(src_matset, reverse_matmap);

        // get ptr to matset values and mat ids
        float64_accessor matset_values = src_field["matset_values"].value();
        int64_accessor material_ids = src_matset["material_ids"].value();

        // create container for new matset vals
        std::map<std::string, std::vector<float64>> new_mset_vals;

        auto o2m_idx = o2mrelation::O2MIndex(src_matset);
        for (int elem_id = 0; elem_id < o2m_idx.size(); elem_id ++)
        {
            for (int many_id = 0; many_id < o2m_idx.size(elem_id); many_id ++)
            {
                index_t data_index = o2m_idx.index(elem_id, many_id);

                float64 mset_val = matset_values[data_index];
                int64 mat_id = material_ids[data_index];
                const std::string &matname = reverse_matmap[mat_id];
                
                new_mset_vals[matname].push_back(mset_val);
            }
        }

        read_from_map_write_out(new_mset_vals, dest_field["matset_values"]);
    }
    else
    {
        dest_field.set(src_field);
    }
}

//-----------------------------------------------------------------------------
// venn full -> sparse_by_material
void
multi_buffer_by_element_to_multi_buffer_by_material_matset(const conduit::Node &src_matset,
                                                           conduit::Node &dest_matset,
                                                           const float64 epsilon)
{
    dest_matset.reset();

    // set the topology
    dest_matset["topology"].set(src_matset["topology"]);
    
    auto mat_itr = src_matset["volume_fractions"].children();
    while (mat_itr.has_next())
    {
        const Node &mat_vol_fracs = mat_itr.next();
        std::string matname = mat_itr.name();

        std::vector<float64> vol_fracs;
        std::vector<int64> elem_ids;

        float64_accessor full_vol_fracs = mat_vol_fracs.value();
        int num_elems = full_vol_fracs.dtype().number_of_elements();
        for (int elem_id = 0; elem_id < num_elems; elem_id ++)
        {
            if (full_vol_fracs[elem_id] > epsilon)
            {
                vol_fracs.push_back(full_vol_fracs[elem_id]);
                elem_ids.push_back(elem_id);
            }
        }

        dest_matset["volume_fractions"][matname].set(vol_fracs.data(), vol_fracs.size());
        dest_matset["element_ids"][matname].set(elem_ids.data(), elem_ids.size());
    }
}

//-----------------------------------------------------------------------------
// venn full -> sparse_by_material
void
multi_buffer_by_element_to_multi_buffer_by_material_field(const conduit::Node &src_matset,
                                                          const conduit::Node &src_field,
                                                          const std::string &dest_matset_name,
                                                          conduit::Node &dest_field,
                                                          const float64 epsilon)
{
    dest_field.reset();

    // if this field is material dependent
    if (src_field.has_child("matset_values"))
    {
        copy_matset_independent_parts_of_field(src_field,
                                               dest_matset_name,
                                               dest_field);

        auto mat_itr = src_matset["volume_fractions"].children();
        auto fmat_itr = src_field["matset_values"].children();
        while (mat_itr.has_next() && fmat_itr.has_next())
        {
            const Node &mat_vol_fracs = mat_itr.next();
            std::string matname = mat_itr.name();
            
            const Node &mat_vals = fmat_itr.next();
            std::string fmatname = fmat_itr.name();

            CONDUIT_ASSERT(matname == fmatname, "Materials must be ordered the same in "
                "material dependent fields and their matsets.");

            std::vector<float64> mset_vals;

            float64_accessor full_vol_fracs = mat_vol_fracs.value();
            float64_accessor full_mset_vals = mat_vals.value();
            int num_elems = full_vol_fracs.dtype().number_of_elements();
            for (int elem_id = 0; elem_id < num_elems; elem_id ++)
            {
                if (full_vol_fracs[elem_id] > epsilon)
                {
                    mset_vals.push_back(full_mset_vals[elem_id]);
                }
            }

            dest_field["matset_values"][matname].set(mset_vals.data(), mset_vals.size());
        }
    }
    else
    {
        dest_field.set(src_field);
    }
}

//-----------------------------------------------------------------------------
// venn sparse by material -> full
void
multi_buffer_by_material_to_multi_buffer_by_element_matset(const conduit::Node &src_matset,
                                                           conduit::Node &dest_matset)
{
    dest_matset.reset();

    // set the topology
    dest_matset["topology"].set(src_matset["topology"]);

    // sparse by material representation
    // we map material names to volume fractions and element ids
    std::map<std::string, std::pair<int64_accessor, float64_accessor>> sbm_rep;

    create_sbm_rep(src_matset["element_ids"], src_matset["volume_fractions"], sbm_rep);

    const int num_elems = determine_num_elems_in_multi_buffer_by_material(src_matset["element_ids"]);

    sbm_rep_to_full(sbm_rep, num_elems, dest_matset["volume_fractions"]);
}

//-----------------------------------------------------------------------------
// venn sparse by material -> full
void
multi_buffer_by_material_to_multi_buffer_by_element_field(const conduit::Node &src_matset,
                                                          const conduit::Node &src_field,
                                                          const std::string &dest_matset_name,
                                                          conduit::Node &dest_field)
{
    dest_field.reset();

    // if this field is material dependent
    if (src_field.has_child("matset_values"))
    {
        copy_matset_independent_parts_of_field(src_field,
                                               dest_matset_name,
                                               dest_field);

        // sparse by material representation
        // we map material names to element ids and matset values
        std::map<std::string, std::pair<int64_accessor, float64_accessor>> sbm_rep;

        create_sbm_rep(src_matset["element_ids"], src_field["matset_values"], sbm_rep);

        const int num_elems = determine_num_elems_in_multi_buffer_by_material(src_matset["element_ids"]);

        sbm_rep_to_full(sbm_rep, num_elems, dest_field["matset_values"]);
    }
    else
    {
        dest_field.set(src_field);
    }
}

//-----------------------------------------------------------------------------
// venn sparse by material -> sparse by element
void
multi_buffer_by_material_to_uni_buffer_by_element_matset(const conduit::Node &src_matset,
                                                         conduit::Node &dest_matset)
{
    dest_matset.reset();

    // set the topology
    dest_matset["topology"].set(src_matset["topology"]);

    // sparse by material representation
    // we map material names to volume fractions and element ids
    std::map<std::string, std::pair<float64_accessor, int64_accessor>> sbm_rep;
    std::map<std::string, int64> matmap;

    int64 mat_id = 0;
    auto vf_itr = src_matset["volume_fractions"].children();
    while (vf_itr.has_next())
    {
        const Node &mat_vol_fracs = vf_itr.next();
        const std::string matname = vf_itr.name();
        sbm_rep[matname].first = mat_vol_fracs.value();
        dest_matset["material_map"][matname] = mat_id;
        matmap[matname] = mat_id;
        mat_id ++;
    }

    auto eid_itr = src_matset["element_ids"].children();
    while (eid_itr.has_next())
    {
        const Node &mat_elem_ids = eid_itr.next();
        const std::string matname = eid_itr.name();
        sbm_rep[matname].second = mat_elem_ids.value();
    }

    const int num_elems = determine_num_elems_in_multi_buffer_by_material(src_matset["element_ids"]);

    // There is no way to pack the volume fractions correctly without
    // first knowing the sizes. So we create an intermediate representation
    // in which volume fractions are packed by element. Later we smooth this out.
    std::vector<std::vector<float64>> intermediate_vol_fracs(num_elems);
    std::vector<std::vector<int64>> intermediate_mat_ids(num_elems);

    for (auto &mapitem : sbm_rep)
    {
        const std::string &matname = mapitem.first;
        int64 mat_id = matmap[matname];

        float64_accessor sbm_vfs = std::get<0>(mapitem.second);
        int64_accessor sbm_eids = std::get<1>(mapitem.second);
        
        int num_vf = sbm_vfs.dtype().number_of_elements();
        for (int mat_vf_id = 0; mat_vf_id < num_vf; mat_vf_id ++)
        {
            int elem_id = sbm_eids[mat_vf_id];
            float64 vol_frac = sbm_vfs[mat_vf_id];

            intermediate_vol_fracs[elem_id].push_back(vol_frac);
            intermediate_mat_ids[elem_id].push_back(mat_id);
        }
    }

    std::vector<float64> vol_fracs;
    std::vector<int64> mat_ids;
    std::vector<int64> sizes;
    std::vector<int64> offsets;

    // final pass
    int64 offset = 0;
    for (int elem_id = 0; elem_id < num_elems; elem_id ++)
    {
        int64 size = static_cast<int64>(intermediate_vol_fracs[elem_id].size());
        for (int64 mat_vf_id = 0; mat_vf_id < size; mat_vf_id ++)
        {
            vol_fracs.push_back(intermediate_vol_fracs[elem_id][mat_vf_id]);
            mat_ids.push_back(intermediate_mat_ids[elem_id][mat_vf_id]);
        }
        sizes.push_back(size);
        offsets.push_back(offset);
        offset += size;
    }

    dest_matset["volume_fractions"].set(vol_fracs.data(), vol_fracs.size());
    dest_matset["material_ids"].set(mat_ids.data(), mat_ids.size());
    dest_matset["sizes"].set(sizes.data(), sizes.size());
    dest_matset["offsets"].set(offsets.data(), offsets.size());
}

//-----------------------------------------------------------------------------
// venn sparse by material -> sparse by element
void
multi_buffer_by_material_to_uni_buffer_by_element_field(const conduit::Node &src_matset,
                                                        const conduit::Node &src_field,
                                                        const std::string &dest_matset_name,
                                                        conduit::Node &dest_field)
{
    dest_field.reset();

    // if this field is material dependent
    if (src_field.has_child("matset_values"))
    {
        copy_matset_independent_parts_of_field(src_field,
                                               dest_matset_name,
                                               dest_field);

        // sparse by material representation
        // we map material names to element ids and matset values
        std::map<std::string, std::pair<int64_accessor, float64_accessor>> sbm_rep;

        auto eid_itr = src_matset["element_ids"].children();
        while (eid_itr.has_next())
        {
            const Node &mat_elem_ids = eid_itr.next();
            const std::string matname = eid_itr.name();
            sbm_rep[matname].first = mat_elem_ids.value();
        }

        auto mvals_itr = src_field["matset_values"].children();
        while (mvals_itr.has_next())
        {
            const Node &matset_vals = mvals_itr.next();
            const std::string matname = mvals_itr.name();
            sbm_rep[matname].second = matset_vals.value();
        }

        const int num_elems = determine_num_elems_in_multi_buffer_by_material(src_matset["element_ids"]);

        // There is no way to pack the matset values correctly without
        // first knowing the sizes. So we create an intermediate representation
        // in which matset values are packed by element. Later we smooth this out.
        std::vector<std::vector<float64>> intermediate_mset_vals(num_elems);

        for (auto &mapitem : sbm_rep)
        {
            int64_accessor sbm_eids = mapitem.second.first;
            float64_accessor sbm_mvals = mapitem.second.second;
            
            int num_vf = sbm_mvals.dtype().number_of_elements();
            for (int mat_vf_id = 0; mat_vf_id < num_vf; mat_vf_id ++)
            {
                int64 elem_id = sbm_eids[mat_vf_id];
                float64 mset_val = sbm_mvals[mat_vf_id];

                intermediate_mset_vals[elem_id].push_back(mset_val);
            }
        }

        std::vector<float64> mset_vals;

        // final pass
        for (int elem_id = 0; elem_id < num_elems; elem_id ++)
        {
            int size = static_cast<int>(intermediate_mset_vals[elem_id].size());
            for (int mat_vf_id = 0; mat_vf_id < size; mat_vf_id ++)
            {
                mset_vals.push_back(intermediate_mset_vals[elem_id][mat_vf_id]);
            }
        }

        dest_field["matset_values"].set(mset_vals.data(), mset_vals.size());
    }
    else
    {
        dest_field.set(src_field);
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
void
to_multi_buffer_full(const conduit::Node &src_matset,
                     conduit::Node &dest_matset)
{
    // extra seat belt here
    if (! src_matset.dtype().is_object())
    {
        CONDUIT_ERROR("blueprint::mesh::matset::to_multi_buffer_full"
                      " passed matset node must be a valid matset tree.");
    }

    // full
    if (is_element_dominant(src_matset) && is_multi_buffer(src_matset))
    {
        // nothing to do
        dest_matset.set(src_matset);
    }
    // sparse_by_element
    else if (is_element_dominant(src_matset))
    {
        detail::uni_buffer_by_element_to_multi_buffer_by_element_matset(src_matset, 
                                                                        dest_matset);
    }
    // sparse_by_material
    else if (is_material_dominant(src_matset))
    {
        detail::multi_buffer_by_material_to_multi_buffer_by_element_matset(src_matset,
                                                                           dest_matset);
    }
    else
    {
        CONDUIT_ERROR("Unknown matset type.");
    }
}

//-----------------------------------------------------------------------------
void
to_sparse_by_element(const conduit::Node &src_matset,
                     conduit::Node &dest_matset,
                     const float64 epsilon)
{
    // extra seat belt here
    if (! src_matset.dtype().is_object())
    {
        CONDUIT_ERROR("blueprint::mesh::matset::to_sparse_by_element"
                      " passed matset node must be a valid matset tree.");
    }

    // full
    if (is_element_dominant(src_matset) && is_multi_buffer(src_matset))
    {
        detail::multi_buffer_by_element_to_uni_buffer_by_element_matset(src_matset, 
                                                                        dest_matset, 
                                                                        epsilon);
    }
    // sparse_by_element
    else if (is_element_dominant(src_matset))
    {
        // nothing to do
        dest_matset.set(src_matset);
    }
    // sparse_by_material
    else if (is_material_dominant(src_matset))
    {
        detail::multi_buffer_by_material_to_uni_buffer_by_element_matset(src_matset,
                                                                         dest_matset);
    }
    else
    {
        CONDUIT_ERROR("Unknown matset type.");
    }
}

//-----------------------------------------------------------------------------
void
to_multi_buffer_by_material(const conduit::Node &src_matset,
                            conduit::Node &dest_matset,
                            const float64 epsilon)
{
    // extra seat belt here
    if (! src_matset.dtype().is_object())
    {
        CONDUIT_ERROR("blueprint::mesh::matset::to_multi_buffer_by_material"
                      " passed matset node must be a valid matset tree.");
    }

    // full
    if (is_element_dominant(src_matset) && is_multi_buffer(src_matset))
    {
        detail::multi_buffer_by_element_to_multi_buffer_by_material_matset(src_matset, 
                                                                           dest_matset, 
                                                                           epsilon);
    }
    // sparse_by_element
    else if (is_element_dominant(src_matset))
    {
        detail::uni_buffer_by_element_to_multi_buffer_by_material_matset(src_matset,
                                                                         dest_matset);
    }
    // sparse_by_material
    else if (is_material_dominant(src_matset))
    {
        // nothing to do
        dest_matset.set(src_matset);
    }
    else
    {
        CONDUIT_ERROR("Unknown matset type.");
    }
}

//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::matset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::specset --
//-----------------------------------------------------------------------------
namespace specset
{
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void
to_silo(const conduit::Node &specset,
        const conduit::Node &matset,
        conduit::Node &dest)
{
    // extra seat belts here b/c we want to avoid folks entering
    // the detail version of to_silo with surprising results.

    if(!specset.dtype().is_object() )
    {
        CONDUIT_ERROR("blueprint::mesh::specset::to_silo passed specset node"
                      " must be a valid specset tree.");
    }

    if(!matset.dtype().is_object() )
    {
        CONDUIT_ERROR("blueprint::mesh::specset::to_silo passed matset node"
                      " must be a valid matset tree.");
        // TODO well, not entirely true...
    }

    // need to check if passed matset is already in the silo rep
    Node silo_matset;

    if (! (matset.has_child("topology") && 
           matset.has_child("material_map") &&
           matset.has_child("mix_vf") && 
           matset.has_child("mix_mat") &&
           matset.has_child("mix_next") &&
           matset.has_child("matlist")))
    {
        // if not, create a silo rep
        conduit::blueprint::mesh::matset::to_silo(matset, silo_matset);
    }
    else
    {
        // if it is, use it and continue
        silo_matset.set_external(matset);
    }

    const int nmat = silo_matset["material_map"].number_of_children();
    CONDUIT_ASSERT(nmat == specset["matset_values"].number_of_children(),
        "blueprint::mesh::specset::to_silo number of materials must match "
        "between passed specset and passed matset.");

    auto matset_vals_itr = specset["matset_values"].children();
    auto matmap_itr = silo_matset["material_map"].children();
    int matmap_index = 0;
    // Map actual material numbers to indicies into the material map
    // We need this map so that, no matter what material numbers we get thrown at us,
    // we can figure out their order in the material map for when we calculate
    // species indices.
    std::map<int, int> matmap_map;
    while (matset_vals_itr.has_next() && matmap_itr.has_next())
    {
        matset_vals_itr.next();
        const Node &matmap_entry = matmap_itr.next();
        // Make sure that materials are in the same order across the specset and matset
        CONDUIT_ASSERT(matset_vals_itr.name() == matmap_itr.name(), 
            "blueprint::mesh::specset::to_silo materials must be in the same order "
            "between passed specset and passed matset.");
        matmap_map[matmap_entry.as_int()] = matmap_index;
        matmap_index ++;
    }

    std::vector<int> nmatspec;
    matset_vals_itr.to_front();
    while (matset_vals_itr.has_next())
    {
        // get the number of species for this material
        const Node &individual_mat_spec = matset_vals_itr.next();
        const std::string matname = matset_vals_itr.name();
        const int num_species_for_this_material = individual_mat_spec.number_of_children();
        nmatspec.push_back(num_species_for_this_material);

        // get the specie names for this material
        auto spec_itr = individual_mat_spec.children();
        while (spec_itr.has_next())
        {
            spec_itr.next();
            const std::string specname = spec_itr.name();
            Node &specname_entry = dest["specnames"].append();
            specname_entry.set(specname);
        }
    }

    // we sum up the nmatspec to get the number of species across all materials
    const int num_species_across_mats = std::accumulate(nmatspec.begin(), 
                                                        nmatspec.end(),
                                                        decltype(nmatspec)::value_type(0));

    // we have to go in order by zones as they appear

    // first we need number of zones
    const int nzones = silo_matset["matlist"].dtype().number_of_elements();

    // TODO
    // I may wish to go through and check if the material is even in the zone
    // to avoid writing unneeded data
    // that could be expensive though

    std::vector<float64> species_mf;
    
    // need to iterate across all species for all materials at once
    for (int zoneId = 0; zoneId < nzones; zoneId ++)
    {
        // iterate through each material
        matset_vals_itr.to_front();
        while (matset_vals_itr.has_next())
        {
            const Node &individual_mat_spec = matset_vals_itr.next();
            // iterate through each specie
            auto spec_itr = individual_mat_spec.children();
            while (spec_itr.has_next())
            {
                const Node &spec = spec_itr.next();
                float64_accessor species_mass_fractions = spec.value();
                // grab the specie mass fraction for this zone id
                species_mf.push_back(species_mass_fractions[zoneId]);
            }
        }
    }

    const int nspecies_mf = static_cast<int>(species_mf.size());

    // get pointers to the silo material representation data
    const int_accessor silo_matlist = silo_matset["matlist"].value();
    const int_accessor silo_mix_mat = silo_matset["mix_mat"].value();
    const int_accessor silo_mix_next = silo_matset["mix_next"].value();

    std::vector<int> speclist;
    std::vector<int> mix_spec;

    auto calculate_species_index = [&](const int zoneId, const int mat_index)
    {
        // To get the value for the speclist for this zone, we must determine
        // the correct 1-index in the species_mf array that corresponds to the 
        // material in this zone. We have organized the species_mf array such 
        // that there are entries for each material's species for each zone,
        // even if those materials are not present in that zone. Thus there are
        // the same number of species entries for each zone in the species_mf
        // array. So we need to determine what I am calling an "outer_index" 
        // that tells us the starting index of the current zone in the species_mf
        // array.

        // how many entries per zone? Use the calculated num_species_across_mats
        const int outer_index = zoneId * num_species_across_mats;

        // Next we need the inner or "local_index", which corresponds to the 
        // starting 1-index of the relevant material's species within this zone.
        // We can use the nmatspec array to determine where that starts for our
        // given material, which we fetch via material number, which we have used
        // to get an index into the nmatspec array.

        // We wish to offset the local index by 1, hence starting from 1 in our call
        // to std::accumulate.

        // local index is the number of species for each material
        // BEFORE this material plus 1, since it is 1 indexed.
        // So if mat0 has 2 species and mat1 has 3 species, then
        // the 1-index start of mat2 will be 2 + 3 + 1 = 6.
        const int local_index = std::accumulate(nmatspec.begin(),
                                                nmatspec.begin() + mat_index,
                                                decltype(nmatspec)::value_type(1));

        // we save the final index for this zone
        return outer_index + local_index;

        // TODO return a zero here if the material in the zone contains only 1 species.
        // This is a further optimization. It doesn't matter for now since we are
        // treating our output like we have all species and all materials in all zones.
        // I think the point of this optimization is to also leave stuff out of the 
        // species_mf array. If I want to do this then I should explore that as well.
    };

    // our negative 1-index into the mix_spec array
    int mix_start_index = -1;

    // now we create the speclist and mix_spec arrays, traversing through the zones
    for (int zoneId = 0; zoneId < nzones; zoneId ++)
    {
        const int matlist_entry = silo_matlist[zoneId];
        // is this zone clean?
        if (matlist_entry >= 0) // this relies on matset_ptr->allowmat0 == 0
        {
            // clean

            // I can use the material number to determine which part of the speclist to index into
            const int &matno = matlist_entry;
            const int mat_index = matmap_map[matno];
            speclist.push_back(calculate_species_index(zoneId, mat_index));
        }
        else
        {
            // mixed

            // we save the negated 1-index into the mix_spec array
            speclist.push_back(mix_start_index);

            // for mixed zones, the numbers in the matlist are negated 1-indices into
            // the silo mixed data arrays. To turn them into zero-indices, we must add
            // 1 and negate the result. Example:
            // indices: -1 -2 -3 -4 ...
            // become:   0  1  2  3 ...

            int mix_id = -1 * (matlist_entry + 1);

            // when silo_mix_next[mix_id] is 0, we are on the last one
            while (mix_id >= 0)
            {                
                // I can use the material number to determine which part of the speclist to index into
                const int matno = silo_mix_mat[mix_id];
                const int mat_index = matmap_map[matno];
                mix_spec.push_back(calculate_species_index(zoneId, mat_index));

                // since mix_id is a 1-index, we must subtract one
                // this makes sure that mix_id = 0 is the last case,
                // since it will make our mix_id == -1, which ends
                // the while loop.
                mix_id = silo_mix_next[mix_id] - 1;

                // decrement this index every time we write another index to the mix_spec
                mix_start_index --;
            }
        }
    }

    // get the length of the mixed data arrays
    const int mixlen = static_cast<int>(mix_spec.size());

    // number of materials
    dest["nmat"] = nmat;
    // number of species associated with each material
    dest["nmatspec"].set(nmatspec.data(), nmatspec.size());
    // indices into species_mf and mix_spec
    dest["speclist"].set(speclist.data(), speclist.size());
    // length of the species_mf array
    dest["nspecies_mf"] = nspecies_mf;
    // mass fractions of the matspecies in an array of length nspecies_mf
    dest["species_mf"].set(species_mf.data(), species_mf.size());
    // array of length mixlen containing indices into the species_mf array
    dest["mix_spec"].set(mix_spec.data(), mix_spec.size());
    // length of mix_spec array
    dest["mixlen"] = mixlen;
    // we already saved species names
}

//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::specset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::field --
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
void
to_multi_buffer_full(const conduit::Node &src_matset,
                     const conduit::Node &src_field,
                     const std::string &dest_matset_name,
                     conduit::Node &dest_field)
{
    // extra seat belt here
    if (! src_matset.dtype().is_object())
    {
        CONDUIT_ERROR("blueprint::mesh::field::to_multi_buffer_full"
                      " passed matset node must be a valid matset tree.");
    }

    if (! src_field.dtype().is_object())
    {
        CONDUIT_ERROR("blueprint::mesh::field::to_multi_buffer_full"
                      " passed field node must be a valid field tree.");
    }

    // full
    if (conduit::blueprint::mesh::matset::is_element_dominant(src_matset) && 
        conduit::blueprint::mesh::matset::is_multi_buffer(src_matset))
    {
        // nothing to do
        dest_field.set(src_field);
        dest_field["matset"].reset();
        dest_field["matset"] = dest_matset_name;
    }
    // sparse_by_element
    else if (conduit::blueprint::mesh::matset::is_element_dominant(src_matset))
    {
        conduit::blueprint::mesh::matset::detail::uni_buffer_by_element_to_multi_buffer_by_element_field(
            src_matset, src_field, dest_matset_name, dest_field);
    }
    // sparse_by_material
    else if (conduit::blueprint::mesh::matset::is_material_dominant(src_matset))
    {
        conduit::blueprint::mesh::matset::detail::multi_buffer_by_material_to_multi_buffer_by_element_field(
            src_matset, src_field, dest_matset_name, dest_field);
    }
    else
    {
        CONDUIT_ERROR("Unknown matset type.");
    }
}

//-----------------------------------------------------------------------------
void
to_sparse_by_element(const conduit::Node &src_matset,
                     const conduit::Node &src_field,
                     const std::string &dest_matset_name,
                     conduit::Node &dest_field,
                     const float64 epsilon)
{
    // extra seat belt here
    if (! src_matset.dtype().is_object())
    {
        CONDUIT_ERROR("blueprint::mesh::field::to_sparse_by_element"
                      " passed matset node must be a valid matset tree.");
    }

    if (! src_field.dtype().is_object())
    {
        CONDUIT_ERROR("blueprint::mesh::field::to_sparse_by_element"
                      " passed field node must be a valid field tree.");
    }

    // full
    if (conduit::blueprint::mesh::matset::is_element_dominant(src_matset) && 
        conduit::blueprint::mesh::matset::is_multi_buffer(src_matset))
    {
        conduit::blueprint::mesh::matset::detail::multi_buffer_by_element_to_uni_buffer_by_element_field(
            src_matset, src_field, dest_matset_name, dest_field, epsilon);
    }
    // sparse_by_element
    else if (conduit::blueprint::mesh::matset::is_element_dominant(src_matset))
    {
        // nothing to do
        dest_field.set(src_field);
        dest_field["matset"].reset();
        dest_field["matset"] = dest_matset_name;
    }
    // sparse_by_material
    else if (conduit::blueprint::mesh::matset::is_material_dominant(src_matset))
    {
        conduit::blueprint::mesh::matset::detail::multi_buffer_by_material_to_uni_buffer_by_element_field(
            src_matset, src_field, dest_matset_name, dest_field);
    }
    else
    {
        CONDUIT_ERROR("Unknown matset type.");
    }
}

//-----------------------------------------------------------------------------
void
to_multi_buffer_by_material(const conduit::Node &src_matset,
                            const conduit::Node &src_field,
                            const std::string &dest_matset_name,
                            conduit::Node &dest_field,
                            const float64 epsilon)
{
    // extra seat belt here
    if (! src_matset.dtype().is_object())
    {
        CONDUIT_ERROR("blueprint::mesh::field::to_multi_buffer_by_material"
                      " passed matset node must be a valid matset tree.");
    }

    if (! src_field.dtype().is_object())
    {
        CONDUIT_ERROR("blueprint::mesh::field::to_multi_buffer_by_material"
                      " passed field node must be a valid field tree.");
    }

    // full
    if (conduit::blueprint::mesh::matset::is_element_dominant(src_matset) && 
        conduit::blueprint::mesh::matset::is_multi_buffer(src_matset))
    {
        conduit::blueprint::mesh::matset::detail::multi_buffer_by_element_to_multi_buffer_by_material_field(
            src_matset, src_field, dest_matset_name, dest_field, epsilon);
    }
    // sparse_by_element
    else if (conduit::blueprint::mesh::matset::is_element_dominant(src_matset))
    {
        conduit::blueprint::mesh::matset::detail::uni_buffer_by_element_to_multi_buffer_by_material_field(
            src_matset, src_field, dest_matset_name, dest_field);
    }
    // sparse_by_material
    else if (conduit::blueprint::mesh::matset::is_material_dominant(src_matset))
    {
        // nothing to do
        dest_field.set(src_field);
        dest_field["matset"].reset();
        dest_field["matset"] = dest_matset_name;
    }
    else
    {
        CONDUIT_ERROR("Unknown matset type.");
    }
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

