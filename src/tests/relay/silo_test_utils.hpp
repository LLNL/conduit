// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: silo_test_utils.hpp
///
//-----------------------------------------------------------------------------

#ifndef SILO_TEST_UTILS_HPP
#define SILO_TEST_UTILS_HPP

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "t_config.hpp"

using namespace conduit;
using namespace conduit::utils;

// Some utility functions for SILO tests.

//-----------------------------------------------------------------------------
std::string
relay_test_silo_data_path(const std::string &test_fname)
{
    std::string res = utils::join_path(CONDUIT_T_SRC_DIR, "relay");
    res = utils::join_file_path(res, "data");
    res = utils::join_file_path(res, "silo");
    return utils::join_file_path(res, test_fname);
}

//-----------------------------------------------------------------------------
// The Blueprint to Silo to Blueprint round trip will
// transform uniform to rectilinear so we will do the same
// to allow diffs to succeed
void
silo_uniform_to_rect_conversion(const std::string &coordset_name,
                                const std::string &topo_name,
                                Node &save_mesh)
{
    Node save_mesh_rect;
    Node &save_mesh_rect_coords = save_mesh_rect["coordsets"][coordset_name];
    Node &save_mesh_rect_topo = save_mesh_rect["topologies"][topo_name];
    blueprint::mesh::topology::uniform::to_rectilinear(
        save_mesh["topologies"][topo_name], 
        save_mesh_rect_topo, save_mesh_rect_coords);
    save_mesh["topologies"][topo_name].set(save_mesh_rect_topo);
    save_mesh["coordsets"][coordset_name].set(save_mesh_rect_coords);
}

//-----------------------------------------------------------------------------
// The Blueprint to Silo transformation changes several names 
// and some information is lost. We manually make changes so 
// that the diff will pass.
void
silo_name_changer(const std::string &mmesh_name,
                  Node &save_mesh)
{
    std::map<std::string, std::string> old_to_new_names;

    if (!save_mesh.has_path("state/domain_id"))
    {
        save_mesh["state"]["domain_id"] = 0;
    }

    if (save_mesh.has_child("topologies"))
    {
        auto topo_itr = save_mesh["topologies"].children();
        while(topo_itr.has_next())
        {
            Node &n_topo = topo_itr.next();
            std::string topo_name = topo_itr.name();
            std::string new_topo_name = mmesh_name + "_" + topo_name;

            old_to_new_names[topo_name] = new_topo_name;

            std::string coordset_name = n_topo["coordset"].as_string();
            std::string new_coordset_name = mmesh_name + "_" + topo_name;

            // change the coordset this topo refers to
            save_mesh["topologies"][topo_name]["coordset"].reset();
            save_mesh["topologies"][topo_name]["coordset"] = new_coordset_name;

            // change the name of the topo
            save_mesh["topologies"].rename_child(topo_name, new_topo_name);

            // change the name of the coordset
            if (save_mesh.has_path("coordsets/" + coordset_name))
            {
                save_mesh["coordsets"].rename_child(coordset_name, new_coordset_name);
            }
        }
    }

    if (save_mesh.has_child("matsets"))
    {
        auto matset_itr = save_mesh["matsets"].children();
        while (matset_itr.has_next())
        {
            Node &n_matset = matset_itr.next();
            std::string matset_name = matset_itr.name();

            std::string old_topo_name = n_matset["topology"].as_string();

            if (old_to_new_names.find(old_topo_name) == old_to_new_names.end())
            {
                continue;
                // If this is the case, we probably need to delete this matset.
                // But our job in this function is just to rename things, so we 
                // will just skip.
            }
            std::string new_topo_name = old_to_new_names[old_topo_name];

            // use new topo name
            n_matset["topology"].reset();
            n_matset["topology"] = new_topo_name;

            // come up with new matset name
            std::string new_matset_name = mmesh_name + "_" + matset_name;

            old_to_new_names[matset_name] = new_matset_name;

            // rename the matset
            save_mesh["matsets"].rename_child(matset_name, new_matset_name);
        }
    }

    if (save_mesh.has_child("fields"))
    {
        auto field_itr = save_mesh["fields"].children();
        while (field_itr.has_next())
        {
            Node &n_field = field_itr.next();
            std::string field_name = field_itr.name();

            std::string old_topo_name = n_field["topology"].as_string();
            if (old_to_new_names.find(old_topo_name) == old_to_new_names.end())
            {
                continue;
                // If this is the case, we probably need to delete this field.
                // But our job in this function is just to rename things, so we 
                // will just skip.
            }
            std::string new_topo_name = old_to_new_names[old_topo_name];
            // use new topo name
            n_field["topology"].reset();
            n_field["topology"] = new_topo_name;

            if (n_field.has_child("matset"))
            {
                std::string old_matset_name = n_field["matset"].as_string();
                if (old_to_new_names.find(old_matset_name) == old_to_new_names.end())
                {
                    continue;
                    // If this is the case, we probably need to delete this field.
                    // But our job in this function is just to rename things, so we 
                    // will just skip.
                }
                std::string new_matset_name = old_to_new_names[old_matset_name];
                // use new topo name
                n_field["matset"].reset();
                n_field["matset"] = new_matset_name;
            }

            // remove vol dep
            if (n_field.has_child("volume_dependent"))
            {
                n_field.remove_child("volume_dependent");
            }

            // we need to rename vector components
            if (n_field["values"].dtype().is_object())
            {
                if (n_field["values"].number_of_children() > 0)
                {
                    int child_index = 0;
                    auto val_itr = n_field["values"].children();
                    while (val_itr.has_next())
                    {
                        val_itr.next();
                        std::string comp_name = val_itr.name();

                        // rename vector components
                        n_field["values"].rename_child(comp_name, std::to_string(child_index));

                        child_index ++;
                    }
                }
            }

            // come up with new field name
            std::string new_field_name = mmesh_name + "_" + field_name;

            // rename the field
            save_mesh["fields"].rename_child(field_name, new_field_name);
        }
    }
}

//-----------------------------------------------------------------------------
// The Blueprint to Overlink transformation changes several names 
// and some information is lost. We manually make changes so 
// that the diff will pass.
void
overlink_name_changer(conduit::Node &save_mesh)
{
    // we assume 1 coordset, 1 topo, and fields

    Node &coordsets = save_mesh["coordsets"];
    Node &topologies = save_mesh["topologies"];
    Node &fields = save_mesh["fields"];

    // we assume only 1 child for each
    std::string coordset_name = coordsets.children().next().name();
    std::string topo_name = topologies.children().next().name();

    // rename the coordset and references to it
    coordsets.rename_child(coordset_name, "MMESH");
    topologies[topo_name]["coordset"].reset();
    topologies[topo_name]["coordset"] = "MMESH";

    // rename the topo
    topologies.rename_child(topo_name, "MMESH");

    auto field_itr = fields.children();
    while (field_itr.has_next())
    {
        Node &n_field = field_itr.next();
        std::string field_name = field_itr.name();

        // use new topo name
        n_field["topology"].reset();
        n_field["topology"] = "MMESH";

        // remove vol dep
        if (n_field.has_child("volume_dependent"))
        {
            n_field.remove_child("volume_dependent");
        }

        // we need to rename vector components
        if (n_field["values"].dtype().is_object())
        {
            if (n_field["values"].number_of_children() > 0)
            {
                int child_index = 0;
                auto val_itr = n_field["values"].children();
                while (val_itr.has_next())
                {
                    val_itr.next();
                    std::string comp_name = val_itr.name();

                    // rename vector components
                    n_field["values"].rename_child(comp_name, std::to_string(child_index));

                    child_index ++;
                }
            }
        }
    }

    if (!save_mesh.has_path("state/domain_id"))
    {
        save_mesh["state"]["domain_id"] = 0;
    }

    // TODO materials
}

//-----------------------------------------------------------------------------
void
add_matset_to_spiral(Node &n_mesh, const int ndomains)
{
    // Add a matset to each domain
    for (index_t domain_id = 0; domain_id < n_mesh.number_of_children(); domain_id ++)
    {
        Node &domain = n_mesh[domain_id];
        const auto num_elements = blueprint::mesh::topology::length(domain["topologies/topo"]);
        Node &matset = domain["matsets/matset"];
        // add a matset to it
        matset["topology"].set("topo");

        // Uni buffer requires material map
        for(index_t i = 0; i < ndomains; i ++)
        {
            const std::string mat_name("mat" + std::to_string(i));
            matset["material_map"][mat_name].set((int32) i);
        }

        Node &mat_ids = matset["material_ids"];
        mat_ids.set_dtype(DataType::index_t(num_elements));
        index_t_array ids = mat_ids.value();
        for (index_t i = 0; i < ids.number_of_elements(); i++)
        {
            ids[i] = domain_id;
        }

        Node &mat_vfs = matset["volume_fractions"];
        mat_vfs.set_dtype(DataType::c_float(num_elements));
        float_array data = mat_vfs.value();
        for (index_t i = 0; i < data.number_of_elements(); i++)
        {
            data[i] = 1.f;
        }
    }
}

#endif
