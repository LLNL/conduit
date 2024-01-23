// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_matset_xforms.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_log.hpp"

#include <algorithm>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
// Simply adds an "element_ids" field [0 1 2 ... N-1]
static void
convert_to_material_based(const Node &topo, Node &mset)
{
    const int nelem = static_cast<int>(blueprint::mesh::topology::length(topo));
    mset["element_ids"].set_dtype(DataType::c_int(nelem));
    DataArray<int> eids = mset["element_ids"].value();
    for(int i = 0; i < nelem; i++)
    {
        eids[i] = i;
    }
}

/// Test Cases ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_matset_xforms, mesh_util_to_silo_basic)
{
    Node mesh;
    {
        blueprint::mesh::examples::basic("quads", 2, 2, 0, mesh);

        float64 mset_a_vfs[] = {1.0, 0.5, 0.5, 0.0};
        float64 mset_b_vfs[] = {0.0, 0.5, 0.5, 1.0};

        Node &mset = mesh["matsets/matset"];
        mset["topology"].set(mesh["topologies"].child_names().front());
        mset["volume_fractions/a"].set(&mset_a_vfs[0], 4);
        mset["volume_fractions/b"].set(&mset_b_vfs[0], 4);
    }
    Node &mset = mesh["matsets/matset"];

    Node silo, info;
    blueprint::mesh::matset::to_silo(mset, silo);
    std::cout << silo.to_yaml() << std::endl;

    { // Check General Contents //
        EXPECT_TRUE(silo.has_child("topology"));
        EXPECT_TRUE(silo.has_child("matlist"));
        EXPECT_TRUE(silo.has_child("mix_next"));
        EXPECT_TRUE(silo.has_child("mix_mat"));
        EXPECT_TRUE(silo.has_child("mix_vf"));
    }

    { // Check 'topology' Field //
        const std::string expected_topology = mset["topology"].as_string();
        const std::string actual_topology = silo["topology"].as_string();
        EXPECT_EQ(actual_topology, expected_topology);
    }

    // { // Check 'matlist' Field //
    //     // TODO(JRC): Need to make sure these are the same type.
    //     int64 expected_matlist_vec[] = {1, -1, -3, 2};
    //     Node expected_matlist(DataType::int64(4),
    //         &expected_matlist_vec[0], true);
    //     const Node &actual_matlist = silo["matlist"];

    //     EXPECT_FALSE(actual_matlist.diff(expected_matlist, info));
    // }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_matset_xforms, mesh_util_venn_to_silo)
{
    const int nx = 4, ny = 4;
    const double radius = 0.25;

    Node mset_silo_baseline;
    
    // all of these cases should create the same silo output
    // we diff the 2 and 3 cases with the 1 to test this

    CONDUIT_INFO("venn full to silo");
    {
        Node mesh;
        blueprint::mesh::examples::venn("full", nx, ny, radius, mesh);
        const Node &mset = mesh["matsets/matset"];

        std::cout << mset.to_yaml() << std::endl;

        Node mset_silo;
        blueprint::mesh::matset::to_silo(mset, mset_silo);
        std::cout << mset_silo.to_yaml() << std::endl;

        mset_silo_baseline.set(mset_silo);
    }

    CONDUIT_INFO("venn sparse_by_material to silo");
    {
        Node mesh, info;
        blueprint::mesh::examples::venn("sparse_by_material", nx, ny, radius, mesh);
        const Node &mset = mesh["matsets/matset"];

        std::cout << mset.to_yaml() << std::endl;

        Node mset_silo;
        blueprint::mesh::matset::to_silo(mset, mset_silo);
        std::cout << mset_silo.to_yaml() << std::endl;

        EXPECT_FALSE(mset_silo.diff(mset_silo_baseline,info));
    }

    CONDUIT_INFO("venn sparse_by_element to silo");
    {
        Node mesh, info;
        blueprint::mesh::examples::venn("sparse_by_element", nx, ny, radius, mesh);
        const Node &mset = mesh["matsets/matset"];

        std::cout << mset.to_yaml() << std::endl;

        Node mset_silo;
        blueprint::mesh::matset::to_silo(mset, mset_silo);
        std::cout << mset_silo.to_yaml() << std::endl;

        EXPECT_FALSE(mset_silo.diff(mset_silo_baseline,info));
    }

    CONDUIT_INFO("venn sparse_by_element (converted to material based) to silo")
    {
        Node mesh, info;
        blueprint::mesh::examples::venn("sparse_by_element", nx, ny, radius, mesh);
        Node &mset = mesh["matsets/matset"];
        convert_to_material_based(mesh["topologies/topo"], mset);

        std::cout << mset.to_yaml() << std::endl;

        Node mset_silo;
        blueprint::mesh::matset::to_silo(mset, mset_silo);
        std::cout << mset_silo.to_yaml() << std::endl;

        EXPECT_FALSE(mset_silo.diff(mset_silo_baseline,info));
        info.print();
    }

}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_matset_xforms, mesh_util_venn_to_silo_matset_values)
{
    const int nx = 4, ny = 4;
    const double radius = 0.25;

    Node mset_silo_baseline;
    
    // all of these cases should create the same silo output
    // we diff the 2 and 3 cases with the 1 to test this

    CONDUIT_INFO("venn full to silo");
    {
        Node mesh;
        blueprint::mesh::examples::venn("full", nx, ny, radius, mesh);
        const Node &field = mesh["fields/mat_check"];
        const Node &mset = mesh["matsets/matset"];

        std::cout << mset.to_yaml() << std::endl;
        std::cout << field.to_yaml() << std::endl;

        Node mset_silo;
        blueprint::mesh::field::to_silo(field,
                                         mset,
                                         mset_silo);

        std::cout << mset_silo.to_yaml() << std::endl;

        mset_silo_baseline.set(mset_silo);
    }

    CONDUIT_INFO("venn sparse_by_material to silo");
    {
        Node mesh, info;
        blueprint::mesh::examples::venn("sparse_by_material", nx, ny, radius, mesh);
        const Node &field = mesh["fields/mat_check"];
        const Node &mset = mesh["matsets/matset"];


        std::cout << mset.to_yaml() << std::endl;
        std::cout << field.to_yaml() << std::endl;

        Node mset_silo;
        blueprint::mesh::field::to_silo(field,
                                         mset,
                                         mset_silo);

        std::cout << mset_silo.to_yaml() << std::endl;

        EXPECT_FALSE(mset_silo.diff(mset_silo_baseline,info));
    }

    CONDUIT_INFO("venn sparse_by_element to silo");
    {
        Node mesh, info;
        blueprint::mesh::examples::venn("sparse_by_element", nx, ny, radius, mesh);
        const Node &field = mesh["fields/mat_check"];
        const Node &mset = mesh["matsets/matset"];


        std::cout << mset.to_yaml() << std::endl;
        std::cout << field.to_yaml() << std::endl;

        Node mset_silo;
        blueprint::mesh::field::to_silo(field,
                                        mset,
                                        mset_silo);

        std::cout << mset_silo.to_yaml() << std::endl;

        EXPECT_FALSE(mset_silo.diff(mset_silo_baseline,info));
    }

    CONDUIT_INFO("venn sparse_by_element (converted to material based) to silo")
    {
        Node mesh, info;
        blueprint::mesh::examples::venn("sparse_by_element", nx, ny, radius, mesh);
        const Node &field = mesh["fields/mat_check"];
        Node &mset = mesh["matsets/matset"];
        convert_to_material_based(mesh["topologies/topo"], mset);

        std::cout << mset.to_yaml() << std::endl;
        std::cout << field.to_yaml() << std::endl;

        Node mset_silo;
        blueprint::mesh::field::to_silo(field,
                                        mset,
                                        mset_silo);

        std::cout << mset_silo.to_yaml() << std::endl;

        EXPECT_FALSE(mset_silo.diff(mset_silo_baseline,info));
    }

}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_matset_xforms, mesh_util_matset_full_to_sparse_by_element)
{
    const int nx = 4, ny = 4;
    const double radius = 0.25;

    Node mesh_full, mesh_sbe, mesh_sbm, info;
    blueprint::mesh::examples::venn("full", nx, ny, radius, mesh_full);
    blueprint::mesh::examples::venn("sparse_by_element", nx, ny, radius, mesh_sbe);
    blueprint::mesh::examples::venn("sparse_by_material", nx, ny, radius, mesh_sbm);

    CONDUIT_INFO("venn full -> full");
    {
        // diff full -> full with full

        const Node &mset = mesh_full["matsets/matset"];
        const Node &field = mesh_full["fields/importance"];
        const Node &full_mset_baseline = mesh_full["matsets/matset"];
        const Node &full_field_baseline = mesh_full["fields/importance"];

        std::cout << mset.to_yaml() << std::endl;
        std::cout << field.to_yaml() << std::endl;

        Node converted_mset, converted_mset2, converted_field;
        blueprint::mesh::field::to_multi_buffer_full(mset, field, converted_mset, converted_field);
        blueprint::mesh::matset::to_multi_buffer_full(mset, converted_mset2);
        std::cout << converted_mset.to_yaml() << std::endl;
        std::cout << converted_field.to_yaml() << std::endl;

        EXPECT_FALSE(converted_mset.diff(full_mset_baseline, info));
        EXPECT_FALSE(converted_mset2.diff(full_mset_baseline, info));
        EXPECT_FALSE(converted_field.diff(full_field_baseline, info));
    }

    CONDUIT_INFO("venn full -> sparse_by_element");
    {
        // diff full -> sbe with sbe

        const Node &mset = mesh_full["matsets/matset"];
        const Node &field = mesh_full["fields/importance"];
        const Node &sbe_mset_baseline = mesh_sbe["matsets/matset"];
        const Node &sbe_field_baseline = mesh_sbe["fields/importance"];

        std::cout << mset.to_yaml() << std::endl;
        std::cout << field.to_yaml() << std::endl;

        Node converted_mset, converted_mset2, converted_field;
        blueprint::mesh::field::to_sparse_by_element(mset, field, converted_mset, converted_field);
        blueprint::mesh::matset::to_sparse_by_element(mset, converted_mset2);
        std::cout << converted_mset.to_yaml() << std::endl;
        std::cout << converted_field.to_yaml() << std::endl;

        EXPECT_FALSE(converted_mset.diff(sbe_mset_baseline, info));
        EXPECT_FALSE(converted_mset2.diff(sbe_mset_baseline, info));
        EXPECT_FALSE(converted_field.diff(sbe_field_baseline, info));
    }

    CONDUIT_INFO("venn full -> sparse_by_material");
    {
        // diff full -> sbm with sbm

        const Node &mset = mesh_full["matsets/matset"];
        const Node &field = mesh_full["fields/importance"];
        const Node &sbm_mset_baseline = mesh_sbm["matsets/matset"];
        const Node &sbm_field_baseline = mesh_sbm["fields/importance"];

        std::cout << mset.to_yaml() << std::endl;
        std::cout << field.to_yaml() << std::endl;

        Node converted_mset, converted_mset2, converted_field;
        blueprint::mesh::field::to_multi_buffer_by_material(mset, field, converted_mset, converted_field);
        blueprint::mesh::matset::to_multi_buffer_by_material(mset, converted_mset2);
        std::cout << converted_mset.to_yaml() << std::endl;
        std::cout << converted_field.to_yaml() << std::endl;

        EXPECT_FALSE(converted_mset.diff(sbm_mset_baseline, info));
        EXPECT_FALSE(converted_mset2.diff(sbm_mset_baseline, info));
        EXPECT_FALSE(converted_field.diff(sbm_field_baseline, info));
    }

    CONDUIT_INFO("venn sparse_by_element -> full");
    {
        // diff sbe -> full with full

        const Node &mset = mesh_sbe["matsets/matset"];
        const Node &field = mesh_sbe["fields/importance"];
        const Node &full_mset_baseline = mesh_full["matsets/matset"];
        const Node &full_field_baseline = mesh_full["fields/importance"];

        std::cout << mset.to_yaml() << std::endl;
        std::cout << field.to_yaml() << std::endl;

        Node converted_mset, converted_mset2, converted_field;
        blueprint::mesh::field::to_multi_buffer_full(mset, field, converted_mset, converted_field);
        blueprint::mesh::matset::to_multi_buffer_full(mset, converted_mset2);
        std::cout << converted_mset.to_yaml() << std::endl;
        std::cout << converted_field.to_yaml() << std::endl;

        EXPECT_FALSE(converted_mset.diff(full_mset_baseline, info));
        EXPECT_FALSE(converted_mset2.diff(full_mset_baseline, info));
        EXPECT_FALSE(converted_field.diff(full_field_baseline, info));
    }

    CONDUIT_INFO("venn sparse_by_element -> sparse_by_material");
    {
        // diff sbe -> sbm with sbm

        const Node &mset = mesh_sbe["matsets/matset"];
        const Node &field = mesh_sbe["fields/importance"];
        const Node &sbm_mset_baseline = mesh_sbm["matsets/matset"];
        const Node &sbm_field_baseline = mesh_sbm["fields/importance"];

        std::cout << mset.to_yaml() << std::endl;
        std::cout << field.to_yaml() << std::endl;

        Node converted_mset, converted_mset2, converted_field;
        blueprint::mesh::field::to_multi_buffer_by_material(mset, field, converted_mset, converted_field);
        blueprint::mesh::matset::to_multi_buffer_by_material(mset, converted_mset2);
        std::cout << converted_mset.to_yaml() << std::endl;
        std::cout << converted_field.to_yaml() << std::endl;

        EXPECT_FALSE(converted_mset.diff(sbm_mset_baseline, info));
        EXPECT_FALSE(converted_mset2.diff(sbm_mset_baseline, info));
        EXPECT_FALSE(converted_field.diff(sbm_field_baseline, info));
    }

    CONDUIT_INFO("venn sparse_by_material -> full");
    {
        // diff sbm -> full with full

        const Node &mset = mesh_sbm["matsets/matset"];
        const Node &field = mesh_sbm["fields/importance"];
        const Node &full_mset_baseline = mesh_full["matsets/matset"];
        const Node &full_field_baseline = mesh_full["fields/importance"];

        std::cout << mset.to_yaml() << std::endl;
        std::cout << field.to_yaml() << std::endl;

        Node converted_mset, converted_mset2, converted_field;
        blueprint::mesh::field::to_multi_buffer_full(mset, field, converted_mset, converted_field);
        blueprint::mesh::matset::to_multi_buffer_full(mset, converted_mset2);
        std::cout << converted_mset.to_yaml() << std::endl;
        std::cout << converted_field.to_yaml() << std::endl;

        EXPECT_FALSE(converted_mset.diff(full_mset_baseline, info));
        EXPECT_FALSE(converted_mset2.diff(full_mset_baseline, info));
        EXPECT_FALSE(converted_field.diff(full_field_baseline, info));
    }

    CONDUIT_INFO("venn sparse_by_material -> sparse_by_element");
    {
        // diff sbm -> sbe with sbe

        const Node &mset = mesh_sbm["matsets/matset"];
        const Node &field = mesh_sbm["fields/importance"];
        const Node &sbe_mset_baseline = mesh_sbe["matsets/matset"];
        const Node &sbe_field_baseline = mesh_sbe["fields/importance"];

        std::cout << mset.to_yaml() << std::endl;
        std::cout << field.to_yaml() << std::endl;

        Node converted_mset, converted_mset2, converted_field;
        blueprint::mesh::field::to_sparse_by_element(mset, field, converted_mset, converted_field);
        blueprint::mesh::matset::to_sparse_by_element(mset, converted_mset2);
        std::cout << converted_mset.to_yaml() << std::endl;
        std::cout << converted_field.to_yaml() << std::endl;

        EXPECT_FALSE(converted_mset.diff(sbe_mset_baseline, info));
        EXPECT_FALSE(converted_mset2.diff(sbe_mset_baseline, info));
        EXPECT_FALSE(converted_field.diff(sbe_field_baseline, info));
    }
}
