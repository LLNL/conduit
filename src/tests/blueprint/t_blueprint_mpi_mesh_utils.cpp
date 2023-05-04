// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mpi_mesh_utils.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi_mesh_utils.hpp"
#include "conduit_relay.hpp"
#include "conduit_relay_mpi_io_blueprint.hpp"
#include "conduit_log.hpp"

#include "blueprint_test_helpers.hpp"

#include <algorithm>
#include <vector>
#include <string>
#include <mpi.h>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;
using namespace generate;

//---------------------------------------------------------------------------
/**
 @brief Save the node to an HDF5 compatible with VisIt or the
        conduit_adjset_validate tool.
 */
void save_mesh(const conduit::Node &root, const std::string &filebase)
{
    // NOTE: Enable this to write files for debugging.
#if 0
    const std::string protocol("hdf5");
    conduit::relay::mpi::io::blueprint::save_mesh(root, filebase, protocol, MPI_COMM_WORLD);
#else
    std::cout << "Skip writing " << filebase << std::endl;
#endif
}

//---------------------------------------------------------------------------
bool validate(const conduit::Node &root,
              const std::string &adjsetName,
              conduit::Node &info)
{
    return conduit::blueprint::mpi::mesh::utils::adjset::validate(root,
               adjsetName, info, MPI_COMM_WORLD);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_utils, adjset_validate_element_0d)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    conduit::Node root, info;
    create_2_domain_0d_mesh(root, rank, size);
    save_mesh(root, "adjset_validate_element_0d");
    bool res = validate(root, "main_adjset", info);
    EXPECT_TRUE(res);

    // Now, adjust the adjsets so they are wrong on both domains.
    if(root.has_child("domain0"))
        root["domain0/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{0});
    if(root.has_child("domain1"))
        root["domain1/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{2});
    info.reset();
    save_mesh(root, "adjset_validate_element_0d_bad");
    res = validate(root, "main_adjset", info);
    EXPECT_FALSE(res);
    //info.print();

    if(rank == 0 || size == 1)
    {
        EXPECT_TRUE(info.has_path("domain0/main_adjset/domain0_1"));
        const conduit::Node &n0 = info["domain0/main_adjset/domain0_1"];
        EXPECT_EQ(n0.number_of_children(), 1);
        const conduit::Node &c0 = n0[0];
        EXPECT_TRUE(c0.has_path("element"));
        EXPECT_TRUE(c0.has_path("neighbor"));
        EXPECT_EQ(c0["element"].to_int(), 0);
        EXPECT_EQ(c0["neighbor"].to_int(), 1);
    }
    if(rank == 1 || size == 1)
    {
        EXPECT_TRUE(info.has_path("domain1/main_adjset/domain0_1"));
        const conduit::Node &n1 = info["domain1/main_adjset/domain0_1"];
        EXPECT_EQ(n1.number_of_children(), 1);
        const conduit::Node &c1 = n1[0];
        EXPECT_TRUE(c1.has_path("element"));
        EXPECT_TRUE(c1.has_path("neighbor"));
        EXPECT_EQ(c1["element"].to_int(), 2);
        EXPECT_EQ(c1["neighbor"].to_int(), 0);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_utils, adjset_validate_element_1d)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    conduit::Node root, info;
    create_2_domain_1d_mesh(root, rank, size);
    save_mesh(root, "adjset_validate_element_1d");
    bool res = validate(root, "main_adjset", info);
    EXPECT_TRUE(res);

    // Now, adjust the adjsets so they are wrong on both domains.
    if(root.has_child("domain0"))
        root["domain0/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{0});
    if(root.has_child("domain1"))
        root["domain1/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{1});
    info.reset();
    save_mesh(root, "adjset_validate_element_1d_bad");
    res = validate(root, "main_adjset", info);
    EXPECT_FALSE(res);
    //info.print();

    if(rank == 0 || size == 1)
    {
        EXPECT_TRUE(info.has_path("domain0/main_adjset/domain0_1"));
        const conduit::Node &n0 = info["domain0/main_adjset/domain0_1"];
        EXPECT_EQ(n0.number_of_children(), 1);
        const conduit::Node &c0 = n0[0];
        EXPECT_TRUE(c0.has_path("element"));
        EXPECT_TRUE(c0.has_path("neighbor"));
        EXPECT_EQ(c0["element"].to_int(), 0);
        EXPECT_EQ(c0["neighbor"].to_int(), 1);
    }
    if(rank == 1 || size == 1)
    {
        EXPECT_TRUE(info.has_path("domain1/main_adjset/domain0_1"));
        const conduit::Node &n1 = info["domain1/main_adjset/domain0_1"];
        EXPECT_EQ(n1.number_of_children(), 1);
        const conduit::Node &c1 = n1[0];
        EXPECT_TRUE(c1.has_path("element"));
        EXPECT_TRUE(c1.has_path("neighbor"));
        EXPECT_EQ(c1["element"].to_int(), 1);
        EXPECT_EQ(c1["neighbor"].to_int(), 0);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_utils, adjset_validate_element_2d)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    conduit::Node root, info;
    create_2_domain_2d_mesh(root, rank, size);
    save_mesh(root, "adjset_validate_element_2d");
    bool res = validate(root, "main_adjset", info);
    EXPECT_TRUE(res);
    info.print();

    // Now, adjust the adjset for domain1 so it includes an element not present in domain 0
    if(root.has_child("domain1"))
        root["domain1/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{0,2,4});
    info.reset();
    save_mesh(root, "adjset_validate_element_2d_bad");
    res = validate(root, "main_adjset", info);
    EXPECT_FALSE(res);
    //info.print();

    if(rank == 1 || size == 1)
    {
        EXPECT_TRUE(info.has_path("domain1/main_adjset/domain0_1"));
        const conduit::Node &n = info["domain1/main_adjset/domain0_1"];
        EXPECT_EQ(n.number_of_children(), 1);
        const conduit::Node &c = n[0];
        EXPECT_TRUE(c.has_path("element"));
        EXPECT_TRUE(c.has_path("neighbor"));
        EXPECT_EQ(c["element"].to_int(), 2);
        EXPECT_EQ(c["neighbor"].to_int(), 0);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_utils, adjset_validate_element_3d)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    conduit::Node root, info;
    create_2_domain_3d_mesh(root, rank, size);
    save_mesh(root, "adjset_validate_element_3d");
    bool res = validate(root, "main_adjset", info);
    EXPECT_TRUE(res);

    // Now, adjust the adjsets so they are wrong on both domains.
    if(root.has_child("domain0"))
        root["domain0/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{0});
    if(root.has_child("domain1"))
        root["domain1/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{2});
    info.reset();
    save_mesh(root, "adjset_validate_element_3d_bad");
    res = validate(root, "main_adjset", info);
    EXPECT_FALSE(res);
    //info.print();

    if(rank == 0 || size == 1)
    {
        EXPECT_TRUE(info.has_path("domain0/main_adjset/domain0_1"));
        const conduit::Node &n0 = info["domain0/main_adjset/domain0_1"];
        EXPECT_EQ(n0.number_of_children(), 1);
        const conduit::Node &c0 = n0[0];
        EXPECT_TRUE(c0.has_path("element"));
        EXPECT_TRUE(c0.has_path("neighbor"));
        EXPECT_EQ(c0["element"].to_int(), 0);
        EXPECT_EQ(c0["neighbor"].to_int(), 1);
    }
    if(rank == 1 || size == 1)
    {
        EXPECT_TRUE(info.has_path("domain1/main_adjset/domain0_1"));
        const conduit::Node &n1 = info["domain1/main_adjset/domain0_1"];
        EXPECT_EQ(n1.number_of_children(), 1);
        const conduit::Node &c1 = n1[0];
        EXPECT_TRUE(c1.has_path("element"));
        EXPECT_TRUE(c1.has_path("neighbor"));
        EXPECT_EQ(c1["element"].to_int(), 2);
        EXPECT_EQ(c1["neighbor"].to_int(), 0);
    }
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
