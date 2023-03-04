// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_index.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mesh_utils.hpp"

// #include <algorithm>
// #include <memory>
// #include <vector>
// #include <string>
#include "gtest/gtest.h"

using namespace conduit;
namespace meshutils = conduit::blueprint::mesh::utils;

/// Index Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_index, shape)
{
    constexpr int dim = 3;
    constexpr int dx = 7;
    constexpr int dy = 4;
    constexpr int dz = 3;

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;

    meshutils::NDIndex idx(parms);

    EXPECT_EQ(idx.shape(-1), dim);
    EXPECT_EQ(idx.shape(0), dx);
    EXPECT_EQ(idx.shape(1), dy);
    EXPECT_EQ(idx.shape(2), dz);

    EXPECT_EQ(idx.offset(0), 0);
    EXPECT_EQ(idx.offset(1), 0);
    EXPECT_EQ(idx.offset(2), 0);

    EXPECT_EQ(idx.stride(0), 1);
    EXPECT_EQ(idx.stride(1), dx);
    EXPECT_EQ(idx.stride(2), dx * dy);

    EXPECT_EQ(idx.index(0, 0, 0), 0);
    EXPECT_EQ(idx.index(0, 1, 1), 35);
    EXPECT_EQ(idx.index(3, 0, 2), 59);
    EXPECT_EQ(idx.index(4, 3, 0), 25);
    EXPECT_EQ(idx.index(5, 2, 1), 47);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_index, shape_stride)
{
    constexpr int dim = 3;
    constexpr int dx = 7;
    constexpr int dy = 4;
    constexpr int dz = 3;

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;

    meshutils::NDIndex idx(parms);

    EXPECT_EQ(idx.shape(-1), dim);
    EXPECT_EQ(idx.shape(0), dx);
    EXPECT_EQ(idx.shape(1), dy);
    EXPECT_EQ(idx.shape(2), dz);

    EXPECT_EQ(idx.offset(0), 0);
    EXPECT_EQ(idx.offset(1), 0);
    EXPECT_EQ(idx.offset(2), 0);

    EXPECT_EQ(idx.stride(0), 1);
    EXPECT_EQ(idx.stride(1), dx);
    EXPECT_EQ(idx.stride(2), dx * dy);

    EXPECT_EQ(idx.index(0, 0, 0), 0);
    EXPECT_EQ(idx.index(0, 1, 1), 35);
    EXPECT_EQ(idx.index(3, 0, 2), 59);
    EXPECT_EQ(idx.index(4, 3, 0), 25);
    EXPECT_EQ(idx.index(5, 2, 1), 47);
}

/// Test Driver ///

int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    result = RUN_ALL_TESTS();

    return result;
}
