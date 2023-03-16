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

// Copy constructor
// Arrays constructor
// Assignment operator
//     Exercise info() on all constructor tests?
// shape, offset
// shape, offset, stride (and exercise info())


//-----------------------------------------------------------------------------
int calcFlatIndex(index_t x, index_t y, index_t z, const index_t dim,
    const index_t* shape, const index_t* offset, const index_t* stride)
{
    int retval = 0;
    retval += (offset[0] + x);
    retval += (offset[1] + y) * stride[0];
    if (dim > 2)
    {
        retval += (offset[2] + z) * stride[1] * stride[0];
    }
    return retval;
}

//-----------------------------------------------------------------------------
void verify3DCoords(meshutils::NDIndex idx,
    const index_t* shape, const index_t* offset, const index_t* stride)
{
    int dim = 3;
    EXPECT_EQ(idx.shape(-1), dim);
    EXPECT_EQ(idx.shape(0), shape[0]);
    EXPECT_EQ(idx.shape(1), shape[1]);
    EXPECT_EQ(idx.shape(2), shape[2]);

    EXPECT_EQ(idx.offset(0), offset[0]);
    EXPECT_EQ(idx.offset(1), offset[1]);
    EXPECT_EQ(idx.offset(2), offset[2]);

    EXPECT_EQ(idx.stride(0), stride[0]);
    EXPECT_EQ(idx.stride(1), stride[1]);
    EXPECT_EQ(idx.stride(2), stride[2]);

    EXPECT_EQ(idx.index(0, 0, 0), calcFlatIndex(0, 0, 0, dim, shape, offset, stride));
    EXPECT_EQ(idx.index(0, 1, 1), calcFlatIndex(0, 1, 1, dim, shape, offset, stride));
    EXPECT_EQ(idx.index(3, 0, 2), calcFlatIndex(3, 0, 2, dim, shape, offset, stride));
    EXPECT_EQ(idx.index(4, 3, 0), calcFlatIndex(4, 3, 0, dim, shape, offset, stride));
    EXPECT_EQ(idx.index(5, 2, 1), calcFlatIndex(5, 2, 1, dim, shape, offset, stride));
}

void verifySomeCtors(Node& parms, const index_t dim,
    const index_t* shape, const index_t* offset, const index_t* stride)
{
    {
        SCOPED_TRACE("Node reference");
        meshutils::NDIndex idx(parms);

        Node info;
        idx.info(info);

        verify3DCoords(idx, shape, offset, stride);

        {
            SCOPED_TRACE("Copy ctor of Node reference");
            meshutils::NDIndex idx2(idx);

            verify3DCoords(idx2, shape, offset, stride);
        }
    }

    {
        SCOPED_TRACE("Node pointer");
        meshutils::NDIndex idx(&parms);

        Node info;
        idx.info(info);

        verify3DCoords(idx, shape, offset, stride);
    }

    {
        SCOPED_TRACE("Pointers");
        meshutils::NDIndex idx(dim, shape);

        Node info;
        idx.info(info);

        verify3DCoords(idx, shape, offset, stride);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_index, ctor_shape)
{
    constexpr index_t dim = 3;
    constexpr index_t dx = 7;
    constexpr index_t dy = 4;
    constexpr index_t dz = 3;

    const index_t p_shape[dim]{ dx, dy, dz };
    const index_t p_offset[dim]{ 0, 0, 0 };
    const index_t p_stride[dim]{ dx, dy, dz };

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;

    verifySomeCtors(parms, dim, p_shape, p_offset, p_stride);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_index, ctor_shape_stride)
{
    constexpr index_t dim = 3;
    constexpr index_t dx = 7;
    constexpr index_t dy = 4;
    constexpr index_t dz = 3;
    constexpr index_t sx = 8;
    constexpr index_t sy = 5;
    constexpr index_t sz = 4;

    const index_t p_shape[dim]{ dx, dy, dz };
    const index_t p_offset[dim]{ 0, 0, 0 };
    const index_t p_stride[dim]{ sx, sy, sz };

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;
    index_t* stride = parms["stride"].value();
    stride[0] = sx;
    stride[1] = sy;
    stride[2] = sz;

    verifySomeCtors(parms, dim, p_shape, p_offset, p_stride);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_index, ctor_shape_offset)
{
    constexpr index_t dim = 3;
    constexpr index_t dx = 7;
    constexpr index_t dy = 4;
    constexpr index_t dz = 3;
    constexpr index_t ox = 1;
    constexpr index_t oy = 2;
    constexpr index_t oz = 1;

    const index_t p_shape[dim]{ dx, dy, dz };
    const index_t p_offset[dim]{ ox, oy, oz };
    const index_t p_stride[dim]{ dx + ox, dy + oy, dz + oz };

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;
    index_t* offset = parms["offset"].value();
    offset[0] = ox;
    offset[1] = oy;
    offset[2] = oz;

    verifySomeCtors(parms, dim, p_shape, p_offset, p_stride);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_index, ctor_shape_stride_offset)
{
    constexpr index_t dim = 3;
    constexpr index_t dx = 7;
    constexpr index_t dy = 4;
    constexpr index_t dz = 3;
    constexpr index_t ox = 2;
    constexpr index_t oy = 2;
    constexpr index_t oz = 2;
    constexpr index_t sx = 10;
    constexpr index_t sy = 7;
    constexpr index_t sz = 6;

    const index_t p_shape[dim]{ dx, dy, dz };
    const index_t p_offset[dim]{ ox, oy, oz };
    const index_t p_stride[dim]{ sx, sy, sz };

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;

    verifySomeCtors(parms, dim, p_shape, p_offset, p_stride);
}

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
