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
int calcFlatIndex(index_t x, index_t y, index_t z, const index_t dim,
    const index_t* shape, const index_t* offset, const index_t* stride)
{
    int retval = 0;
    retval += (offset[0] + x) * stride[0];
    retval += (offset[1] + y) * stride[1];
    if (dim > 2)
    {
        retval += (offset[2] + z) * stride[2];
    }
    return retval;
}

//-----------------------------------------------------------------------------
void verify2DCoords(meshutils::NDIndex idx,
    const index_t* shape, const index_t* offset, const index_t* stride)
{
    const int DUMMY = -1;

    int dim = 2;
    EXPECT_EQ(idx.ndims(), dim);
    EXPECT_EQ(idx.shape(0), shape[0]);
    EXPECT_EQ(idx.shape(1), shape[1]);

    EXPECT_EQ(idx.offset(0), offset[0]);
    EXPECT_EQ(idx.offset(1), offset[1]);

    EXPECT_EQ(idx.stride(0), stride[0]);
    EXPECT_EQ(idx.stride(1), stride[1]);

    EXPECT_EQ(idx.index(0, 0), calcFlatIndex(0, 0, DUMMY, dim, shape, offset, stride));
    EXPECT_EQ(idx.index(0, 1), calcFlatIndex(0, 1, DUMMY, dim, shape, offset, stride));
    EXPECT_EQ(idx.index(3, 0), calcFlatIndex(3, 0, DUMMY, dim, shape, offset, stride));
    EXPECT_EQ(idx.index(4, 3), calcFlatIndex(4, 3, DUMMY, dim, shape, offset, stride));
    EXPECT_EQ(idx.index(5, 2), calcFlatIndex(5, 2, DUMMY, dim, shape, offset, stride));
}

//-----------------------------------------------------------------------------
void verify3DCoords(meshutils::NDIndex idx,
    const index_t* shape, const index_t* offset, const index_t* stride)
{
    int dim = 3;
    EXPECT_EQ(idx.ndims(), dim);
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

void verifyEquality(const meshutils::NDIndex& idx1, const meshutils::NDIndex& idx2)
{
    ASSERT_EQ(idx1.ndims(), idx2.ndims());

    index_t dim = idx1.ndims();

    for (index_t d = 0; d < dim; ++d)
    {
        EXPECT_EQ(idx1.shape(d), idx2.shape(d));
        EXPECT_EQ(idx1.offset(d), idx2.offset(d));
        EXPECT_EQ(idx1.stride(d), idx2.stride(d));
    }
}

void verifyNodeCtors(Node& parms, const index_t dim,
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
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_index, copy_ctor)
{
    constexpr index_t dim = 3;
    constexpr index_t dx = 7;
    constexpr index_t dy = 4;
    constexpr index_t dz = 3;

    const index_t p_shape[dim]{ dx, dy, dz };
    const index_t p_offset[dim]{ 0, 0, 0 };
    const index_t p_stride[dim]{ 1, dx, dx * dy };

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;

    meshutils::NDIndex idx(parms);
    meshutils::NDIndex idx2(idx);

    verifyEquality(idx, idx2);
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
    const index_t p_stride[dim]{ 1, dx, dx * dy };

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;

    verifyNodeCtors(parms, dim, p_shape, p_offset, p_stride);

    {
        SCOPED_TRACE("Pointer");

        meshutils::NDIndex idx(dim, p_shape);

        Node info;
        idx.info(info);

        verify3DCoords(idx, p_shape, p_offset, p_stride);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_index, ctor_shape_stride)
{
    constexpr index_t dim = 3;
    constexpr index_t dx = 7;
    constexpr index_t dy = 4;
    constexpr index_t dz = 3;
    constexpr index_t sx = 1;
    constexpr index_t sy = sx * (dx + 1);
    constexpr index_t sz = sy * (dy + 1);

    const index_t p_shape[dim]{ dx, dy, dz };
    const index_t p_offset[dim]{ 0, 0, 0 };
    const index_t p_stride[dim]{ sx, sy, sz };

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;
    parms["stride"].set(DataType::index_t(dim));
    index_t* stride = parms["stride"].value();
    stride[0] = sx;
    stride[1] = sy;
    stride[2] = sz;

    verifyNodeCtors(parms, dim, p_shape, p_offset, p_stride);

    {
        SCOPED_TRACE("Pointer");

        meshutils::NDIndex idx(dim, p_shape, NULL, p_stride);

        verify3DCoords(idx, p_shape, p_offset, p_stride);
    }
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
    const index_t p_stride[dim]{ 1, dx + ox, (dx + ox) * (dy + oy) };

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;
    parms["offset"].set(DataType::index_t(dim));
    index_t* offset = parms["offset"].value();
    offset[0] = ox;
    offset[1] = oy;
    offset[2] = oz;

    verifyNodeCtors(parms, dim, p_shape, p_offset, p_stride);

    {
        SCOPED_TRACE("Pointer");

        meshutils::NDIndex idx(dim, p_shape, p_offset, NULL);

        verify3DCoords(idx, p_shape, p_offset, p_stride);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_index, ctor_shape_offset_stride)
{
    constexpr index_t dim = 3;
    constexpr index_t dx = 7;
    constexpr index_t dy = 4;
    constexpr index_t dz = 3;
    constexpr index_t ox = 2;
    constexpr index_t oy = 2;
    constexpr index_t oz = 2;
    constexpr index_t sx = 1;
    constexpr index_t sy = dx + ox + 1;
    constexpr index_t sz = sy * (dy + oy + 1);

    const index_t p_shape[dim]{ dx, dy, dz };
    const index_t p_offset[dim]{ ox, oy, oz };
    const index_t p_stride[dim]{ sx, sy, sz };

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;
    parms["offset"].set(DataType::index_t(dim));
    index_t* offset = parms["offset"].value();
    offset[0] = ox;
    offset[1] = oy;
    offset[2] = oz;
    parms["stride"].set(DataType::index_t(dim));
    index_t* stride = parms["stride"].value();
    stride[0] = sx;
    stride[1] = sy;
    stride[2] = sz;

    verifyNodeCtors(parms, dim, p_shape, p_offset, p_stride);

    {
        SCOPED_TRACE("Pointer");

        meshutils::NDIndex idx(dim, p_shape, p_offset, p_stride);

        verify3DCoords(idx, p_shape, p_offset, p_stride);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_index, twoD)
{
    constexpr index_t dim = 2;
    constexpr index_t dx = 7;
    constexpr index_t dy = 4;
    constexpr index_t ox = 2;
    constexpr index_t oy = 2;

    const index_t p_shape[dim]{ dx, dy };
    const index_t p_offset[dim]{ ox, oy };
    const index_t p_stride[dim]{ 1, dx + ox };

    meshutils::NDIndex idx(dim, p_shape, p_offset);

    verify2DCoords(idx, p_shape, p_offset, p_stride);
}

TEST(conduit_blueprint_mesh_index, assignment)
{
    // First make a 2D index, with pointers.  Verify it.  Make a copy.
    constexpr index_t dim2 = 2;
    constexpr index_t dx2 = 8;
    constexpr index_t dy2 = 6;
    constexpr index_t ox2 = 2;
    constexpr index_t oy2 = 2;

    const index_t p_shape2[dim2]{ dx2, dy2 };
    const index_t p_offset2[dim2]{ ox2, oy2 };
    const index_t p_stride2[dim2]{ 1, dx2 + ox2 };

    meshutils::NDIndex idxA(dim2, p_shape2, p_offset2);
    verify2DCoords(idxA, p_shape2, p_offset2, p_stride2);

    meshutils::NDIndex idxB(idxA);
    verify2DCoords(idxB, p_shape2, p_offset2, p_stride2);

    // Then make a 3D index, with a node.  Verify that.
    constexpr index_t dim = 3;
    constexpr index_t dx = 7;
    constexpr index_t dy = 4;
    constexpr index_t dz = 3;
    constexpr index_t ox = 1;
    constexpr index_t oy = 0;
    constexpr index_t oz = 2;
    constexpr index_t sx = 1;
    constexpr index_t sy = sx * (dx + ox + 2);
    constexpr index_t sz = sy * (dy + oy + 3);

    const index_t p_shape[dim]{ dx, dy, dz };
    const index_t p_offset[dim]{ ox, oy, oz };
    const index_t p_stride[dim]{ sx, sy, sz };

    Node parms;
    parms["shape"].set(DataType::index_t(dim));
    index_t* shape = parms["shape"].value();
    shape[0] = dx;
    shape[1] = dy;
    shape[2] = dz;
    parms["offset"].set(DataType::index_t(dim));
    index_t* offset = parms["offset"].value();
    offset[0] = ox;
    offset[1] = oy;
    offset[2] = oz;
    parms["stride"].set(DataType::index_t(dim));
    index_t* stride = parms["stride"].value();
    stride[0] = sx;
    stride[1] = sy;
    stride[2] = sz;

    meshutils::NDIndex idxC(parms);
    verify3DCoords(idxC, p_shape, p_offset, p_stride);

    // Now assign the 3D index to the first 2D index.
    // Verify the result and check for equality.
    idxA = idxC;
    verify3DCoords(idxA, p_shape, p_offset, p_stride);
    verifyEquality(idxA, idxC);

    // Now assign the second 2D index to the 3D index.
    // Verify the result and check for equality.
    idxC = idxB;
    verify2DCoords(idxC, p_shape2, p_offset2, p_stride2);
    verifyEquality(idxC, idxB);
}

/// Test Driver ///

int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    result = RUN_ALL_TESTS();

    return result;
}
