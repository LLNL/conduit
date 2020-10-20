// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_o2mrelation_verify.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_o2mrelation_verify, o2mrelation_basic)
{
    Node n, info;

    n["a"].set(DataType::float64(20));
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

    n["b"].set(DataType::float64(20));
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

    n["indices"].set(DataType::int32(20));
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

    n["sizes"].set(DataType::int32(5));
    n["offsets"].set(DataType::int32(5));
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

    n.remove("indices");
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_o2mrelation_verify, o2mrelation_structure)
{
    Node n, info;

    n["a"].set(DataType::float64(20));
    n["sizes"].set(DataType::int32(5));
    EXPECT_FALSE(blueprint::o2mrelation::verify(n,info));

    n.remove("sizes");
    n["offsets"].set(DataType::int32(5));
    EXPECT_FALSE(blueprint::o2mrelation::verify(n,info));

    n["sizes"].set(DataType::int32(n["offsets"].dtype().number_of_elements() - 1));
    EXPECT_FALSE(blueprint::o2mrelation::verify(n,info));
    n["sizes"].set(DataType::int32(n["offsets"].dtype().number_of_elements() + 1));
    EXPECT_FALSE(blueprint::o2mrelation::verify(n,info));
    n["sizes"].set(DataType::int64(n["offsets"].dtype().number_of_elements()));
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

    n.remove("a");
    EXPECT_FALSE(blueprint::o2mrelation::verify(n,info));
    n["a"].set(DataType::char8_str(20));
    EXPECT_FALSE(blueprint::o2mrelation::verify(n,info));
    n["a"].set(DataType::float64(20));
    n["b"].set(DataType::char8_str(20));
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_o2mrelation_verify, o2mrelation_type)
{
    Node n, info;

    n.reset();
    EXPECT_FALSE(blueprint::o2mrelation::verify(n,info));

    n.reset();
    n.set(DataType::float64(20));
    EXPECT_FALSE(blueprint::o2mrelation::verify(n,info));

    n.reset();
    n.append().set(DataType::float64(20));
    EXPECT_FALSE(blueprint::o2mrelation::verify(n,info));

    n.reset();
    n["a"].set(DataType::char8_str(20));
    EXPECT_FALSE(blueprint::o2mrelation::verify(n,info));

    n.reset();
    n["a"].set(DataType::float32(20));
    n["sizes"].set(DataType::int32(20));
    n["offsets"].set(DataType::int32(20));
    n["indices"].set(DataType::int32(20));
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

    const std::string o2m_comps[] = {"sizes", "offsets", "indices"};
    const index_t o2m_comps_count = sizeof(o2m_comps) / sizeof(o2m_comps[0]);
    for(index_t comp_idx = 0; comp_idx < o2m_comps_count; comp_idx++)
    {
        const std::string &o2m_comp = o2m_comps[comp_idx];
        Node temp = n[o2m_comp];
        n[o2m_comp].set(DataType::float32(20));
        EXPECT_FALSE(blueprint::o2mrelation::verify(n,info));
        n[o2m_comp] = temp;
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_o2mrelation_verify, o2mrelation_verify_with_protocol)
{
    Node n, info;

    EXPECT_FALSE(blueprint::o2mrelation::verify("protocol",n,info));
    EXPECT_FALSE(blueprint::o2mrelation::verify("o2mrelation",n,info));
}
