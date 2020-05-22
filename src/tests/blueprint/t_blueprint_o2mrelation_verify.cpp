//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

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
