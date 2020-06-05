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
/// file: t_blueprint_o2mrelation_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <algorithm>
#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

/// Testing Helpers ///

std::vector<float> get_o2m_raw(Node &o2m, bool forward)
{
    float* o2m_ptr = (float*)o2m["data"].data_ptr();
    std::vector<float> o2m_data(o2m_ptr, o2m_ptr + o2m["data"].dtype().number_of_elements());

    if(!forward)
    {
        std::reverse(o2m_data.begin(), o2m_data.end());
    }

    return o2m_data;
}

std::vector<float> get_o2m_iter(Node &o2m, bool forward)
{
    blueprint::o2mrelation::O2MIterator iter(o2m);
    if(forward)
    {
        iter.to_front();
    }
    else
    {
        iter.to_back();
    }

    float32_array o2m_array = o2m["data"].value();
    std::vector<float> o2m_data;
    while(forward ? iter.has_next() : iter.has_previous())
    {
        o2m_data.push_back(o2m_array[forward ? iter.next() : iter.previous()]);
    }

    return o2m_data;
}

/// Test Cases ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_o2mrelation_examples, o2mrelation_verify)
{
    Node n, info;

    n.reset();
    blueprint::o2mrelation::examples::uniform(n, 10);
    std::cout << n.to_yaml() << std::endl;
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

    n.reset();
    blueprint::o2mrelation::examples::uniform(n, 5, 2);
    std::cout << n.to_yaml() << std::endl;
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

    n.reset();
    blueprint::o2mrelation::examples::uniform(n, 5, 2, 4);
    std::cout << n.to_yaml() << std::endl;
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

    n.reset();
    blueprint::o2mrelation::examples::uniform(n, 5, 0, 0, "reversed");
    std::cout << n.to_yaml() << std::endl;
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

    n.reset();
    blueprint::o2mrelation::examples::uniform(n, 5, 3, 4, "default");
    std::cout << n.to_yaml() << std::endl;
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_o2mrelation_examples, o2mrelation_query_paths)
{
    Node n, res;

    Node baseline, info;
    baseline["data"].set(conduit::DataType::float32(20));
    baseline["sizes"].set(conduit::DataType::int32(4));
    baseline["offsets"].set(conduit::DataType::int32(4));
    baseline["indices"].set(conduit::DataType::int32(16));
    EXPECT_TRUE(blueprint::o2mrelation::verify(baseline,info));

    n.set(baseline);
    blueprint::o2mrelation::query_paths(n, res);
    EXPECT_TRUE(res.dtype().is_object());
    EXPECT_EQ(res.child_names(), std::vector<std::string>({"data"}));

    n.set(baseline);
    n["more_data"].set(n["data"]);
    n["not_data_str"].set("string");
    n["not_data_obj"]["nv1"].set(n["data"]);
    n["not_data_obj"]["nv2"].set(n["data"]);
    blueprint::o2mrelation::query_paths(n, res);
    EXPECT_TRUE(res.dtype().is_object());
    EXPECT_EQ(res.child_names(), std::vector<std::string>({"data", "more_data"}));
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_o2mrelation_examples, o2mrelation_to_compact)
{
    Node n, ref, info;

    blueprint::o2mrelation::examples::uniform(ref, 5, 3, 3);
    n.set(ref);
    blueprint::o2mrelation::to_compact(n);
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));
    EXPECT_FALSE(ref.diff(n, info));

    blueprint::o2mrelation::examples::uniform(ref, 5, 3, 4);
    n.set(ref);
    blueprint::o2mrelation::to_compact(n);
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));
    EXPECT_TRUE(ref.diff(n, info));

    blueprint::o2mrelation::examples::uniform(ref, 5, 3, 3);
    EXPECT_FALSE(ref["sizes"].diff(n["sizes"], info));
    EXPECT_FALSE(ref["offsets"].diff(n["offsets"], info));
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_o2mrelation_examples, o2mrelation_iterator_properties)
{
    Node n, info;

    // o2m:
    //   data: [1.0, 2.0, -1.0, -1.0, 3.0, 4.0, -1.0, -1.0, 5.0, 6.0, -1.0, -1.0]
    //   sizes: [2, 2, 2]
    //   offsets: [0, 4, 8]
    blueprint::o2mrelation::examples::uniform(n, 3, 2, 4);
    std::cout << n.to_yaml() << std::endl;
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

    { // Index Tests //
        blueprint::o2mrelation::O2MIterator niter(n);
        niter.next(blueprint::O2MIndexType::DATA);
        EXPECT_EQ(niter.index(blueprint::O2MIndexType::ONE), 0);
        EXPECT_EQ(niter.index(blueprint::O2MIndexType::MANY), 0);
        EXPECT_EQ(niter.index(blueprint::O2MIndexType::DATA), 0);

        niter.next(blueprint::O2MIndexType::MANY);
        EXPECT_EQ(niter.index(blueprint::O2MIndexType::ONE), 0);
        EXPECT_EQ(niter.index(blueprint::O2MIndexType::MANY), 1);
        EXPECT_EQ(niter.index(blueprint::O2MIndexType::DATA), 1);

        niter.next(blueprint::O2MIndexType::ONE);
        EXPECT_EQ(niter.index(blueprint::O2MIndexType::ONE), 1);
        EXPECT_EQ(niter.index(blueprint::O2MIndexType::MANY), 1);
        EXPECT_EQ(niter.index(blueprint::O2MIndexType::DATA), 5);
    }

    { // Elements Tests //
        blueprint::o2mrelation::O2MIterator niter(n);
        niter.next(blueprint::O2MIndexType::DATA);
        EXPECT_EQ(niter.elements(blueprint::O2MIndexType::ONE), 3);
        EXPECT_EQ(niter.elements(blueprint::O2MIndexType::MANY), 2);
        EXPECT_EQ(niter.elements(blueprint::O2MIndexType::DATA), 6);

        niter.next(blueprint::O2MIndexType::ONE);
        EXPECT_EQ(niter.elements(blueprint::O2MIndexType::ONE), 3);
        EXPECT_EQ(niter.elements(blueprint::O2MIndexType::MANY), 2);
        EXPECT_EQ(niter.elements(blueprint::O2MIndexType::DATA), 6);
    }

    { // Next/Previous Tests //
        blueprint::o2mrelation::O2MIterator niter(n);

        EXPECT_EQ(niter.next(blueprint::O2MIndexType::ONE), 0);
        EXPECT_EQ(niter.peek_next(blueprint::O2MIndexType::ONE), 1);
        EXPECT_EQ(niter.peek_next(blueprint::O2MIndexType::MANY), 1);

        EXPECT_EQ(niter.next(blueprint::O2MIndexType::ONE), 1);
        EXPECT_EQ(niter.peek_next(blueprint::O2MIndexType::ONE), 2);
        EXPECT_EQ(niter.peek_next(blueprint::O2MIndexType::MANY), 1);
        EXPECT_EQ(niter.next(blueprint::O2MIndexType::MANY), 1);
        EXPECT_EQ(niter.peek_next(blueprint::O2MIndexType::ONE), 2);
        EXPECT_EQ(niter.peek_next(blueprint::O2MIndexType::MANY), 2);
        EXPECT_EQ(niter.peek_previous(blueprint::O2MIndexType::ONE), 0);
        EXPECT_EQ(niter.peek_previous(blueprint::O2MIndexType::MANY), 0);

        EXPECT_EQ(niter.previous(blueprint::O2MIndexType::ONE), 0);
        EXPECT_EQ(niter.peek_next(blueprint::O2MIndexType::ONE), 1);
        EXPECT_EQ(niter.peek_next(blueprint::O2MIndexType::MANY), 2);
        EXPECT_EQ(niter.previous(blueprint::O2MIndexType::MANY), 0);
        EXPECT_EQ(niter.peek_next(blueprint::O2MIndexType::ONE), 1);
        EXPECT_EQ(niter.peek_next(blueprint::O2MIndexType::MANY), 1);
    }

    { // Next/Previous Edge Case Tests //
        blueprint::o2mrelation::O2MIterator niter(n);
        EXPECT_TRUE(niter.has_next());
        EXPECT_FALSE(niter.has_previous());

        niter.to_back();
        EXPECT_FALSE(niter.has_next());
        EXPECT_TRUE(niter.has_previous());

        niter.to_front();
        EXPECT_TRUE(niter.has_next());
        EXPECT_FALSE(niter.has_previous());
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_o2mrelation_examples, o2mrelation_iterator_iteration)
{
    Node n, ref, info;

    { // Forward/Offsets Tests //
        blueprint::o2mrelation::examples::uniform(n, 3, 2, 4);
        blueprint::o2mrelation::examples::uniform(ref, 3, 2);
        std::cout << n.to_yaml() << std::endl;
        EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

        std::vector<float> ref_data = get_o2m_raw(ref, true);
        std::vector<float> n_data = get_o2m_iter(n, true);
        EXPECT_EQ(ref_data, n_data);
    }

    { // Backward/Indices Tests //
        blueprint::o2mrelation::examples::uniform(n, 2, 3, 4, "default");
        blueprint::o2mrelation::examples::uniform(ref, 2, 3);
        std::cout << n.to_yaml() << std::endl;
        EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));

        std::vector<float> ref_data = get_o2m_raw(ref, false);
        std::vector<float> n_data = get_o2m_iter(n, false);
        EXPECT_EQ(ref_data, n_data);
    }
}
