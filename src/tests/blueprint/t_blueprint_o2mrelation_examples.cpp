// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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
TEST(conduit_blueprint_o2mrelation_examples, o2mrelation_data_paths)
{
    Node n;
    std::vector<std::string> paths;

    Node baseline, info;
    baseline["data"].set(conduit::DataType::float32(20));
    baseline["sizes"].set(conduit::DataType::int32(4));
    baseline["offsets"].set(conduit::DataType::int32(4));
    baseline["indices"].set(conduit::DataType::int32(16));
    EXPECT_TRUE(blueprint::o2mrelation::verify(baseline,info));

    n.set(baseline);
    paths = blueprint::o2mrelation::data_paths(n);
    EXPECT_EQ(paths, std::vector<std::string>({"data"}));

    n.set(baseline);
    n["more_data"].set(n["data"]);
    n["not_data_str"].set("string");
    n["not_data_obj"]["nv1"].set(n["data"]);
    n["not_data_obj"]["nv2"].set(n["data"]);
    paths = blueprint::o2mrelation::data_paths(n);
    EXPECT_EQ(paths, std::vector<std::string>({"data", "more_data"}));
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_o2mrelation_examples, o2mrelation_compact_to)
{
    Node n, ref, info;

    { // No Compaction Tests //
        blueprint::o2mrelation::examples::uniform(ref, 5, 3, 3);
        blueprint::o2mrelation::compact_to(ref, n);
        EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));
        EXPECT_FALSE(ref.diff(n, info));

        blueprint::o2mrelation::examples::uniform(ref, 5, 0, 0, "default");
        blueprint::o2mrelation::compact_to(ref, n);
        EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));
        EXPECT_FALSE(ref.diff(n, info));
    }

    { // Sizes/Offsets Compaction Tests //
        blueprint::o2mrelation::examples::uniform(ref, 5, 3, 5);
        blueprint::o2mrelation::compact_to(ref, n);
        EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));
        EXPECT_TRUE(ref.diff(n, info));

        blueprint::o2mrelation::examples::uniform(ref, 5, 3, 3);
        EXPECT_FALSE(ref.diff(n, info));
    }

    { // Data Compaction Tests //
        blueprint::o2mrelation::examples::uniform(ref, 3, 4, 5, "reversed");
        blueprint::o2mrelation::compact_to(ref, n);
        EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));
        EXPECT_TRUE(ref.diff(n, info));

        { // Check Path Schema //
            std::set<std::string> n_childset(n.child_names().begin(), n.child_names().end());
            std::set<std::string> ref_childset(ref.child_names().begin(), ref.child_names().end());
            EXPECT_NE(ref_childset, n_childset);

            n_childset.insert("indices");
            EXPECT_EQ(ref_childset, n_childset);

            conduit::NodeConstIterator niter = n.children();
            while(niter.has_next())
            {
                const std::string child_name = niter.next().name();
                EXPECT_EQ(ref[child_name].dtype().id(), n[child_name].dtype().id());
            }
        }

        { // Check Data //
            blueprint::o2mrelation::examples::uniform(ref, 3, 4);
            EXPECT_FALSE(ref["sizes"].diff(n["sizes"], info));
            EXPECT_FALSE(ref["offsets"].diff(n["offsets"], info));
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_o2mrelation_examples, o2mrelation_generate_offsets)
{
    Node n, ref, info;

    EXPECT_FALSE(blueprint::o2mrelation::generate_offsets(n, info));

    n["test"].set("value");
    EXPECT_FALSE(blueprint::o2mrelation::generate_offsets(n, info));

    blueprint::o2mrelation::examples::uniform(ref, 5, 3, 3);
    n.set(ref);
    n.remove("offsets");
    EXPECT_TRUE(blueprint::o2mrelation::generate_offsets(n, info));
    EXPECT_TRUE(blueprint::o2mrelation::verify(n,info));
    EXPECT_FALSE(ref.diff(n, info));

    blueprint::o2mrelation::examples::uniform(ref, 5, 3, 4);
    n.set(ref);
    n.remove("offsets");
    EXPECT_TRUE(blueprint::o2mrelation::generate_offsets(n, info));
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
        niter.next(blueprint::o2mrelation::DATA);
        EXPECT_EQ(niter.index(blueprint::o2mrelation::ONE), 0);
        EXPECT_EQ(niter.index(blueprint::o2mrelation::MANY), 0);
        EXPECT_EQ(niter.index(blueprint::o2mrelation::DATA), 0);

        niter.next(blueprint::o2mrelation::MANY);
        EXPECT_EQ(niter.index(blueprint::o2mrelation::ONE), 0);
        EXPECT_EQ(niter.index(blueprint::o2mrelation::MANY), 1);
        EXPECT_EQ(niter.index(blueprint::o2mrelation::DATA), 1);

        niter.next(blueprint::o2mrelation::ONE);
        EXPECT_EQ(niter.index(blueprint::o2mrelation::ONE), 1);
        EXPECT_EQ(niter.index(blueprint::o2mrelation::MANY), 1);
        EXPECT_EQ(niter.index(blueprint::o2mrelation::DATA), 5);
    }

    { // Elements Tests //
        blueprint::o2mrelation::O2MIterator niter(n);
        niter.next(blueprint::o2mrelation::DATA);
        EXPECT_EQ(niter.elements(blueprint::o2mrelation::ONE), 3);
        EXPECT_EQ(niter.elements(blueprint::o2mrelation::MANY), 2);
        EXPECT_EQ(niter.elements(blueprint::o2mrelation::DATA), 6);

        niter.next(blueprint::o2mrelation::ONE);
        EXPECT_EQ(niter.elements(blueprint::o2mrelation::ONE), 3);
        EXPECT_EQ(niter.elements(blueprint::o2mrelation::MANY), 2);
        EXPECT_EQ(niter.elements(blueprint::o2mrelation::DATA), 6);
    }

    { // Next/Previous Tests //
        blueprint::o2mrelation::O2MIterator niter(n);

        EXPECT_EQ(niter.next(blueprint::o2mrelation::ONE), 0);
        EXPECT_EQ(niter.peek_next(blueprint::o2mrelation::ONE), 1);
        EXPECT_EQ(niter.peek_next(blueprint::o2mrelation::MANY), 1);

        EXPECT_EQ(niter.next(blueprint::o2mrelation::ONE), 1);
        EXPECT_EQ(niter.peek_next(blueprint::o2mrelation::ONE), 2);
        EXPECT_EQ(niter.peek_next(blueprint::o2mrelation::MANY), 1);
        EXPECT_EQ(niter.next(blueprint::o2mrelation::MANY), 1);
        EXPECT_EQ(niter.peek_next(blueprint::o2mrelation::ONE), 2);
        EXPECT_EQ(niter.peek_next(blueprint::o2mrelation::MANY), 2);
        EXPECT_EQ(niter.peek_previous(blueprint::o2mrelation::ONE), 0);
        EXPECT_EQ(niter.peek_previous(blueprint::o2mrelation::MANY), 0);

        EXPECT_EQ(niter.previous(blueprint::o2mrelation::ONE), 0);
        EXPECT_EQ(niter.peek_next(blueprint::o2mrelation::ONE), 1);
        EXPECT_EQ(niter.peek_next(blueprint::o2mrelation::MANY), 2);
        EXPECT_EQ(niter.previous(blueprint::o2mrelation::MANY), 0);
        EXPECT_EQ(niter.peek_next(blueprint::o2mrelation::ONE), 1);
        EXPECT_EQ(niter.peek_next(blueprint::o2mrelation::MANY), 1);
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
