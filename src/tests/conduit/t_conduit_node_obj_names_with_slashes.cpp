//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-666778
//
// All rights reserved.
//
// This file is part of Conduit.
//
// For details, see: http://software.llnl.gov/conduit/.
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
/// file: conduit_node_obj_names_with_slashes.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <vector>
#include <string>
#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_node_paths, key_embedded_slashes)
{
    Node n;

    n["a/b/c/d/e/f"] = 10;
    n.add_child("a/b/c/d/e/f") = 42;
    std::cout << n.to_json() << std::endl;
    std::cout << n.to_yaml() << std::endl;


    Node &n_sub = n["a/b/c/d/e/f"];
    EXPECT_EQ(n_sub.to_int64(),10);

    Node &n_sub_2 = n.get_child("a/b/c/d/e/f");
    EXPECT_EQ(n_sub_2.to_int64(),42);


    Node info;
    // make sure no diff with self
    EXPECT_FALSE(n.diff(n,info));
    EXPECT_FALSE(n.diff_compatible(n,info));

    return;
    // check iters

    NodeIterator itr = n.children();

    int num_itr_children = 0;
    while(itr.has_next())
    {
        Node &curr = itr.next();
        std::string curr_name = itr.name();
        std::cout << "child: " << num_itr_children
                  << " name:" << curr_name << std::endl;
        if(num_itr_children == 0)
        {
            EXPECT_EQ("a",curr_name);
        }
        else if(num_itr_children == 1)
        {
            EXPECT_EQ("a/b/c/d/e/f",curr_name);
        }

        num_itr_children++;
    }

    EXPECT_EQ(n.number_of_children(),num_itr_children);

    n.save("tout_test_key_with_slashes.yaml");
    n.save("tout_test_key_with_slashes.json");
    n.save("tout_test_key_with_slashes.bin","conduit_bin");
    n.save("tout_test_key_with_slashes.conduit_json","conduit_json");
    n.save("tout_test_key_with_slashes.conduit_base64_json","conduit_base64_json");

    Node n_other;

    // standard set
    n_other.reset();
    n_other.set(n);
    std::cout << n_other.to_yaml() << std::endl;
    EXPECT_FALSE(n.diff(n_other,info));
    std::cout << info.to_yaml() << std::endl;

    // set_external
    n_other.reset();
    n_other.set_external(n);
    std::cout << info.to_yaml() << std::endl;
    EXPECT_FALSE(n.diff(n_other,info));
    std::cout << info.to_yaml() << std::endl;
    //
    // // all basic i/o protocols
    // n_other.reset();
    // n_other.load("tout_test_key_with_slashes.yaml","yaml");
    // std::cout << n_other.to_yaml() << std::endl;
    // EXPECT_FALSE(n.diff(n_other,info));
    // std::cout << info.to_yaml() << std::endl;
    //
    // n_other.reset();
    // n_other.load("tout_test_key_with_slashes.json","json");
    // std::cout << n_other.to_yaml() << std::endl;
    // EXPECT_FALSE(n.diff(n_other,info));
    // std::cout << info.to_yaml() << std::endl;
    //
    // n_other.reset();
    // n_other.load("tout_test_key_with_slashes.bin","conduit_bin");
    // std::cout << n_other.to_yaml() << std::endl;
    // EXPECT_FALSE(n.diff(n_other,info));
    // std::cout << info.to_yaml() << std::endl;
    //
    // n_other.reset();
    // n_other.load("tout_test_key_with_slashes.conduit_json","conduit_json");
    // std::cout << n_other.to_yaml() << std::endl;
    // EXPECT_FALSE(n.diff(n_other,info));
    // std::cout << info.to_yaml() << std::endl;
    //
    // n_other.reset();
    // n_other.load("tout_test_key_with_slashes.conduit_base64_json","conduit_base64_json");
    // std::cout << n_other.to_yaml() << std::endl;
    // EXPECT_FALSE(n.diff(n_other,info));
    // std::cout << info.to_yaml() << std::endl;

}


//-----------------------------------------------------------------------------
TEST(conduit_node_paths, key_embedded_slashes_b)
{
    Node n;

    n["normal_thing"] = 10;
    n.add_child("thing_with_/_inside") = 42;
    n["normal_thing"].add_child("thing_with_/_inside") = 43;
    n.remove_child("thing_with_/_inside");
    n.add_child("thing_with_/_inside") = 44;

    std::cout << n.to_yaml() << std::endl;
}

//-----------------------------------------------------------------------------
TEST(conduit_node_paths, key_embedded_slashes_a)
{
    Node n;

    n["normal_thing"] = 10;
    n["normal_thing_2"] = 10;
    // n.add_child("thing_with_/_inside") = 42;
    //std::cout << n.to_json() << std::endl;
    std::cout << n.to_yaml() << std::endl;

    // Node &n_sub = n["normal_thing"];
  //   EXPECT_EQ(n_sub.to_int64(),10);
  //
  //   Node &n_sub_2 = n.child("thing_with_/_inside");
  //   EXPECT_EQ(n_sub_2.to_int64(),42);


    Node n2,info;
    n2.set(n);
    // make sure no diff with self
    EXPECT_FALSE(n.diff(n2,info));

    std::cout << info.to_yaml() << std::endl;

    EXPECT_FALSE(n.diff_compatible(n,info));

    std::cout << info.to_yaml() << std::endl;
}




