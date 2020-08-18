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
TEST(conduit_node_obj_names_with_slashes, key_embedded_slashes_double_add_rm_child)
{
    Node n;

    n["normal/path"] = (int64)10;
    n.add_child("child_with_/_inside") = (int64)42;

    EXPECT_TRUE(n.has_path("normal/path"));
    EXPECT_FALSE(n.has_child("normal/path"));
    EXPECT_FALSE(n.has_path("child_with_/_inside"));
    EXPECT_TRUE(n.has_child("child_with_/_inside"));

    EXPECT_EQ(2,n.number_of_children());
    EXPECT_EQ(n["normal/path"].to_int64(),(int64)10);
    EXPECT_EQ(n.child("child_with_/_inside").to_int64(),(int64)42);

    std::cout << n.to_yaml() << std::endl;

    // check double add of child
    n.add_child("child_with_/_inside") = (int64)42;

    EXPECT_EQ(2,n.number_of_children());
    EXPECT_EQ(n["normal/path"].to_int64(),(int64)10);
    EXPECT_EQ(n.child("child_with_/_inside").to_int64(),(int64)42);

    std::cout << n.to_yaml() << std::endl;

    Node info;
    // make sure no diff with self
    EXPECT_FALSE(n.diff(n,info));
    EXPECT_FALSE(n.diff_compatible(n,info));

    n.remove_child("child_with_/_inside");
    EXPECT_EQ(n.number_of_children(),1);
    EXPECT_TRUE(n.has_path("normal/path"));
    EXPECT_FALSE(n.has_child("child_with_/_inside"));
    
}

//-----------------------------------------------------------------------------
TEST(conduit_node_obj_names_with_slashes, iters)
{
    Node n;

    n["normal/path"] = (int64)10;
    n.add_child("child_with_/_inside") = (int64)42;
    
    // std iter
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
            EXPECT_EQ("normal",curr_name);
        }
        else if(num_itr_children == 1)
        {
            EXPECT_EQ("child_with_/_inside",curr_name);
        }

        num_itr_children++;
    }

    EXPECT_EQ(n.number_of_children(),num_itr_children);
    
    // consg itr
    NodeConstIterator citr = n.children();

    int num_citr_children = 0;
    while(citr.has_next())
    {
        const Node &curr = citr.next();
        std::string curr_name = citr.name();
        std::cout << "child: " << num_itr_children
                  << " name:" << curr_name << std::endl;
        if(num_itr_children == 0)
        {
            EXPECT_EQ("normal",curr_name);
        }
        else if(num_itr_children == 1)
        {
            EXPECT_EQ("child_with_/_inside",curr_name);
        }

        num_citr_children++;
    }

    EXPECT_EQ(n.number_of_children(),num_citr_children);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_obj_names_with_slashes, set_etc)
{
    Node n;

    n["normal/path"] = (int64)10;
    n.add_child("child_with_/_inside") = (int64)42;

    std::cout << n.to_yaml() << std::endl;

    Node info;
    
    // check diff and diff compat
    // make sure no diff with self
    EXPECT_FALSE(n.diff(n,info));
    EXPECT_FALSE(n.diff_compatible(n,info));

    Node n_other;
    
    // set 
    n_other.set(n);
    EXPECT_EQ(n_other.number_of_children(),2);
    EXPECT_EQ(n_other["normal/path"].as_int64(),(int64)10);
    EXPECT_EQ(n_other.child("child_with_/_inside").as_int64(),(int64)42);
    EXPECT_FALSE(n_other.diff(n,info));

    // set_external
    n_other.reset();
    n_other.set_external(n);
    EXPECT_EQ(n_other.number_of_children(),2);
    EXPECT_EQ(n_other["normal/path"].as_int64(),(int64)10);
    EXPECT_EQ(n_other.child("child_with_/_inside").as_int64(),(int64)42);
    EXPECT_FALSE(n_other.diff(n,info));

    // compact_to
    n_other.reset();
    n.compact_to(n_other);
    EXPECT_EQ(n_other.number_of_children(),2);
    EXPECT_EQ(n_other["normal/path"].as_int64(),(int64)10);
    EXPECT_EQ(n_other.child("child_with_/_inside").as_int64(),(int64)42);
    EXPECT_FALSE(n_other.diff(n,info));

    // update
    n_other.reset();
    n_other.update(n);
    EXPECT_EQ(n_other.number_of_children(),2);
    EXPECT_EQ(n_other["normal/path"].as_int64(),(int64)10);
    EXPECT_EQ(n_other.child("child_with_/_inside").as_int64(),(int64)42);
    EXPECT_FALSE(n_other.diff(n,info));

    // update compat
    n_other.set(n); // we need structure to test compat
    n_other.update_compatible(n);
    EXPECT_EQ(n_other.number_of_children(),2);
    EXPECT_EQ(n_other["normal/path"].as_int64(),(int64)10);
    EXPECT_EQ(n_other.child("child_with_/_inside").as_int64(),(int64)42);
    EXPECT_FALSE(n_other.diff(n,info));

    // update external
    n_other.reset();
    n_other.update_external(n);
    EXPECT_EQ(n_other.number_of_children(),2);
    EXPECT_EQ(n_other["normal/path"].as_int64(),(int64)10);
    EXPECT_EQ(n_other.child("child_with_/_inside").as_int64(),(int64)42);
    EXPECT_FALSE(n_other.diff(n,info));

}



//-----------------------------------------------------------------------------
TEST(conduit_node_obj_names_with_slashes, basic_io)
{
    Node n;

    n["normal/path"] = (int64)10;
    n.add_child("child_with_/_inside") = (int64)42;
    std::cout << n.to_yaml() << std::endl;

    EXPECT_EQ(n.number_of_children(),2);
    EXPECT_EQ(n["normal/path"].as_int64(),(int64)10);
    EXPECT_EQ(n.child("child_with_/_inside").as_int64(),(int64)42);


    n.save("tout_test_key_with_slashes.yaml");
    n.save("tout_test_key_with_slashes.json");
    n.save("tout_test_key_with_slashes.bin","conduit_bin");
    n.save("tout_test_key_with_slashes.conduit_json","conduit_json");
    n.save("tout_test_key_with_slashes.conduit_base64_json","conduit_base64_json");

    Node info;
    // make sure no diff with self
    EXPECT_FALSE(n.diff(n,info));
    EXPECT_FALSE(n.diff_compatible(n,info));

    Node n_other;
    // all basic i/o protocols
    n_other.reset();
    n_other.load("tout_test_key_with_slashes.yaml","yaml");
    std::cout << n_other.to_yaml() << std::endl;
    EXPECT_FALSE(n.diff(n_other,info));
    std::cout << info.to_yaml() << std::endl;

    n_other.reset();
    n_other.load("tout_test_key_with_slashes.json","json");
    std::cout << n_other.to_yaml() << std::endl;
    EXPECT_FALSE(n.diff(n_other,info));
    std::cout << info.to_yaml() << std::endl;

    n_other.reset();
    n_other.load("tout_test_key_with_slashes.bin","conduit_bin");
    std::cout << n_other.to_yaml() << std::endl;
    EXPECT_FALSE(n.diff(n_other,info));
    std::cout << info.to_yaml() << std::endl;

    n_other.reset();
    n_other.load("tout_test_key_with_slashes.conduit_json","conduit_json");
    std::cout << n_other.to_yaml() << std::endl;
    EXPECT_FALSE(n.diff(n_other,info));
    std::cout << info.to_yaml() << std::endl;

    n_other.reset();
    n_other.load("tout_test_key_with_slashes.conduit_base64_json","conduit_base64_json");
    std::cout << n_other.to_yaml() << std::endl;
    EXPECT_FALSE(n.diff(n_other,info));
    std::cout << info.to_yaml() << std::endl;

}

//-----------------------------------------------------------------------------
TEST(conduit_node_obj_names_with_slashes, paths)
{
    Node n;
    n["normal/path"].add_child("child_with_/_inside") = (int64)42;

    std::string test_path = n["normal/path"].child("child_with_/_inside").path();
    std::cout << test_path << std::endl;

    EXPECT_EQ(test_path,"normal/path/{child_with_/_inside}");

    n.reset();
    n.add_child("child_with_/_inside") = (int64)42;

    test_path = n.child(0).path();
    std::cout << test_path << std::endl;
    EXPECT_EQ(test_path,"{child_with_/_inside}");
}



