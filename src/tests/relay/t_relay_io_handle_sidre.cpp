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
/// file: t_relay_io_handle_sidre.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include <iostream>
#include "gtest/gtest.h"

#include "t_config.hpp"


using namespace conduit;
using namespace conduit::relay;

std::string
relay_test_data_path(const std::string &test_fname)
{
    std::string res = utils::join_path(CONDUIT_T_SRC_DIR,"relay");
    res = utils::join_path(res,"data");
    return utils::join_path(res,test_fname);
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_handle, test_sidre_with_root)
{
    return;
    io::IOHandle h;
    h.open(relay_test_data_path("braid_3d_sidre_examples.root"),
           "sidre_hdf5");

    std::vector<std::string> rchld;
    h.list_child_names(rchld);

    for(int i=0;i< rchld.size();i++)
    {
        std::cout << rchld[i] << std::endl;
    }

    Node n;
    h.read("root",n);
    n.print();

    n.reset();
    h.read("root/blueprint_index",n);
    n.print();

    n.reset();
    h.read("root/number_of_trees",n);
    n.print();

    n.reset();
    h.read("root/protocol",n);
    n.print();

    n.reset();;
    h.read("root/blueprint_index/uniform/coordsets/coords/path",n);
    n.print();

    n.reset();
    h.read("0/uniform/coordsets/",n);
    n.print();

    n.reset();
    h.read("0/uniform/coordsets/coords/spacing",n);
    n.print();

    
    n.reset();
    h.read("0/uniform/topologies/",n);
    n.print();

    h.close();
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_handle, test_sidre_basic)
{
    std::string tbase = "texample_sidre_basic_ds_demo.";
    std::vector<std::string> tprotos;
    //tprotos.push_back("sidre_conduit_json");
    //tprotos.push_back("sidre_json");
    tprotos.push_back("sidre_hdf5");

    //
    // create an equiv conduit tree for testing
    //

    conduit::int64    conduit_vals_1[5] = {0,1,2,3,4};
    conduit::float64  conduit_vals_2[6] = { 1.0, 2.0,
                                            1.0, 2.0,
                                            1.0, 2.0,};

    Node n;
    n["my_scalars/i64"].set_int64(1);
    n["my_scalars/f64"].set_float64(10.0);
    n["my_strings/s0"] = "s0 string";
    n["my_strings/s1"] = "s1 string";
    n["my_arrays/a5_i64_std"].set(conduit_vals_1,5);
    n["my_arrays/a5_i64_ext"].set_external(conduit_vals_1,5);
    n["my_arrays/b_v1"].set(conduit_vals_2,
                            3,
                            0,
                            2 * sizeof(conduit::float64));
    n["my_arrays/b_v2"].set(conduit_vals_2,
                            3,
                            sizeof(conduit::float64),
                            2 * sizeof(conduit::float64));

    // change val to make sure this is reflected as external
    conduit_vals_1[4] = -5;
    CONDUIT_INFO("Conduit Test Tree:");
    n.print();

    
    for(size_t i =0; i < tprotos.size(); i++)
    {
        io::IOHandle h;
        std::string protocol = tprotos[i];
        h.open(relay_test_data_path(tbase + protocol),
               protocol);

        EXPECT_TRUE(h.has_path("my_scalars"));
        EXPECT_TRUE(h.has_path("my_strings"));
        EXPECT_TRUE(h.has_path("my_strings/s0"));

        std::vector<std::string> rchld;
        h.list_child_names(rchld);

        for(int i=0;i< rchld.size();i++)
        {
            std::cout << rchld[i] << std::endl;
        }

        Node n_read, n_info;

        // h.read(n_read);
        // n_read.print();
        // EXPECT_FALSE(n.diff(n_read,n_info));

        // check subpath read.
        n_read.reset();

        h.read("my_arrays",n_read);
        // n_read.print();
        EXPECT_FALSE(n["my_arrays"].diff(n_read,n_info));
        h.close();
    }
}


