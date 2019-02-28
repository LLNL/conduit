//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
/// file: t_relay_io_handle.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::relay;


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_handle, basic_bin)
{
    std::string test_file = "test_conduit_relay_io_handle.conduit_bin";

    uint32 a_val = 20;
    uint32 b_val = 8;
    uint32 c_val = 13;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;
    n["d/here"] = 10;


    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["b"].as_uint32(), b_val);
    EXPECT_EQ(n["c"].as_uint32(), c_val);

    std::vector<std::string> cnames;

    io::IOHandle h;
    h.open(test_file);
    h.write(n);

    EXPECT_TRUE(h.has_path("d/here"));
    h.list_child_names(cnames);

    EXPECT_EQ(cnames[0],"a");
    EXPECT_EQ(cnames[1],"b");
    EXPECT_EQ(cnames[2],"c");
    EXPECT_EQ(cnames[3],"d");

    h.list_child_names("d",cnames);
    EXPECT_EQ(cnames[0],"here");

    h.remove("d");
    EXPECT_FALSE(h.has_path("d"));
    EXPECT_FALSE(h.has_path("d/here"));
    h.close();

    Node n2;
    io::IOHandle h2;
    h2.open(test_file);
    h2.list_child_names(cnames);
    EXPECT_EQ(cnames[0],"a");
    EXPECT_EQ(cnames[1],"b");
    EXPECT_EQ(cnames[2],"c");

    Node n_val;
    n_val = 10;
    // write with path
    h2.write(n_val,"d/here");

    h2.read(n2);
    Node info;
    EXPECT_FALSE(n.diff(n2, info, 0.0));
    info.print();

    // read with path
    n_val.reset();
    h2.read("c",n_val);

    EXPECT_EQ(n_val.as_uint32(),c_val);

    h2.close();
}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_handle, hdf5)
{
    std::string test_file = "test_conduit_relay_io_handle.hdf5";

    uint32 a_val = 20;
    uint32 b_val = 8;
    uint32 c_val = 13;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;
    n["d/here"] = 10;


    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["b"].as_uint32(), b_val);
    EXPECT_EQ(n["c"].as_uint32(), c_val);

    std::vector<std::string> cnames;

    io::IOHandle h;
    h.open(test_file);
    h.write(n);

    EXPECT_TRUE(h.has_path("d/here"));
    h.list_child_names(cnames);

    EXPECT_EQ(cnames[0],"a");
    EXPECT_EQ(cnames[1],"b");
    EXPECT_EQ(cnames[2],"c");
    EXPECT_EQ(cnames[3],"d");

    h.list_child_names("d",cnames);
    EXPECT_EQ(cnames[0],"here");

    h.remove("d");
    EXPECT_FALSE(h.has_path("d"));
    EXPECT_FALSE(h.has_path("d/here"));
    h.close();

    Node n2;
    io::IOHandle h2;
    h2.open(test_file);
    h2.list_child_names(cnames);
    EXPECT_EQ(cnames[0],"a");
    EXPECT_EQ(cnames[1],"b");
    EXPECT_EQ(cnames[2],"c");

    Node n_val;
    n_val = 10;
    // write with path
    h2.write(n_val,"d/here");

    h2.read(n2);
    Node info;
    EXPECT_FALSE(n.diff(n2, info, 0.0));
    info.print();

    // read with path
    n_val.reset();
    h2.read("c",n_val);

    EXPECT_EQ(n_val.as_uint32(),c_val);

    h2.close();
}



