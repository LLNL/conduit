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
/// file: t_relay_io_basic.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::relay;


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_basic, basic_bin)
{
    uint32 a_val = 20;
    uint32 b_val = 8;
    uint32 c_val = 13;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;

    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["b"].as_uint32(), b_val);
    EXPECT_EQ(n["c"].as_uint32(), c_val);
    

    io::save(n, "test_conduit_relay_io_dump.conduit_bin");

    Node n_load;
    io::load("test_conduit_relay_io_dump.conduit_bin",n_load);
    
    EXPECT_EQ(n_load["a"].as_uint32(), a_val);
    EXPECT_EQ(n_load["b"].as_uint32(), b_val);
    EXPECT_EQ(n_load["c"].as_uint32(), c_val);
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_basic, json)
{
    uint32 a_val = 20;
    uint32 b_val = 8;
    uint32 c_val = 13;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;

    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["b"].as_uint32(), b_val);
    EXPECT_EQ(n["c"].as_uint32(), c_val);
    

    io::save(n, "test_conduit_relay_io_dump.json");

    Node n_load;
    io::load("test_conduit_relay_io_dump.json",n_load);
    
    // note type diff for pure json
    EXPECT_EQ(n_load["a"].as_int64(), a_val);
    EXPECT_EQ(n_load["b"].as_int64(), b_val);
    EXPECT_EQ(n_load["c"].as_int64(), c_val);
    
    EXPECT_EQ(n_load["a"].to_uint32(), a_val);
    EXPECT_EQ(n_load["b"].to_uint32(), b_val);
    EXPECT_EQ(n_load["c"].to_uint32(), c_val);

    
    io::save(n, "test_conduit_relay_io_dump.conduit_json");

    n_load.reset();
    io::load("test_conduit_relay_io_dump.conduit_json",n_load);
    
    EXPECT_EQ(n_load["a"].as_uint32(), a_val);
    EXPECT_EQ(n_load["b"].as_uint32(), b_val);
    EXPECT_EQ(n_load["c"].as_uint32(), c_val);

    io::save(n, "test_conduit_relay_io_dump.conduit_base64_json");

    n_load.reset();
    io::load("test_conduit_relay_io_dump.conduit_base64_json",n_load);
    
    EXPECT_EQ(n_load["a"].as_uint32(), a_val);
    EXPECT_EQ(n_load["b"].as_uint32(), b_val);
    EXPECT_EQ(n_load["c"].as_uint32(), c_val);
}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_basic, yaml)
{
    uint32 a_val = 20;
    uint32 b_val = 8;
    uint32 c_val = 13;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;

    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["b"].as_uint32(), b_val);
    EXPECT_EQ(n["c"].as_uint32(), c_val);
    

    io::save(n, "test_conduit_relay_io_dump.yaml");

    Node n_load;
    io::load("test_conduit_relay_io_dump.yaml",n_load);
    
    // note type diff for pure json
    EXPECT_EQ(n_load["a"].as_int64(), a_val);
    EXPECT_EQ(n_load["b"].as_int64(), b_val);
    EXPECT_EQ(n_load["c"].as_int64(), c_val);
    
    EXPECT_EQ(n_load["a"].to_uint32(), a_val);
    EXPECT_EQ(n_load["b"].to_uint32(), b_val);
    EXPECT_EQ(n_load["c"].to_uint32(), c_val);

}



//-----------------------------------------------------------------------------
TEST(conduit_relay_io_basic, identify_protocol)
{
    std::string protocol;

    // conduit bin check
    io::identify_protocol("test.conduit_bin",protocol);
    EXPECT_EQ(protocol,"conduit_bin");

    // json checks
    io::identify_protocol("test.conduit_json",protocol);
    EXPECT_EQ(protocol,"conduit_json");

    io::identify_protocol("test.conduit_base64_json",protocol);
    EXPECT_EQ(protocol,"conduit_base64_json");

    io::identify_protocol("test.json",protocol);
    EXPECT_EQ(protocol,"json");

    // yaml check
    io::identify_protocol("test.yaml",protocol);
    EXPECT_EQ(protocol,"yaml");

    // silo check
    io::identify_protocol("test.silo",protocol);
    EXPECT_EQ(protocol,"conduit_silo");

    // hdf5 checks
    io::identify_protocol("test.hdf5",protocol);
    EXPECT_EQ(protocol,"hdf5");

    io::identify_protocol("test.h5",protocol);
    EXPECT_EQ(protocol,"hdf5");

    // adios checks
    io::identify_protocol("test.bp",protocol);
    EXPECT_EQ(protocol,"adios");

    io::identify_protocol("test.bp",protocol);
    EXPECT_EQ(protocol,"adios");

    io::identify_protocol("test.adios",protocol);
    EXPECT_EQ(protocol,"adios");
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_basic, save_empty)
{
    Node n;
    io::save(n, "test_conduit_relay_io_save_empty.conduit_bin");
}
