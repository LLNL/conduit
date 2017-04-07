//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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
/// file: t_relay_io_hdf5_opts.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include "conduit_relay_hdf5.hpp"
#include "hdf5.h"
#include <iostream>
#include "gtest/gtest.h"
#include <cstdlib> 

#include <sstream>

using namespace conduit;
using namespace conduit::relay;



//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5_atts, read_right_atts_simple)
{
    Node n;

    n["test_a"].set(DataType::int64(10));
    n["test_b"].set(DataType::int64(10));

    int64_array vals_a = n["test_a"].value();
    int64_array vals_b = n["test_b"].value();
    
    for(int i=0; i < 10 ; i++)
    {
        vals_a[i] = i;
        vals_b[i] = -i;
    }


    io::hdf5_write(n,"tout_relay_io_hdf5_atts_test_std.hdf5");
    
    Node opts;
    opts["attributes/read/enabled"] = "true";
    opts["attributes/write/enabled"] = "true";
    io::hdf5_set_options(opts);
    
    Node rl_about;
    relay::about(rl_about);
    CONDUIT_INFO("hdf5 options:" << rl_about["io/options/hdf5"].to_json());

    io::hdf5_write(n,"tout_relay_io_hdf5_atts_test_atts.hdf5");
    
    
    Node load;
    
    io::hdf5_read("tout_relay_io_hdf5_atts_test_atts.hdf5",load);
    

    int64_array load_vals_a = load["test_a"].value();
    int64_array load_vals_b = load["test_b"].value();

    for(int i=0; i < 10 ; i++)
    {
        EXPECT_EQ(vals_a[i],load_vals_a[i]);
        EXPECT_EQ(vals_b[i],load_vals_b[i]);
    }
}



