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


std::string opts_file = "";


//-----------------------------------------------------------------------------
float64
rand_float64()
{
    return float64(rand()) / float64(RAND_MAX);
}


//-----------------------------------------------------------------------------
int
rand_size(int smin, int smax)
{
    return smin + rand_float64() * (smax - smin + 1);
}

//-----------------------------------------------------------------------------
void
rand_fill(float64_array &vals)
{
    // this is bad for compression tests, we would be better of with
    // a more rep noise source (smoothed, perlin-like, etc)
    for(int i=0;i< vals.dtype().number_of_elements();i++)
    {
        vals[i] = rand_float64();
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_write_synth)
{
    Node opts;

    opts["data/num_objects"] = 10;
    opts["data/num_leaves"]  = 10;
    opts["data/leaf_size_min"]  = 5;
    opts["data/leaf_size_max"]  = 1000;
    opts["data/leaf_seed"]      = 0;
    

    if(opts_file != "")
    {
        CONDUIT_INFO("Using opts file:" << opts_file);
        io::load_merged(opts_file,opts);
        io::hdf5_set_options(opts["hdf5"]);
    }
    
    CONDUIT_INFO("options:" << opts.to_json());
    
    Node rl_about;
    relay::about(rl_about);
    CONDUIT_INFO("hdf5 options:" << rl_about["io/options/hdf5"].to_json());
    
    int num_obj =  opts["data/num_objects"].to_value();
    int num_l   =  opts["data/num_leaves"].to_value();
    int ds_min  =  opts["data/leaf_size_min"].to_value();
    int ds_max  =  opts["data/leaf_size_max"].to_value();
    
    srand(opts["data/leaf_seed"].to_int());
    
    Node n;
    
    std::ostringstream oss;
    for(int i=0; i< num_obj; i++)
    {
        oss.str("");
        oss << "entry_" << i;

        Node &parent = n[oss.str()];
        
        for(int j=0; j< num_l; j++)
        {
            oss << "child_" << j;
            Node &cld = parent[oss.str()];
            
            int ds_size = rand_size(ds_min,ds_max);
            cld.set(DataType::float64(ds_size));
            float64_array vals = cld.value();
            rand_fill(vals);
        }
    }
    
    
    CONDUIT_INFO("total data size = " << n.total_bytes_compact());
    
    io::hdf5_write(n,"tout_hdf5_opts_test.hdf5");
    
    //Node n_load;
    //io::hdf5_read("tout_hdf5_opts_test.hdf5",n_load);

}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    for(int i=0; i < argc ; i++)
    {
        std::string arg_str(argv[i]);
        if(arg_str == "--opts" && i+1 < argc)
        {
            opts_file = std::string(argv[i+1]);
        }
    }

    result = RUN_ALL_TESTS();
    return result;
}


