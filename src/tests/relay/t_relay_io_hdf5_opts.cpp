// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_relay_io_hdf5_opts.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include "conduit_relay_io_hdf5.hpp"
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
    return (int)(smin + rand_float64() * (smax - smin + 1));
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
    opts["output_file"]         = "tout_hdf5_opts_test.hdf5";
    

    if(opts_file != "")
    {
        CONDUIT_INFO("Using opts file:" << opts_file);
        io::load_merged(opts_file,opts);
        io::hdf5_set_options(opts["hdf5"]);
    }
    
    CONDUIT_INFO("options:" << opts.to_json());
    
    Node rl_about;
    relay::about(rl_about["io"]);
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
            oss.str("");
            oss << "child_" << j;
            Node &cld = parent[oss.str()];
            
            int ds_size = rand_size(ds_min,ds_max);
            cld.set(DataType::float64(ds_size));
            float64_array vals = cld.value();
            rand_fill(vals);
        }
    }
    
    
    CONDUIT_INFO("total data size = " << n.total_bytes_compact());
    
    std::string ofile = opts["output_file"].as_string();
    
    CONDUIT_INFO("Writing to " << ofile);
    io::hdf5_write(n,ofile);

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


