// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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
TEST(conduit_relay_io_basic, identify_file_type)
{
    std::string protocol;


    if(utils::is_file("tout_ident_identify.empty"))
        utils::remove_file("tout_ident_identify.empty");

    // create an empty file!
    std::ofstream ofs;
    ofs.open("tout_ident_identify.empty");
    ofs.close();

    if(utils::is_file("tout_ident_identify.txt"))
        utils::remove_file("tout_ident_identify.txt");

    // create a text file
    ofs.open("tout_ident_identify.txt");
    ofs << "stuff!" << std::endl;
    ofs.close();

    io::identify_file_type("tout_ident_ftype.empty",protocol);
    EXPECT_EQ(protocol,"unknown");

    io::identify_file_type("tout_ident_ftype.txt",protocol);
    EXPECT_EQ(protocol,"unknown");


    if(utils::is_file("tout_ident_identify.json"))
        utils::remove_file("tout_ident_identify.json");

    if(utils::is_file("tout_ident_identify.yaml"))
        utils::remove_file("tout_ident_identify.yaml");

    Node n;
    n["answer"] = 42;

    // create json and yaml files
    n.save("tout_identify_ftype.json");
    n.save("tout_identify_ftype.yaml");

    io::identify_file_type("tout_identify_ftype.json",protocol);
    EXPECT_EQ(protocol,"json");

    // TODO: add YAML heurstic
    io::identify_file_type("tout_identify_ftype.yaml",protocol);
    EXPECT_EQ(protocol,"unknown");

    Node io_protos;
    relay::io::about(io_protos["io"]);
    bool hdf5_enabled = io_protos["io/protocols/hdf5"].as_string() == "enabled";
    if(hdf5_enabled)
    {

        if(utils::is_file("tout_ident_identify.hdf5"))
            utils::remove_file("tout_ident_identify.hdf5");

        // create a hdf5 file
        io::save(n,"tout_identify_ftype.hdf5");
        io::identify_file_type("tout_identify_ftype.hdf5",protocol);
        EXPECT_EQ(protocol,"hdf5");
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_basic, save_empty)
{
    Node n;
    io::save(n, "test_conduit_relay_io_save_empty.conduit_bin");
}
