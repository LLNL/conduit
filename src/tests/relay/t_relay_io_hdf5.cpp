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
/// file: t_relay_io_hdf5.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include "conduit_relay_hdf5.hpp"
#include "hdf5.h"
#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::relay;


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_write_read_by_file_name)
{
    uint32 a_val = 20;
    uint32 b_val = 8;
    uint32 c_val = 13;
    uint32 d_val = 121;
    
    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;

    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["b"].as_uint32(), b_val);
    EXPECT_EQ(n["c"].as_uint32(), c_val);

    // write our node as a group @ "myobj"
    io::hdf5_write(n,"tout_hdf5_wr.hdf5:myobj");

    // directly read our object
    Node n_load;
    io::hdf5_read("tout_hdf5_wr.hdf5:myobj",n_load);

    n_load.print_detailed();

    EXPECT_EQ(n_load["a"].as_uint32(), a_val);
    EXPECT_EQ(n_load["b"].as_uint32(), b_val);
    EXPECT_EQ(n_load["c"].as_uint32(), c_val);

    Node n_load_2;
    // read from root of hdf5 file
    io::hdf5_read("tout_hdf5_wr.hdf5",n_load_2);
    
    EXPECT_EQ(n_load_2["myobj/a"].as_uint32(), a_val);
    EXPECT_EQ(n_load_2["myobj/b"].as_uint32(), b_val);
    EXPECT_EQ(n_load_2["myobj/c"].as_uint32(), c_val);


    Node n_load_generic;
    // read from root of hdf5 file
    io::load("tout_hdf5_wr.hdf5",n_load_generic);
    
    EXPECT_EQ(n_load_generic["myobj/a"].as_uint32(), a_val);
    EXPECT_EQ(n_load_generic["myobj/b"].as_uint32(), b_val);
    EXPECT_EQ(n_load_generic["myobj/c"].as_uint32(), c_val);
    
    

    
    // save load from generic io interface 
    io::save(n_load_generic["myobj"],"tout_hdf5_wr_generic.hdf5:myobj");

    n_load_generic["myobj/d"] = d_val;
    
    io::load_merged("tout_hdf5_wr_generic.hdf5",n_load_generic);
    
    EXPECT_EQ(n_load_generic["myobj/a"].as_uint32(), a_val);
    EXPECT_EQ(n_load_generic["myobj/b"].as_uint32(), b_val);
    EXPECT_EQ(n_load_generic["myobj/c"].as_uint32(), c_val);
    EXPECT_EQ(n_load_generic["myobj/d"].as_uint32(), d_val);
    

}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_write_read_special_paths)
{
    uint32 a_val = 20;
    uint32 b_val = 8;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;

    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["b"].as_uint32(), b_val);

    // write our node as a group @ "/myobj"
    io::hdf5_write(n,"tout_hdf5_wr_special_paths_1.hdf5:/myobj");

    // write our node as a group @ "/"
    // make sure "/" works
    io::hdf5_write(n,"tout_hdf5_wr_special_paths_2.hdf5:/");
    
    // make sure empty after ":" this works
    io::hdf5_write(n,"tout_hdf5_wr_special_paths_3.hdf5:");
    

    Node n_load;
    io::hdf5_read("tout_hdf5_wr_special_paths_2.hdf5:/",n_load);
    EXPECT_EQ(n_load["a"].as_uint32(), a_val);
    EXPECT_EQ(n_load["b"].as_uint32(), b_val);

    n_load.reset();
    
    io::hdf5_read("tout_hdf5_wr_special_paths_2.hdf5:/",n_load);
    EXPECT_EQ(n_load["a"].as_uint32(), a_val);
    EXPECT_EQ(n_load["b"].as_uint32(), b_val);

    n_load.reset();

    io::hdf5_read("tout_hdf5_wr_special_paths_2.hdf5:",n_load);
    EXPECT_EQ(n_load["a"].as_uint32(), a_val);
    EXPECT_EQ(n_load["b"].as_uint32(), b_val);
}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_write_read_string)
{
    uint32 a_val = 20;
    
    std::string s_val = "{string value!}";

    Node n;
    n["a"] = a_val;
    n["s"] = s_val;

    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["s"].as_string(), s_val);

    // write our node as a group @ "myobj"
    io::hdf5_write(n,"tout_hdf5_wr_string.hdf5:myobj");

    Node n_out;
    
    io::hdf5_read("tout_hdf5_wr_string.hdf5:myobj",n_out);
    
    EXPECT_EQ(n_out["a"].as_uint32(), a_val);
    EXPECT_EQ(n_out["s"].as_string(), s_val);
    

}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_write_read_array)
{
    Node n_in(DataType::float64(10));
    
    float64_array val_in = n_in.value();
    
    for(index_t i=0;i<10;i++)
    {
        val_in[i] = 3.1415 * i;
    }

    // write our node as a group @ "myobj"
    io::hdf5_write(n_in,"tout_hdf5_wr_array.hdf5:myobj");

    Node n_out;
    
    io::hdf5_read("tout_hdf5_wr_array.hdf5:myobj",n_out);
    
    float64_array val_out = n_out.value();
    
    
    for(index_t i=0;i<10;i++)
    {
        EXPECT_EQ(val_in[i],val_out[i]);
    }

}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, write_and_read_conduit_leaf_to_hdf5_dataset_handle)
{
    std::string ofname = "tout_hdf5_wr_conduit_leaf_to_hdf5_dataset_handle.hdf5";

    hid_t h5_file_id = H5Fcreate(ofname.c_str(),
                                 H5F_ACC_TRUNC,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);

    // create a dataset for a 16-bit signed integer  array with 2 elements


    hid_t h5_dtype = H5T_NATIVE_SHORT;

    hsize_t num_eles = 2;
    
    hid_t   h5_dspace_id = H5Screate_simple(1,
                                            &num_eles,
                                            NULL);

    // create new dataset
    hid_t h5_dset_id  = H5Dcreate(h5_file_id,
                                  "mydata",
                                  h5_dtype,
                                  h5_dspace_id,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT);

    Node n;
    n.set(DataType::c_short(2));
    short_array vals = n.value();

    vals[0] = -16;
    vals[1] = -16;

    // this should succeed 
    io::hdf5_write(n,h5_dset_id);
    

    // this should also succeed 
    vals[1] = 16;
    
    io::hdf5_write(n,h5_dset_id);
    
    n.set(DataType::uint16(10));
    // this should fail
    EXPECT_THROW(io::hdf5_write(n,h5_dset_id),Error);

    Node n_read;
    io::hdf5_read(h5_dset_id,n_read);

    // check values of data
    short_array read_vals = n_read.value();
    EXPECT_EQ(-16,read_vals[0]);
    EXPECT_EQ(16,read_vals[1]);

    H5Sclose(h5_dspace_id);
    H5Dclose(h5_dset_id);
    H5Fclose(h5_file_id);


}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, write_conduit_object_to_hdf5_group_handle)
{
    std::string ofname = "tout_hdf5_wr_conduit_object_to_hdf5_group_handle.hdf5";

    hid_t h5_file_id = H5Fcreate(ofname.c_str(),
                                 H5F_ACC_TRUNC,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);

    hid_t h5_group_id = H5Gcreate(h5_file_id,
                                  "mygroup",
                                  H5P_DEFAULT,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT);
    
    
    Node n;
    n["a/b"].set(DataType::int16(2));
    int16_array vals = n["a/b"].value();
    vals[0] =-16;
    vals[1] =-16;
    
    // this should succeed 
    io::hdf5_write(n,h5_group_id);
    
    n["a/c"] = "mystring";
    
    // this should also succeed 
    vals[1] = 16;
    
    io::hdf5_write(n,h5_group_id);
    
    n["a/b"].set(DataType::uint16(10));
    // this should fail
    EXPECT_THROW(io::hdf5_write(n,h5_group_id),Error);

    Node n_read;
    io::hdf5_read(h5_group_id,n_read);

    // check values of data
    int16_array read_vals = n_read["a/b"].value();
    EXPECT_EQ(-16,read_vals[0]);
    EXPECT_EQ(16,read_vals[1]);
    EXPECT_EQ("mystring",n_read["a/c"].as_string());
    
    H5Gclose(h5_group_id);
    H5Fclose(h5_file_id);
}

//-----------------------------------------------------------------------------
// This variant tests when a caller code has already opened a HDF5 file
// and has a handle ready.
TEST(conduit_relay_io_hdf5, conduit_hdf5_write_read_by_file_handle)
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

    std::string test_file_name = "tout_hdf5_write_read_by_file_handle.hdf5";

    // Set up hdf5 file and group that caller code would already have.
    hid_t  h5_file_id = H5Fcreate(test_file_name.c_str(),
                           H5F_ACC_TRUNC,
                           H5P_DEFAULT,
                           H5P_DEFAULT);

    // Prepare group that caller code wants conduit to save it's tree to that
    // group. (could also specify group name for conduit to create via
    // hdf5_path argument to write call.
    hid_t h5_group_id = H5Gcreate(h5_file_id,
                            "sample_group_name",
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);

    io::hdf5_write(n,h5_group_id);
    hid_t status = H5Gclose(h5_group_id);

    // Another variant of this - caller code has a pre-existing group they
    // want to write into, but they want to use the 'group name' arg to do it
    // Relay should be able to write into existing group.
    h5_group_id = H5Gcreate(h5_file_id,
                            "sample_group_name2",
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);
    io::hdf5_write(n,h5_file_id, "sample_group_name2");

    status = H5Gclose(h5_group_id);

    status = H5Fclose(h5_file_id);

    h5_file_id = H5Fopen(test_file_name.c_str(),
                         H5F_ACC_RDONLY,
                         H5P_DEFAULT);

    // Caller code switches to group it wants to read in. (could also
    // specify group name for conduit to read out via hdf5_path arg to read
    // call)
    h5_group_id = H5Gopen(h5_file_id, "sample_group_name", 0);
                          
    Node n_load;

    io::hdf5_read(h5_group_id, n_load);
    
    status = H5Gclose(h5_group_id);
    status = H5Fclose(h5_file_id);

    EXPECT_EQ(n_load["a"].as_uint32(), a_val);
    EXPECT_EQ(n_load["b"].as_uint32(), b_val);
    EXPECT_EQ(n_load["c"].as_uint32(), c_val);
    
}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_write_to_existing_dset)
{
    Node n_in(DataType::uint32(2));
    
    
    uint32_array val_in = n_in.value();
    
    val_in[0] = 1;
    val_in[1] = 2;
    
    
    // Set up hdf5 file and group that caller code would already have.
    hid_t  h5_file_id = H5Fcreate("tout_hdf5_wr_existing_dset.hdf5",
                                  H5F_ACC_TRUNC,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT);
    
    
    io::hdf5_write(n_in,h5_file_id,"myarray");
    
    
    val_in[0] = 3;
    val_in[1] = 4;

    io::hdf5_write(n_in,h5_file_id,"myarray");

    // trying to write an incompatible dataset will throw an error
    Node n_incompat;
    n_incompat = 64;
    EXPECT_THROW(io::hdf5_write(n_incompat,h5_file_id,"myarray"),
                 conduit::Error);

    
    H5Fclose(h5_file_id);
    
    // check that the second set of values are the ones we get back
    
    Node n_read;
    
    io::hdf5_read("tout_hdf5_wr_existing_dset.hdf5:myarray",n_read);
    
    uint32_array val = n_read.value();
    
    EXPECT_EQ(val[0],3);
    EXPECT_EQ(val[1],4);
    
    Node n_w2;
    n_w2["myarray"].set_external(n_read);
    n_w2["a/b/c"].set_uint64(123);
    
    // this should be compatible 
    io::hdf5_write(n_w2,"tout_hdf5_wr_existing_dset.hdf5");
    
    n_read.reset();
    
    io::hdf5_read("tout_hdf5_wr_existing_dset.hdf5",n_read);
    
    
    uint32_array myarray_val = n_read["myarray"].value();
    
    uint64 a_b_c_val = n_read["a/b/c"].value();
    
    EXPECT_EQ(myarray_val[0],3);
    EXPECT_EQ(myarray_val[1],4);
    EXPECT_EQ(a_b_c_val,123);
    
    
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_write_read_leaf_arrays)
{
    Node n;
    
    n["v_int8"].set(DataType::int8(5));
    n["v_int16"].set(DataType::int16(5));
    n["v_int32"].set(DataType::int32(5));
    n["v_int64"].set(DataType::int64(5));
    
    n["v_uint8"].set(DataType::uint8(5));
    n["v_uint16"].set(DataType::uint16(5));
    n["v_uint32"].set(DataType::uint32(5));
    n["v_uint64"].set(DataType::uint64(5));
    
    n["v_float32"].set(DataType::float32(5));
    n["v_float64"].set(DataType::float64(5));
    
    n["v_string"].set("my_string");
    
    
    int8  *v_int8_ptr  = n["v_int8"].value();
    int16 *v_int16_ptr = n["v_int16"].value();
    int32 *v_int32_ptr = n["v_int32"].value();
    int64 *v_int64_ptr = n["v_int64"].value();

    uint8  *v_uint8_ptr  = n["v_uint8"].value();
    uint16 *v_uint16_ptr = n["v_uint16"].value();
    uint32 *v_uint32_ptr = n["v_uint32"].value();
    uint64 *v_uint64_ptr = n["v_uint64"].value();

    float32 *v_float32_ptr = n["v_float32"].value();
    float64 *v_float64_ptr = n["v_float64"].value();

    for(index_t i=0; i < 5; i++)
    {
        v_int8_ptr[i]  = -8;
        v_int16_ptr[i] = -16;
        v_int32_ptr[i] = -32;
        v_int64_ptr[i] = -64;

        v_uint8_ptr[i]  = 8;
        v_uint16_ptr[i] = 16;
        v_uint32_ptr[i] = 32;
        v_uint64_ptr[i] = 64;

        v_float32_ptr[i] = 32.0;
        v_float64_ptr[i] = 64.0;
    }
    
    n.print_detailed();
    
    io::hdf5_write(n,"tout_hdf5_wr_leaf_arrays.hdf5");
    
    
    
    Node n_load;
    
    io::hdf5_read("tout_hdf5_wr_leaf_arrays.hdf5",n_load);
    
    n_load.print_detailed();
    
    

    int8_array  v_int8_out  = n_load["v_int8"].value();
    int16_array v_int16_out = n_load["v_int16"].value();
    int32_array v_int32_out = n_load["v_int32"].value();
    int64_array v_int64_out = n_load["v_int64"].value();
    
    EXPECT_EQ(v_int8_out.number_of_elements(),5);
    EXPECT_EQ(v_int16_out.number_of_elements(),5);
    EXPECT_EQ(v_int32_out.number_of_elements(),5);
    EXPECT_EQ(v_int64_out.number_of_elements(),5);

    uint8_array  v_uint8_out  = n_load["v_uint8"].value();
    uint16_array v_uint16_out = n_load["v_uint16"].value();
    uint32_array v_uint32_out = n_load["v_uint32"].value();
    uint64_array v_uint64_out = n_load["v_uint64"].value();

    EXPECT_EQ(v_uint8_out.number_of_elements(),5);
    EXPECT_EQ(v_uint16_out.number_of_elements(),5);
    EXPECT_EQ(v_uint32_out.number_of_elements(),5);
    EXPECT_EQ(v_uint64_out.number_of_elements(),5);


    float32_array v_float32_out = n_load["v_float32"].value();
    float64_array v_float64_out = n_load["v_float64"].value();

    EXPECT_EQ(v_float32_out.number_of_elements(),5);
    EXPECT_EQ(v_float64_out.number_of_elements(),5);


    std::string v_string_out = n_load["v_string"].as_string();
    
    EXPECT_EQ(v_string_out,"my_string");
}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_write_read_empty)
{
    Node n;
    n["path/to/empty"];
    n.print_detailed();
    
    io::hdf5_write(n,"tout_hdf5_wr_empty.hdf5");
    
    Node n_load;
    io::hdf5_read("tout_hdf5_wr_empty.hdf5",n_load);
    n_load.print_detailed();
    
    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              n_load["path/to/empty"].dtype().id());
}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_write_read_childless_object)
{
    Node n;
    n["path/to/empty"].set(DataType::object());
    n.print_detailed();
    
    io::hdf5_write(n,"tout_hdf5_wr_cl_obj.hdf5");
    
    Node n_load;
    io::hdf5_read("tout_hdf5_wr_cl_obj.hdf5",n_load);
    n_load.print_detailed();
    
    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              n_load["path/to/empty"].dtype().id());
}



//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_test_write_incompat)
{
    
    Node n;
    n["a/b/leaf"] = DataType::uint32(2);
    n["a/b/grp/leaf"].set_uint32(10);
    
    uint32_array vals =  n["a/b/leaf"].value();
    
    vals[0] = 1;
    vals[1] = 2;
    
    io::hdf5_write(n,"tout_hdf5_test_write_incompat.hdf5");

    n.print();

    Node n2;
    n2["a/b/leaf/v"] = DataType::float64(2);
    n2["a/b/grp/leaf/v"].set_float64(10.0);
    
    n2.print();

    hid_t h5_file_id = H5Fopen("tout_hdf5_test_write_incompat.hdf5",
                               H5F_ACC_RDWR,
                               H5P_DEFAULT);

    try
    {
        io::hdf5_write(n2,h5_file_id);
    }
    catch(Error &e)
    {
        CONDUIT_INFO(e.message());
    }
    
    H5Fclose(h5_file_id);

}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, auto_endian)
{
    Node n;
    n["a"].set_int64(12345689);
    n["b"].set_int64(-12345689);
    
    if(Endianness::machine_is_big_endian())
    {
        n.endian_swap_to_little();
    }
    else
    {
        n.endian_swap_to_big();
    }

    io::hdf5_write(n,"tout_hdf5_wr_opp_endian.hdf5");
    
    Node n_load;
    io::hdf5_read("tout_hdf5_wr_opp_endian.hdf5",n_load);

    EXPECT_EQ(n_load["a"].as_int64(),12345689);
    EXPECT_EQ(n_load["b"].as_int64(),-12345689);

}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, hdf5_path_exists)
{

    std::string test_file_name = "tout_hdf5_wr_hdf5_path_exists.hdf5";

    Node n;
    n["a/b/c/d"] = 10;
    n["a/b/c/f"] = 20;

    io::hdf5_write(n,test_file_name);
    
    
    hid_t h5_file_id = H5Fopen(test_file_name.c_str(),
                               H5F_ACC_RDONLY,
                               H5P_DEFAULT);

    hid_t h5_grp_a = H5Gopen(h5_file_id, "a", 0);
    

    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"a"));
    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"a/b"));
    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"a/b/c"));
    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"a/b/c/d"));
    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"a/b/c/f"));

    EXPECT_TRUE(io::hdf5_has_path(h5_grp_a,"b"));
    EXPECT_TRUE(io::hdf5_has_path(h5_grp_a,"b/c"));
    EXPECT_TRUE(io::hdf5_has_path(h5_grp_a,"b/c/d"));
    EXPECT_TRUE(io::hdf5_has_path(h5_grp_a,"b/c/f"));


    EXPECT_FALSE(io::hdf5_has_path(h5_file_id,"BAD"));
    EXPECT_FALSE(io::hdf5_has_path(h5_file_id,"a/BAD"));
    EXPECT_FALSE(io::hdf5_has_path(h5_file_id,"a/b/BAD"));
    EXPECT_FALSE(io::hdf5_has_path(h5_file_id,"a/b/c/BAD"));
    EXPECT_FALSE(io::hdf5_has_path(h5_file_id,"a/b/c/d/e/f/g"));

    EXPECT_FALSE(io::hdf5_has_path(h5_grp_a,"BAD"));
    EXPECT_FALSE(io::hdf5_has_path(h5_grp_a,"b/BAD"));
    EXPECT_FALSE(io::hdf5_has_path(h5_grp_a,"b/c/BAD"));
    EXPECT_FALSE(io::hdf5_has_path(h5_grp_a,"b/c/d/e/f/g"));

        
    H5Gclose(h5_grp_a);
    H5Fclose(h5_file_id);
}






