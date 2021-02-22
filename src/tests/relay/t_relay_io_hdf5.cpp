// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_relay_io_hdf5.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include "conduit_relay_io_hdf5.hpp"
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
TEST(conduit_relay_io_hdf5, write_and_read_conduit_leaf_to_extendible_hdf5_dataset_handle_with_offset)
{
    std::string ofname = "tout_hdf5_wr_conduit_leaf_to_hdf5_extendible_dataset_handle_with_offset.hdf5";

    hid_t h5_file_id = H5Fcreate(ofname.c_str(),
                                 H5F_ACC_TRUNC,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);

    // create a dataset for a 16-bit signed integer  array with 2 elements


    hid_t h5_dtype = H5T_NATIVE_SHORT;

    hsize_t num_eles = 2;
    hsize_t dims[1] = {H5S_UNLIMITED};

    hid_t   h5_dspace_id = H5Screate_simple(1,
                                            &num_eles,
                                            dims);

    /*
     * Modify dataset creation properties, i.e. enable chunking.
     */
    hid_t cparms;
    hsize_t chunk_dims[1] = {1};

    cparms = H5Pcreate (H5P_DATASET_CREATE);
    H5Pset_chunk(cparms, 1, chunk_dims);


    // create new dataset
    hid_t h5_dset_id  = H5Dcreate1(h5_file_id,
                                   "mydata",
                                   h5_dtype,
                                   h5_dspace_id,
                                   cparms);

    Node n, opts;
    n.set(DataType::c_short(2));
    short_array vals = n.value();

    vals[0] = -16;
    vals[1] = -15;

    // this should succeed
    io::hdf5_write(n,h5_dset_id);

    vals[0] = 1;
    vals[1] = 2;
    opts["offset"] = 2;
    opts["stride"] = 1;

    io::hdf5_write(n,h5_dset_id,opts);

    Node n_read, opts_read;

    io::hdf5_read_info(h5_dset_id,opts_read,n_read);
    EXPECT_EQ(4,(int) n_read["num_elements"].to_value());

    io::hdf5_read(h5_dset_id,opts_read,n_read);

    // check values of data
    short_array read_vals = n_read.value();
    EXPECT_EQ(-16,read_vals[0]);
    EXPECT_EQ(-15,read_vals[1]);
    EXPECT_EQ(1,read_vals[2]);
    EXPECT_EQ(2,read_vals[3]);

    opts_read["offset"] = 2;
    opts_read["stride"] = 1;
    io::hdf5_read(h5_dset_id,opts_read,n_read);

    // check values of data
    read_vals = n_read.value();
    EXPECT_EQ(1,read_vals[0]);
    EXPECT_EQ(2,read_vals[1]);

    vals[0] = -1;
    vals[1] = -3;
    opts["offset"] = 0;
    opts["stride"] = 2;

    io::hdf5_write(n,h5_dset_id,opts);

    opts_read["offset"] = 0;
    opts_read["stride"] = 1;
    io::hdf5_read(h5_dset_id,opts_read,n_read);

    // check values of data
    read_vals = n_read.value();
    EXPECT_EQ(-1,read_vals[0]);
    EXPECT_EQ(-15,read_vals[1]);
    EXPECT_EQ(-3,read_vals[2]);
    EXPECT_EQ(2,read_vals[3]);

    vals[0] = 5;
    vals[1] = 6;
    opts["offset"] = 7;
    opts["stride"] = 1;

    io::hdf5_write(n,h5_dset_id,opts);

    opts_read["offset"] = 0;
    opts_read["stride"] = 1;
    io::hdf5_read(h5_dset_id,opts_read,n_read);
    // check values of data
    read_vals = n_read.value();
    EXPECT_EQ(-1,read_vals[0]);
    EXPECT_EQ(-15,read_vals[1]);
    EXPECT_EQ(-3,read_vals[2]);
    EXPECT_EQ(2,read_vals[3]);
    EXPECT_EQ(0,read_vals[4]);
    EXPECT_EQ(0,read_vals[5]);
    EXPECT_EQ(0,read_vals[6]);
    EXPECT_EQ(5,read_vals[7]);
    EXPECT_EQ(6,read_vals[8]);

    opts["offset"] = -1;
    opts["stride"] = 2;
    opts_read["offset"] = -1;
    opts_read["stride"] = 2;

    //this should fail
    EXPECT_THROW(io::hdf5_write(n,h5_dset_id,opts),Error);
    EXPECT_THROW(io::hdf5_read(h5_dset_id,opts_read,n_read),Error);

    opts["offset"] = 0;
    opts["stride"] = 0;
    opts_read["offset"] = 0;
    opts_read["stride"] = 0;

    //this should fail
    EXPECT_THROW(io::hdf5_write(n,h5_dset_id,opts),Error);
    EXPECT_THROW(io::hdf5_read(h5_dset_id,opts_read,n_read),Error);

    H5Sclose(h5_dspace_id);
    H5Dclose(h5_dset_id);
    H5Fclose(h5_file_id);


}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, write_and_read_conduit_leaf_to_fixed_hdf5_dataset_handle_with_offset)
{
    std::string ofname = "tout_hdf5_wr_conduit_leaf_to_fixed_hdf5_dataset_handle_with_offset.hdf5";

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

    Node n, opts;
    n.set(DataType::c_short(2));
    short_array vals = n.value();

    vals[0] = -16;
    vals[1] = -15;

    // this should succeed
    io::hdf5_write(n,h5_dset_id);

    vals[0] = 1;
    vals[1] = 2;
    opts["offset"] = 2;
    opts["stride"] = 1;

    io::hdf5_write(n,h5_dset_id,opts);

    Node n_read, opts_read;
    io::hdf5_read_info(h5_dset_id,opts_read,n_read);
    EXPECT_EQ(4,(int) n_read["num_elements"].to_value());

    io::hdf5_read(h5_dset_id,opts_read,n_read);

    // check values of data
    short_array read_vals = n_read.value();
    EXPECT_EQ(-16,read_vals[0]);
    EXPECT_EQ(-15,read_vals[1]);
    EXPECT_EQ(1,read_vals[2]);
    EXPECT_EQ(2,read_vals[3]);

    opts_read["offset"] = 2;
    opts_read["stride"] = 1;
    io::hdf5_read(h5_dset_id,opts_read,n_read);

    // check values of data
    read_vals = n_read.value();
    EXPECT_EQ(1,read_vals[0]);
    EXPECT_EQ(2,read_vals[1]);

    vals[0] = -1;
    vals[1] = -3;
    opts["offset"] = 0;
    opts["stride"] = 2;

    io::hdf5_write(n,h5_dset_id,opts);

    opts_read["offset"] = 0;
    opts_read["stride"] = 1;
    io::hdf5_read(h5_dset_id,opts_read,n_read);

    // check values of data
    read_vals = n_read.value();
    EXPECT_EQ(-1,read_vals[0]);
    EXPECT_EQ(-15,read_vals[1]);
    EXPECT_EQ(-3,read_vals[2]);
    EXPECT_EQ(2,read_vals[3]);

    vals[0] = 5;
    vals[1] = 6;
    opts["offset"] = 7;
    opts["stride"] = 1;

    io::hdf5_write(n,h5_dset_id,opts);

    opts_read["offset"] = 0;
    opts_read["stride"] = 1;
    io::hdf5_read(h5_dset_id,opts_read,n_read);
    // check values of data
    read_vals = n_read.value();
    EXPECT_EQ(-1,read_vals[0]);
    EXPECT_EQ(-15,read_vals[1]);
    EXPECT_EQ(-3,read_vals[2]);
    EXPECT_EQ(2,read_vals[3]);
    EXPECT_EQ(0,read_vals[4]);
    EXPECT_EQ(0,read_vals[5]);
    EXPECT_EQ(0,read_vals[6]);
    EXPECT_EQ(5,read_vals[7]);
    EXPECT_EQ(6,read_vals[8]);

    opts["offset"] = -1;
    opts["stride"] = 2;
    opts_read["offset"] = -1;
    opts_read["stride"] = 2;

    //this should fail
    EXPECT_THROW(io::hdf5_write(n,h5_dset_id,opts),Error);
    EXPECT_THROW(io::hdf5_read(h5_dset_id,opts_read,n_read),Error);

    opts["offset"] = 0;
    opts["stride"] = 0;
    opts_read["offset"] = 0;
    opts_read["stride"] = 0;

    //this should fail
    EXPECT_THROW(io::hdf5_write(n,h5_dset_id,opts),Error);
    EXPECT_THROW(io::hdf5_read(h5_dset_id,opts_read,n_read),Error);

    H5Sclose(h5_dspace_id);
    H5Dclose(h5_dset_id);
    H5Fclose(h5_file_id);


}



//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, write_conduit_object_to_hdf5_group_handle_with_offset)
{
    std::string ofname = "tout_hdf5_wr_conduit_object_to_hdf5_group_handle_with_offset.hdf5";

    hid_t h5_file_id = H5Fcreate(ofname.c_str(),
                                 H5F_ACC_TRUNC,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);

    hid_t h5_group_id = H5Gcreate(h5_file_id,
                                  "mygroup",
                                  H5P_DEFAULT,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT);


    Node n, opts;
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

    Node n_read;
    io::hdf5_read(h5_group_id,n_read);

    n["a/b"].set(DataType::int16(10));
    // this should fail
    EXPECT_THROW(io::hdf5_write(n,h5_group_id),Error);

    n["a/b"].set(DataType::int16(10));
    vals = n["a/b"].value();
    opts["offset"] = 5;
    for (int i = 0; i < 10; i++) {
        vals[i] = i + 1;
    }

    io::hdf5_write(n,h5_group_id,opts);

    io::hdf5_read(h5_group_id,n_read);

    // check values of data with offset
    int16_array read_vals = n_read["a/b"].value();
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(i + 1, read_vals[i + 5]);
    }

    // this is also offset
    EXPECT_EQ("mystrmystring",n_read["a/c"].as_string());

    opts["offset"] = 20;
    opts["stride"] = 2;
    for (int i = 0; i < 10; i++) {
        vals[i] = i + 1;
    }
    n["a/d"].set(DataType::int16(5));
    int16_array vals2 = n["a/d"].value();
    for (int i = 0; i < 5; i++) {
        vals2[i] = (i + 1) * -1;
    }

    io::hdf5_write(n,h5_group_id,opts);

    io::hdf5_read(h5_group_id,n_read);

    // check values of data
    read_vals = n_read["a/b"].value();
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(i + 1, read_vals[i + 5]);
    }
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(i + 1, read_vals[2*i + 20]);
    }
    read_vals = n_read["a/d"].value();
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ((i + 1) * -1, read_vals[2*i + 20]);
    }

    Node n_read_info;
    io::hdf5_read_info(h5_group_id,n_read_info);
    EXPECT_EQ(39,(int) n_read_info["a/b/num_elements"].to_value());
    EXPECT_EQ(37,(int) n_read_info["a/c/num_elements"].to_value());
    EXPECT_EQ(29,(int) n_read_info["a/d/num_elements"].to_value());

    // this doesn't change because the null-terminated character
    // wasn't overwritten
    EXPECT_EQ("mystrmystring",n_read["a/c"].as_string());

    Node opts_read;
    opts_read["offset"] = 5;
    io::hdf5_read_info(h5_group_id,opts_read,n_read_info);
    io::hdf5_read(h5_group_id,opts_read,n_read);

    // check values of data
    read_vals = n_read["a/b"].value();
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(i + 1, read_vals[i]);
    }
    EXPECT_EQ("mystring",n_read["a/c"].as_string());
    EXPECT_EQ(34,(int) n_read_info["a/b/num_elements"].to_value());
    EXPECT_EQ(32,(int) n_read_info["a/c/num_elements"].to_value());

    opts_read["offset"] = 20;
    opts_read["stride"] = 2;
    io::hdf5_read_info(h5_group_id,opts_read,n_read_info);
    io::hdf5_read(h5_group_id,opts_read,n_read);

    // check values of data
    read_vals = n_read["a/b"].value();
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(i + 1, read_vals[i]);
    }
    read_vals = n_read["a/d"].value();
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ((i + 1) * -1, read_vals[i]);
    }
    EXPECT_EQ(10,(int) n_read_info["a/b/num_elements"].to_value());
    EXPECT_EQ(5,(int) n_read_info["a/d/num_elements"].to_value());

    opts_read["offset"] = 20;
    opts_read["stride"] = 3;
    io::hdf5_read_info(h5_group_id,opts_read,n_read_info);
    io::hdf5_read(h5_group_id,opts_read,n_read);

    // check values of data
    read_vals = n_read["a/b"].value();
    EXPECT_EQ(1, read_vals[0]);
    EXPECT_EQ(4, read_vals[2]);
    EXPECT_EQ(7, read_vals[4]);
    EXPECT_EQ(10, read_vals[6]);
    read_vals = n_read["a/d"].value();
    EXPECT_EQ(-1, read_vals[0]);
    EXPECT_EQ(-4, read_vals[2]);
    EXPECT_EQ(7,(int) n_read_info["a/b/num_elements"].to_value());
    EXPECT_EQ(3,(int) n_read_info["a/d/num_elements"].to_value());

    H5Gclose(h5_group_id);
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
TEST(conduit_relay_io_hdf5, hdf5_write_zero_sized_leaf)
{
    // this tests
    Node n;
    n["a"].set(DataType::float64(0));
    n.print_detailed();
    io::hdf5_write(n,"tout_hdf5_w_0.hdf5");

    EXPECT_EQ(n["a"].dtype().number_of_elements(),0);
    EXPECT_EQ(n["a"].dtype().id(),DataType::FLOAT64_ID);

    Node n_load;
    io::hdf5_read("tout_hdf5_w_0.hdf5",n_load);
    n_load.print_detailed();

    EXPECT_EQ(n_load["a"].dtype().number_of_elements(),0);
    EXPECT_EQ(n_load["a"].dtype().id(),DataType::FLOAT64_ID);
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


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, hdf5_create_append_methods)
{
    std::string test_file_name = "tout_hdf5_open_append.hdf5";
    utils::remove_path_if_exists(test_file_name);

    Node n;
    n["a/b/c/d"] = 10;

    io::hdf5_write(n,test_file_name,true);

    n.reset();
    n["a/b/c/e"] = 20;

    io::hdf5_write(n,test_file_name,true);


    Node n_load;
    io::hdf5_read(test_file_name,n_load);

    EXPECT_TRUE(n_load.has_path("a"));
    EXPECT_TRUE(n_load.has_path("a/b"));
    EXPECT_TRUE(n_load.has_path("a/b/c"));
    EXPECT_TRUE(n_load.has_path("a/b/c/d"));
    EXPECT_TRUE(n_load.has_path("a/b/c/e"));

    EXPECT_EQ(n_load["a/b/c/d"].to_int32(),10);
    EXPECT_EQ(n_load["a/b/c/e"].to_int32(),20);


    io::hdf5_write(n,test_file_name,false);

    n_load.reset();
    io::hdf5_read(test_file_name,n_load);

    EXPECT_FALSE(n_load.has_path("a/b/c/d"));
    EXPECT_EQ(n_load["a/b/c/e"].to_int32(),20);

    n.reset();
    n["a/b/c/d"] = 10;
    io::hdf5_save(n,test_file_name);

    n_load.reset();
    io::hdf5_read(test_file_name,n_load);

    EXPECT_FALSE(n_load.has_path("a/b/c/e"));
    EXPECT_EQ(n_load["a/b/c/d"].to_int32(),10);

    n.reset();
    n["a/b/c/e"] = 20;

    io::hdf5_write(n,test_file_name,true);

    n.reset();
    n["a/b/c/e"] = 20;

    io::hdf5_append(n,test_file_name);
    n_load.reset();
    io::hdf5_read(test_file_name,n_load);

    EXPECT_TRUE(n_load.has_path("a/b/c/d"));
    EXPECT_TRUE(n_load.has_path("a/b/c/e"));
    EXPECT_EQ(n_load["a/b/c/d"].to_int32(),10);
    EXPECT_EQ(n_load["a/b/c/e"].to_int32(),20);

}



//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, hdf5_create_open_methods)
{
    std::string test_file_name = "tout_hdf5_open_and_create.hdf5";

    Node n;
    n["a/b/c/d"] = 10;

    hid_t h5_file_id = io::hdf5_create_file(test_file_name);
    io::hdf5_write(n,h5_file_id);
    io::hdf5_close_file(h5_file_id);

    h5_file_id = io::hdf5_open_file_for_read(test_file_name);

    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"a"));
    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"a/b"));
    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"a/b/c"));
    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"a/b/c/d"));
    Node n_read;
    io::hdf5_read(h5_file_id,"a/b/c/d",n_read);

    EXPECT_EQ(10,n_read.to_int());

    io::hdf5_close_file(h5_file_id);

    h5_file_id = io::hdf5_open_file_for_read_write(test_file_name);

    Node n2;
    n2 = 12;
    io::hdf5_write(n2,h5_file_id,"a/b/c/e");
    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"a/b/c/e"));
    io::hdf5_read(h5_file_id,"a/b/c/e",n_read);
    EXPECT_EQ(12,n_read.to_int());

    io::hdf5_close_file(h5_file_id);

}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_save_generic_options)
{
    // 5k zeros, should compress well, but under default
    // threshold size

    Node n;
    n["value"] = DataType::float64(5000);

    Node opts;
    opts["hdf5/chunking/threshold"]  = 2000;
    opts["hdf5/chunking/chunk_size"] = 2000;

    std::string tout_std = "tout_hdf5_save_generic_default_options.hdf5";
    std::string tout_cmp = "tout_hdf5_save_generic_test_options.hdf5";

    utils::remove_path_if_exists(tout_std);
    utils::remove_path_if_exists(tout_cmp);

    io::save(n,tout_std, "hdf5");
    io::save(n,tout_cmp, "hdf5", opts);

    EXPECT_TRUE(utils::is_file(tout_std));
    EXPECT_TRUE(utils::is_file(tout_cmp));

    int64 tout_std_fs = utils::file_size(tout_std);
    int64 tout_cmp_fs = utils::file_size(tout_cmp);
    CONDUIT_INFO("fs test: std = "
                 << tout_std_fs
                 << ", cmp ="
                 << tout_cmp_fs);
    EXPECT_TRUE(tout_cmp_fs < tout_std_fs);
}



//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_group_list_children)
{
    Node n;

    n["path/sub1/a"];
    n["path/sub1/b"];
    n["path/sub1/c"];

    n["path/sub2/d"];
    n["path/sub2/e"];
    n["path/sub2/f"];

    std::string tout = "tout_hdf5_grp_chld_names.hdf5";

    utils::remove_path_if_exists(tout);

    io::save(n,tout, "hdf5");

    EXPECT_TRUE(utils::is_file(tout));

    hid_t h5_file_id = io::hdf5_open_file_for_read(tout);
    std::vector<std::string> cnames;

    io::hdf5_group_list_child_names(h5_file_id,"/",cnames);

    EXPECT_EQ(cnames.size(),1);
    EXPECT_EQ(cnames[0],"path");

    io::hdf5_group_list_child_names(h5_file_id,"path",cnames);

    EXPECT_EQ(cnames.size(),2);
    EXPECT_EQ(cnames[0],"sub1");
    EXPECT_EQ(cnames[1],"sub2");


    io::hdf5_group_list_child_names(h5_file_id,"path/sub1",cnames);

    EXPECT_EQ(cnames.size(),3);
    EXPECT_EQ(cnames[0],"a");
    EXPECT_EQ(cnames[1],"b");
    EXPECT_EQ(cnames[2],"c");

    io::hdf5_group_list_child_names(h5_file_id,"path/sub2",cnames);

    EXPECT_EQ(cnames.size(),3);
    EXPECT_EQ(cnames[0],"d");
    EXPECT_EQ(cnames[1],"e");
    EXPECT_EQ(cnames[2],"f");

    // check leaf, which has no children
    // this doesn't throw an error, but it creates an empty list
    io::hdf5_group_list_child_names(h5_file_id,"path/sub1/a",cnames);
    EXPECT_EQ(cnames.size(),0);

    // totally bogus paths will trigger an error
    EXPECT_THROW(io::hdf5_group_list_child_names(h5_file_id,"this/isnt/right",cnames),Error);

    // empty string won't work in this case
    EXPECT_THROW(io::hdf5_group_list_child_names(h5_file_id,"",cnames),Error);


}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, check_if_file_is_hdf5_file)
{
    Node n;
    n["path/mydata"] = 20;
    std::string tout = "tout_hdf5_check_hdf5_file.hdf5";

    utils::remove_path_if_exists(tout);

    io::save(n,tout, "hdf5");

    // this should be recoged as hdf5
    EXPECT_TRUE(io::is_hdf5_file(tout));

    // check behavior with files that have open handles
    hid_t h5_file_id = io::hdf5_open_file_for_read_write(tout);
    EXPECT_TRUE(io::is_hdf5_file(tout));
    io::hdf5_close_file(h5_file_id);

    h5_file_id = io::hdf5_open_file_for_read(tout);
    EXPECT_TRUE(io::is_hdf5_file(tout));
    io::hdf5_close_file(h5_file_id);

    tout = "tout_hdf5_check_non_hdf5_file.json";

    utils::remove_path_if_exists(tout);

    io::save(n,tout,"json");

    // this should *not* be recoged as hdf5
    EXPECT_FALSE(io::is_hdf5_file(tout));

    // check totally bad path
    EXPECT_FALSE(io::is_hdf5_file("/path/to/somewhere/that/cant/exist"));

}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, test_remove_path)
{
    Node n;
    n["path/mydata"] = 20;
    n["path/otherdata/leaf"] = 42;
    std::string tout = "tout_test_remove_path.hdf5";

    utils::remove_path_if_exists(tout);

    io::save(n,tout, "hdf5");

    hid_t h5_file_id = io::hdf5_open_file_for_read_write(tout);
    io::hdf5_remove_path(h5_file_id,"path/otherdata/leaf");
    io::hdf5_close_file(h5_file_id);

    n.reset();
    io::load(tout,n);
    EXPECT_FALSE(n.has_path("path/otherdata/leaf"));
    EXPECT_TRUE(n.has_path("path/otherdata"));
    n.print();


    h5_file_id = io::hdf5_open_file_for_read_write(tout);
    io::hdf5_remove_path(h5_file_id,"path/otherdata");
    io::hdf5_close_file(h5_file_id);

    n.reset();
    io::load(tout,n);
    EXPECT_FALSE(n.has_path("path/otherdata"));
    EXPECT_TRUE(n.has_path("path"));
    n.print();

}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, file_name_in_error)
{
    Node n;
    n["path/mydata"] = 20;
    n["path/otherdata/leaf"] = 42;
    std::string tout = "tout_our_file_to_test.hdf5";

    utils::remove_path_if_exists(tout);

    io::save(n,tout, "hdf5");

    hid_t h5_file_id = io::hdf5_open_file_for_read_write(tout);

    Node n_read;
    bool had_error = false;
    try
    {
        io::hdf5_read(h5_file_id,"bad",n_read);
    }
    catch(Error &e)
    {
        had_error = true;
        std::cout << "error message: " << e.message() ;
        // error should have the file name in it
        std::size_t found = e.message().find(tout);
        EXPECT_TRUE(found!=std::string::npos);
    }

    // make sure we took the error path
    EXPECT_TRUE(had_error);

    io::hdf5_close_file(h5_file_id);
}



//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, test_read_various_string_style)
{

    std::string tout = "tout_hdf5_wr_various_string_style.hdf5";
    hid_t h5_file_id = H5Fcreate(tout.c_str(),
                                 H5F_ACC_TRUNC,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);

    // write this string several ways and make sure relay
    // can reads them all as we expect

    // 21 chars + null term (22 total)

    std::string my_string = "this is my {} string!";


    // case 0: conduit's current way to of doing things
    Node n;
    n.set(my_string);
    io::hdf5_write(n,h5_file_id,"case_0");


    // case 1: string that reflects our current data type

    hid_t   h5_dtype_id = H5Tcopy(H5T_C_S1);
    // set size
    hsize_t num_eles = 22;
    H5Tset_size(h5_dtype_id, (size_t)num_eles);
    H5Tset_strpad(h5_dtype_id, H5T_STR_NULLTERM);
    hid_t   h5_dspace_id = H5Screate(H5S_SCALAR);

    // create new dataset
    hid_t h5_dset_id  = H5Dcreate(h5_file_id,
                                  "case_1",
                                  h5_dtype_id,
                                  h5_dspace_id,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT);
    // write data
    hid_t status = H5Dwrite(h5_dset_id,
                            h5_dtype_id,
                            H5S_ALL,
                            H5S_ALL,
                            H5P_DEFAULT,
                            my_string.c_str());

    H5Tclose(h5_dtype_id);
    H5Sclose(h5_dspace_id);
    H5Dclose(h5_dset_id);

    // case 2: string that is a simple array (old conduit way)

    h5_dtype_id = H5T_C_S1;
    num_eles = 22;
    h5_dspace_id = H5Screate_simple(1,
                                    &num_eles,
                                    NULL);

    // create new dataset
    h5_dset_id  = H5Dcreate(h5_file_id,
                            "case_2",
                            h5_dtype_id,
                            h5_dspace_id,
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);
    // write data
    status = H5Dwrite(h5_dset_id,
                      h5_dtype_id,
                      H5S_ALL,
                      H5S_ALL,
                      H5P_DEFAULT,
                      my_string.c_str());

    // H5Tclose(h5_dtype_id) -- don't need b/c we are using standard dtype
    H5Sclose(h5_dspace_id);
    H5Dclose(h5_dset_id);


    // case 3: fixed lenght with diff term style


     std::string my_string3 = "this is my {} string!        ";

    // len w/o null = 30

    h5_dtype_id = H5Tcopy(H5T_C_S1);
    num_eles = 30;
    H5Tset_size(h5_dtype_id, (size_t)num_eles);
    H5Tset_strpad(h5_dtype_id, H5T_STR_SPACEPAD);
    h5_dspace_id = H5Screate(H5S_SCALAR);

    // create new dataset
    h5_dset_id  = H5Dcreate(h5_file_id,
                            "case_3",
                            h5_dtype_id,
                            h5_dspace_id,
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);
    // write data
    status = H5Dwrite(h5_dset_id,
                      h5_dtype_id,
                      H5S_ALL,
                      H5S_ALL,
                      H5P_DEFAULT,
                      my_string3.c_str());

    H5Tclose(h5_dtype_id);
    H5Sclose(h5_dspace_id);
    H5Dclose(h5_dset_id);

    // temp buffer to create a null padded string as
    // input to write to hdf5
    Node n_tmp;
    n_tmp.set(DataType::uint8(30));

    uint8 *mystring4_char_ptr = n_tmp.value();
    // null out entire string (leave no doubt for test)
    for(int i=0; i < 30; i++)
    {
        mystring4_char_ptr[i] = 0;
    }

    // copy over part of my_string before final space pad
    for(size_t i=0; i < my_string.size(); i++)
    {
        mystring4_char_ptr[i] = my_string[i];
    }


    h5_dtype_id = H5Tcopy(H5T_C_S1);
    num_eles = 30;
    H5Tset_size(h5_dtype_id, (size_t)num_eles);
    H5Tset_strpad(h5_dtype_id, H5T_STR_NULLPAD);
    h5_dspace_id = H5Screate(H5S_SCALAR);

    // create new dataset
    h5_dset_id  = H5Dcreate(h5_file_id,
                            "case_4",
                            h5_dtype_id,
                            h5_dspace_id,
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);
    // write data
    status = H5Dwrite(h5_dset_id,
                      h5_dtype_id,
                      H5S_ALL,
                      H5S_ALL,
                      H5P_DEFAULT,
                      mystring4_char_ptr);

    H5Tclose(h5_dtype_id);
    H5Sclose(h5_dspace_id);
    H5Dclose(h5_dset_id);


    // case 5: string written using variable length

    h5_dtype_id  = H5Tcreate(H5T_STRING, H5T_VARIABLE);
    h5_dspace_id = H5Screate(H5S_SCALAR);



    const char *mystr_char_ptr = my_string.c_str();

    // create new dataset

    h5_dset_id  = H5Dcreate(h5_file_id,
                            "case_5",
                            h5_dtype_id,
                            h5_dspace_id,
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);
    // write data
    status = H5Dwrite(h5_dset_id,
                      h5_dtype_id,
                      H5S_ALL,
                      H5S_ALL,
                      H5P_DEFAULT,
                      &mystr_char_ptr);
    //
    H5Tclose(h5_dtype_id);
    H5Sclose(h5_dspace_id);
    H5Dclose(h5_dset_id);

    H5Fclose(h5_file_id);

    // load back in and make sure we get the correct string for each case
    Node n_load;
    io::load(tout,n_load);
    n_load.print();

    EXPECT_EQ(n_load["case_0"].as_string(), my_string );
    EXPECT_EQ(n_load["case_1"].as_string(), my_string );
    EXPECT_EQ(n_load["case_2"].as_string(), my_string );
    EXPECT_EQ(n_load["case_3"].as_string(), my_string3 );
    EXPECT_EQ(n_load["case_4"].as_string(), my_string );
    EXPECT_EQ(n_load["case_5"].as_string(), my_string );

}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_write_read_string_compress)
{
    uint32 s_len = 10000;
    std::string tout_std = "tout_hdf5_wr_string_no_compression.hdf5";
    std::string tout_cmp = "tout_hdf5_wr_string_with_compression.hdf5";

    std::string s_val = std::string(s_len, 'z');

    Node n;
    n["my_string"] = s_val;

    Node opts;
    opts["hdf5/chunking/threshold"]  = 100;
    opts["hdf5/chunking/chunk_size"] = 100;

    // write out the string w and w/o compression
    io::save(n,tout_std, "hdf5");
    io::save(n,tout_cmp, "hdf5", opts);

    Node n_out;
    io::hdf5_read(tout_cmp,n_out);
    EXPECT_EQ(n_out["my_string"].as_string(), s_val);


    int64 tout_std_fs = utils::file_size(tout_std);
    int64 tout_cmp_fs = utils::file_size(tout_cmp);
    CONDUIT_INFO("fs test: std = "
                 << tout_std_fs
                 << ", cmp ="
                 << tout_cmp_fs);
    EXPECT_TRUE(tout_cmp_fs < tout_std_fs);

}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_list)
{
    std::string tout_std = "tout_hdf5_list.hdf5";

    Node n;
    n.append() = "42";
    n.append() = "42";
    n.append() = "42";
    n.append() = "42";

    Node &n_sub = n.append();
    n_sub.append() = 42;
    n_sub.append() = 42;
    n_sub.append() = 42;
    n_sub.append() = 42;
    n_sub.append() = 42.0;

    io::save(n,tout_std, "hdf5");

    Node n_load, info;
    io::load(tout_std,"hdf5",n_load);

    n_load.print();

    EXPECT_FALSE(n.diff(n_load,info));

    hid_t h5_file_id = io::hdf5_open_file_for_read(tout_std);
    /// check subpath of written list
    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"0"));
    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"4"));
    EXPECT_FALSE(io::hdf5_has_path(h5_file_id,"5"));

    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"4/0"));
    EXPECT_TRUE(io::hdf5_has_path(h5_file_id,"4/4"));
    EXPECT_FALSE(io::hdf5_has_path(h5_file_id,"4/5"));

    n_load.reset();
    io::hdf5_read(h5_file_id,"4/4",n_load);
    EXPECT_EQ(n_load.to_float64(),42.0);

    io::hdf5_close_file(h5_file_id);

    // simple compat check (could be expanded)
    Node n_check;
    n_check = 42.0;
    // this isn't compat with the existing file

    h5_file_id = io::hdf5_open_file_for_read_write(tout_std);
    EXPECT_THROW(io::hdf5_write(n_check,h5_file_id),Error);

    // orig should be compat
    n_check.set(n);
    io::hdf5_write(n_check,h5_file_id);
    // lets change the value of one of the list entries
    n_check[4][4] = 3.1415;
    io::hdf5_write(n_check,h5_file_id);
    io::hdf5_close_file(h5_file_id);

    io::load(tout_std,"hdf5",n_load);

    n_load.print();
    EXPECT_FALSE(n_check.diff(n_load,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_list_with_offset)
{
    std::string tout_std = "tout_hdf5_list_with_offset.hdf5";

    Node n, opts;
    n.append() = DataType::c_short(1);
    short_array vals = n[0].value();
    vals[0] = 1;
    io::save(n,tout_std, "hdf5");

    vals[0] = 2;
    opts["offset"] = 1;
    io::save(n,tout_std, "hdf5",opts);

    Node n_load, info;
    io::load(tout_std,"hdf5",n_load);

    // check values of data
    // since we didn't use save_merged, the first value should be overwritten
    short_array read_vals = n_load[0].value();
    EXPECT_EQ(0,read_vals[0]);
    EXPECT_EQ(2,read_vals[1]);

    // let's try again
    vals[0] = 1;
    io::save(n,tout_std, "hdf5");

    vals[0] = 2;
    opts["offset"] = 1;
    io::save_merged(n,tout_std, "hdf5",opts);

    io::load(tout_std,"hdf5",n_load);

    read_vals = n_load[0].value();
    EXPECT_EQ(1,read_vals[0]);
    EXPECT_EQ(2,read_vals[1]);

    vals[0] = 3;
    opts["offset"] = 2;
    io::save_merged(n,tout_std, "hdf5",opts);

    vals[0] = 4;
    opts["offset"] = 3;
    io::save_merged(n,tout_std, "hdf5",opts);

    vals[0] = 5;
    opts["offset"] = 4;
    io::save_merged(n,tout_std, "hdf5",opts);

    vals[0] = 6;
    opts["offset"] = 5;
    io::save_merged(n,tout_std, "hdf5",opts);

    io::load_merged(tout_std,"hdf5",n_load);

    read_vals = n_load[0].value();
    EXPECT_EQ(1,read_vals[0]);
    EXPECT_EQ(2,read_vals[1]);
    EXPECT_EQ(3,read_vals[2]);
    EXPECT_EQ(4,read_vals[3]);
    EXPECT_EQ(5,read_vals[4]);
    EXPECT_EQ(6,read_vals[5]);

    // try loading with offset and size
    Node opts_read;
    opts_read["offset"] = 2;
    opts_read["size"] = 2;
    io::load(tout_std,"hdf5",opts_read,n_load);

    read_vals = n_load[0].value();
    EXPECT_EQ(3,read_vals[0]);
    EXPECT_EQ(4,read_vals[1]);
    EXPECT_EQ(2, read_vals.number_of_elements());

}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_compat_with_empty)
{
    std::string tout_std = "tout_hdf5_empty_compat.hdf5";

    Node n;
    n["myval"] = 42;
    io::save(n,tout_std);
    n["empty"] = DataType::empty();
    n.print();
    io::save_merged(n,tout_std);
    // used to fail due to bad compat check
    io::save_merged(n,tout_std);

    Node n_load, n_diff_info;
    io::load(tout_std,"hdf5",n_load);
    n_load.print();

    EXPECT_FALSE(n.diff(n_load,n_diff_info));
}
