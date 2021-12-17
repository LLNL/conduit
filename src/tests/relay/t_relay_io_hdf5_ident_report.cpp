// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_relay_io_hdf5_inden_report.cpp
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

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_inden_report_basic)
{
    Node info;
    io::hdf5_identifier_report(info);
    EXPECT_TRUE(info.dtype().is_empty());
}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, conduit_hdf5_inden_report_open)
{
    std::string test_file_name = "tout_hdf5_inden_report_test.hdf5";
    utils::remove_path_if_exists(test_file_name);

    Node info;
    io::hdf5_identifier_report(info);
    EXPECT_TRUE(info.dtype().is_empty());

    //
    // create a file
    //

    hid_t h5_file_id = io::hdf5_create_file(test_file_name);
    io::hdf5_identifier_report(info);
    CONDUIT_INFO("Post File Create");
    info.print();
    EXPECT_EQ(info.number_of_children(),1);
    EXPECT_EQ(info[0]["type"].as_string(),"file");

    //
    // create a group
    //
    hid_t h5_group_id = H5Gcreate(h5_file_id,
                                  "mygroup",
                                  H5P_DEFAULT,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT);
    io::hdf5_identifier_report(info);
    CONDUIT_INFO("Post Group Create");
    info.print();
    EXPECT_EQ(info.number_of_children(),2);
    EXPECT_EQ(info[1]["type"].as_string(),"group");

    //
    // create a dataset (and space + param)
    //
    hid_t h5_dtype = H5T_NATIVE_SHORT;

    hsize_t num_eles = 2;

    hid_t   h5_dspace_id = H5Screate_simple(1,
                                            &num_eles,
                                            NULL);

    // create new dataset
    hid_t h5_dset_id  = H5Dcreate(h5_group_id,
                                  "mydata",
                                  h5_dtype,
                                  h5_dspace_id,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT);

    // test using file id as well
    io::hdf5_identifier_report(info);
    CONDUIT_INFO("Post Dataset Create");
    info.print();

    Node info_b, diff_info;
    io::hdf5_identifier_report(h5_file_id,info_b);
    CONDUIT_INFO("Check vs report using hid");
    info_b.print();
    EXPECT_FALSE(info.diff(info_b,diff_info));

    EXPECT_EQ(info.number_of_children(),3);

    // cleanup for dataset creation
    H5Sclose(h5_dspace_id);
    H5Dclose(h5_dset_id);
    io::hdf5_identifier_report(info);
    CONDUIT_INFO("Post Dataset close");
    info.print();
    EXPECT_EQ(info.number_of_children(),2);

    // close our group
    H5Gclose(h5_group_id);
    io::hdf5_identifier_report(info);
    CONDUIT_INFO("Post Group close");
    info.print();
    EXPECT_EQ(info.number_of_children(),1);

    // close our file
    io::hdf5_close_file(h5_file_id);
    io::hdf5_identifier_report(info);
    CONDUIT_INFO("Post File close");
    info.print();
    EXPECT_TRUE(info.dtype().is_empty());
}



