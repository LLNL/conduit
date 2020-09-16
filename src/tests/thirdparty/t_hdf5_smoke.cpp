// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: hdf5_smoke.cpp
///
//-----------------------------------------------------------------------------

#include <hdf5.h>
#include <iostream>
#include "gtest/gtest.h"

// adapted from hdf5 tutorial: h5_crtdata.c

//-----------------------------------------------------------------------------
TEST(hdf5_smoke, basic_use)
{
    /* identifiers  */
    hid_t     file_id;
    hid_t     dataset_id;
    hid_t     dataspace_id;  
    hsize_t   dims[2];
    herr_t    status;

    /* Create a new file using default properties. */
    file_id = H5Fcreate("hdf5_smoke_test.hdf5",
                        H5F_ACC_TRUNC,
                        H5P_DEFAULT, H5P_DEFAULT);

    /* Create the data space for the dataset. */
    dims[0] = 4; 
    dims[1] = 6; 
    dataspace_id = H5Screate_simple(2, dims, NULL);

    /* Create the dataset. */
    dataset_id = H5Dcreate2(file_id,
                            "/dset",
                            H5T_STD_I32BE,
                            dataspace_id, 
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);

    /* End access to the dataset and release resources used by it. */
    status = H5Dclose(dataset_id);
    EXPECT_TRUE(status >= 0 );

    /* Terminate access to the data space. */ 
    status = H5Sclose(dataspace_id);
    EXPECT_TRUE(status >= 0 );

    /* Close the file. */
    status = H5Fclose(file_id);
    EXPECT_TRUE(status >= 0 );
}
