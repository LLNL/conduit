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
#include <hdf5.h>
#include "H5Zzfp_lib.h"
#include "H5Zzfp_props.h"
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

// void write_ndarray(hid_t hdf5_id,
//                    const std::string &path,
//                    const Node &values,
//                    const Node &opts)
// {
//     // in memory stuff:?
//
//     // values == 1d array, handle stride, offsets, etc
//     // during write, or use compact
//
//
//     // in file stuff:
//     // shape
//     // chunks
//     // offsets
//     // strides
// }
// // options
//
//
// void create_ndarray_leaf(hid_t file_id,
//                          const std::string &path,
//                          const Node &ndarray_info)
// {
//     int64_array shape  = ndarray_info["shape"];
//     int64_array chunks = ndarray_info["chunks"];
//
// }
//
// void write_leaf_ndarray(hid_t file_id,
//                         const std::string &path,
//                         const Node &value,
//                         const Node &dest_hyperslab)
// {
//     /* identifiers  */
//     hid_t     file_id;
//     hid_t     dataset_cprops_id;
//     hid_t     dataset_id;
//     hid_t     dataspace_id;
//     hsize_t   dims[2];
//     hsize_t   chunks[2];
//     herr_t    status;
//
//
//     int64_array shape  = dest_hyperslab["shape"];
//     int64_array chunks = dest_hyperslab["chunks"];
//
//
//     if(dest_hyperslab.has_child("offsets"))
//     {
//         /// DO OFFSETS IN FILE MSPACE
//     }
//
//     hsize_t ndims = shape.dtype().number_of_elements();
//
//     /* Create the data space for the dataset. */
//     dataspace_id = H5Screate_simple(ndims,
//                                     shape.data_ptr(),
//                                     NULL);
//
//     EXPECT_TRUE (dataspace_id >= 0 );
//
//     /* setup dataset creation properties */
//     dataset_cprops_id = H5Pcreate(H5P_DATASET_CREATE);
//     EXPECT_TRUE(dataset_cprops_id >= 0);
//
//     status = H5Pset_chunk(dataset_cprops_id,
//                           ndims,
//                           chunks.data_ptr());
//
//     EXPECT_TRUE(status >= 0);
//
//     /* When the h5z-zfp filter is used as a library, we need to init it */
//     H5Z_zfp_initialize();
//
//     /* use zfp rate mode */
//     H5Pset_zfp_rate(dataset_cprops_id, 4);
//
//     /* Create the dataset. */
//     dataset_id = H5Dcreate2(file_id,
//                             "/compressed_dset",
//                             H5T_NATIVE_DOUBLE,
//                             dataspace_id,
//                             H5P_DEFAULT,
//                             dataset_cprops_id,
//                             H5P_DEFAULT);
//
//     EXPECT_TRUE (dataset_id >= 0);
//
//     /* write our data */
//     status = H5Dwrite(dataset_id,
//                       H5T_NATIVE_DOUBLE,
//                       H5S_ALL,
//                       H5S_ALL,
//                       H5P_DEFAULT,
//                       vals);
//
//     EXPECT_TRUE(status >= 0 );
//
//
//
//
//
//     // dest hyper slab has:
//
//     // shape
//
// // optional:
//     // strides
//     // offsets
// }
//
// void write_leaf_ndarray(const std::string &path,
//                         const Node &ndarray)
// {
//     // shape
//
// // optional:
//     // strides
//     // offsets
// }
//
// void read_leaf_ndarray_details(const std::string &path,
//                                Node &res)
// {
//     // shape
//
// // optional:
//     // strides
//     // offsets
// }
//
// void read_leaf_ndarray(hid_t file_id,
//                        const std::string &path,
//                        const Node &hyper_slab,
//                        Node &res)
// {
//     // shape
//
// // optional:
//     // strides
//     // offsets
// }
//
// TEST(conduit_relay_io_hdf5_h5zzfp, basic_use)
// {
//     /* identifiers  */
//     hid_t     file_id;
//     hid_t     dataset_cprops_id;
//     hid_t     dataset_id;
//     hid_t     dataspace_id;
//     hsize_t   dims[2];
//     hsize_t   chunks[2];
//     herr_t    status;
//
//
//     chunks[0] = 256;
//     chunks[1] = 8;
//
//     // example 2d array
//     dims[0] = chunks[0] * 2;
//     dims[1] = chunks[1] * 3;
//
//     hsize_t ntotal = dims[0] * dims[1];
//
//     double * vals = new double[ntotal];
//
//     for (hsize_t i = 0; i < ntotal; i++)
//     {
//         vals[i] = i;
//     }
//
//     /* Create a new file using default properties. */
//     file_id = H5Fcreate("h5zzfp_smoke_test.hdf5",
//                         H5F_ACC_TRUNC,
//                         H5P_DEFAULT,
//                         H5P_DEFAULT);
//
//     EXPECT_TRUE (file_id >= 0 );
//
//     /* Create the data space for the dataset. */
//     dataspace_id = H5Screate_simple(2, dims, NULL);
//     EXPECT_TRUE (dataspace_id >= 0 );
//
//     /* setup dataset creation properties */
//     dataset_cprops_id = H5Pcreate(H5P_DATASET_CREATE);
//     EXPECT_TRUE(dataset_cprops_id >= 0);
//
//     status = H5Pset_chunk(dataset_cprops_id, 2, chunks);
//     EXPECT_TRUE(status >= 0);
//
//     /* When the h5z-zfp filter is used as a library, we need to init it */
//     H5Z_zfp_initialize();
//
//     /* use zfp rate mode */
//     H5Pset_zfp_rate(dataset_cprops_id, 4);
//
//     /* Create the dataset. */
//     dataset_id = H5Dcreate2(file_id,
//                             "/compressed_dset",
//                             H5T_NATIVE_DOUBLE,
//                             dataspace_id,
//                             H5P_DEFAULT,
//                             dataset_cprops_id,
//                             H5P_DEFAULT);
//
//     EXPECT_TRUE (dataset_id >= 0);
//
//     /* write our data */
//     status = H5Dwrite(dataset_id,
//                       H5T_NATIVE_DOUBLE,
//                       H5S_ALL,
//                       H5S_ALL,
//                       H5P_DEFAULT,
//                       vals);
//
//     EXPECT_TRUE(status >= 0 );
//
//     /* End access to the prop list and release resources used by it. */
//     status = H5Pclose(dataset_cprops_id);
//     EXPECT_TRUE(status >= 0 );
//
//     /* End access to the dataset and release resources used by it. */
//     status = H5Dclose(dataset_id);
//     EXPECT_TRUE(status >= 0 );
//
//     /* Terminate access to the data space. */
//     status = H5Sclose(dataspace_id);
//     EXPECT_TRUE(status >= 0 );
//
//     /* Close the file. */
//     status = H5Fclose(file_id);
//     EXPECT_TRUE(status >= 0 );
//
//     /* cleanup our vals array */
//     delete [] vals;
//
//     /* cleanup h5z-zfp */
//     H5Z_zfp_finalize();
// }
//

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5_h5zzfp, conduit_h5zzfp_write_synth)
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


