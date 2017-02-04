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
/// file: t_relay_io_hdf5_slab.cpp
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
// This example tests reads of slabs from a hdf5 dataset.
// 
// we may provide something like this in in the relay hdf5 interface
// in the future. 
//-----------------------------------------------------------------------------
bool
hdf5_read_dset_slab(const std::string &file_path,
                    const std::string &fetch_path,
                    const DataType &dtype,
                    void *data_ptr)
{
    // assume fetch_path points to a hdf5 dataset
    // open the hdf5 file for reading
    hid_t h5_file_id = H5Fopen(file_path.c_str(),
                               H5F_ACC_RDONLY,
                               H5P_DEFAULT);
    CONDUIT_CHECK_HDF5_ERROR(h5_file_id,
                     "Error opening HDF5 file for reading: "  << file_path);

    // open the dataset
    hid_t h5_dset_id = H5Dopen( h5_file_id, fetch_path.c_str(),H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR(h5_dset_id,
                     "Error opening HDF5 dataset at: " << fetch_path);


    // get info about the dataset
    hid_t h5_dspace_id = H5Dget_space(h5_dset_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_dspace_id,
                     "Error reading HDF5 Dataspace: " << h5_dset_id);

    // check for empty case
    if(H5Sget_simple_extent_type(h5_dspace_id) == H5S_NULL)
    {
        // we have an error, but to try to clean up the hdf5 handles
        // before we issue the error.
        
        CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                          "Error closing HDF5 data space: " << file_path);

        CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                          "Error closing HDF5 dataset: " << file_path);
        // close the hdf5 file
        CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                         "Error closing HDF5 file: " << file_path);
                      
        CONDUIT_ERROR("Can't slab fetch from an empty hdf5 data set.");
    }

    hid_t h5_dtype_id  = H5Dget_type(h5_dset_id);

    CONDUIT_CHECK_HDF5_ERROR(h5_dtype_id,
                     "Error reading HDF5 Datatype: "
                     << h5_dset_id);

    // TODO: bounds check  (check that we are fetching a subset of the elems)
    index_t  h5_nelems = H5Sget_simple_extent_npoints(h5_dspace_id);
    if( dtype.number_of_elements() > h5_nelems)
    {
        // we have an error, but to try to clean up the hdf5 handles
        // before we issue the error.
        
        CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                          "Error closing HDF5 data space: " << file_path);

        CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                          "Error closing HDF5 dataset: " << file_path);
        // close the hdf5 file
        CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                         "Error closing HDF5 file: " << file_path);
                      
        CONDUIT_ERROR("Can't slab fetch a buffer larger than the source"
                        " hdf5 data set. Requested number of elements" 
                        << dtype.number_of_elements()
                        << " hdf5 dataset number of elements" << h5_nelems);
    }


    // we need to compute an offset, stride, and element bytes
    // that will work for reading in the general case
    // right now we assume the dest type of data and the hdf5 datasets
    // data type are compatible  
    
    // conduit's offsets, strides, are all in terms of bytes
    // hdf5's are in terms of elements
    
    // what we really want is a way to read bytes from the hdf5 dset with
    // out any type conversion, but that doesn't exist.

    // general support would include reading a a view of one type that
    //  points to a buffer of another
    // (for example a view of doubles that is defined on a buffer of bytes)

    // but hdf5 doens't support slab fetch across datatypes
    // so for now we make sure the datatype is consistent. 

    DataType h5_dt = conduit::relay::io::hdf5_dtype_to_conduit_dtype(h5_dtype_id,1);

    if( h5_dt.id() != dtype.id() )
    {
        CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                          "Error closing HDF5 data space: " << file_path);

        CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                          "Error closing HDF5 dataset: " << file_path);
        // close the hdf5 file
        CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                         "Error closing HDF5 file: " << file_path);

        CONDUIT_INFO("Cannot fetch hdf5 slab of buffer and view are"
                       "different data types.")
        return false;
    }



    hid_t h5_status    = 0;

    hsize_t elem_bytes = dtype.element_bytes();
    hsize_t offset  = dtype.offset() / elem_bytes; // in bytes, convert to elems
    hsize_t stride  = dtype.stride() / elem_bytes; // in bytes, convert to elems
    hsize_t num_ele = dtype.number_of_elements();
    
    CONDUIT_INFO("slab dtype: " << dtype.to_json());
    
    CONDUIT_INFO("hdf5 slab: "  <<
                   " element_offset: " << offset <<
                   " element_stride: " << stride <<
                   " number_of_elements: " << num_ele);
    
    h5_status = H5Sselect_hyperslab(h5_dspace_id,
                                    H5S_SELECT_SET,
                                    &offset,
                                    &stride,
                                    &num_ele,
                                    0); // 0 here means NULL pointers; HDF5 *knows* dimension is 1
    // check subset sel
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                      "Error selecting hyper slab from HDF5 dataspace: " << h5_dspace_id);


    hid_t h5_dspace_compact_id = H5Screate_simple(1,
                                                  &num_ele,
                                                  NULL);

    CONDUIT_CHECK_HDF5_ERROR(h5_dspace_id,
                             "Failed to create HDF5 data space (memory dspace)");

    h5_status = H5Dread(h5_dset_id, // data set id
                        h5_dtype_id, // memory type id  // use same data type?
                        h5_dspace_compact_id,  // memory space id ...
                        h5_dspace_id, // file space id
                        H5P_DEFAULT,
                        data_ptr);
    // check read
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                      "Error reading bytes from HDF5 dataset: " << h5_dset_id);

    // close the data space 
    CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                      "Error closing HDF5 data space: " << file_path);

    // close the compact data space 
    CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_compact_id),
                      "Error closing HDF5 data space (memory dspace)" << file_path);


    // close the dataset
    CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                      "Error closing HDF5 dataset: " << file_path);

    // close the hdf5 file
    CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                      "Error closing HDF5 file: " << file_path);

    return true;
}


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5, hdf5_dset_slab_read_test)
{

    // create a simple buffer of doubles
    Node n;
    
    n["full_data"].set(DataType::c_double(20));
    
    double *vin = n["full_data"].value();
    
    for(int i=0;i<20;i++)
    {
        vin[i] = i;
    }
    
    CONDUIT_INFO("Example Full Data")
    
    n.print();

    io::hdf5_write(n,"tout_hdf5_slab.hdf5");
    
    Node nload;
    nload.set(DataType::c_double(10));
    
    double *vload = nload.value();
    
    // stride to read every other entry into compact storage 
    hdf5_read_dset_slab("tout_hdf5_slab.hdf5",
                        "full_data",
                        DataType::c_double(10,
                                           sizeof(double),   // offset 1 double
                                           sizeof(double)*2, //stride 2 doubles
                                           sizeof(double)),
                                           vload);
    CONDUIT_INFO("Load Result");
    nload.print();
    
    for(int i=0;i<10;i++)
    {
        EXPECT_NEAR(vload[i],1.0 + i * 2.0,1e-3);
    }
}



