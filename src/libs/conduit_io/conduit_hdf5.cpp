//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://llnl.github.io/conduit/.
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
/// file: conduit_hdf5.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_hdf5.hpp"

//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------
#include <iostream>

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <hdf5.h>


//-----------------------------------------------------------------------------
//
/// The CONDUIT_CHECK_HDF5_ERROR macro is used to check error codes from HDF5.
//-----------------------------------------------------------------------------
#define CONDUIT_CHECK_HDF5_ERROR( hdf5_err, msg    )                \
{                                                                   \
    if( hdf5_err < 0 )                                              \
    {                                                               \
        std::ostringstream hdf5_err_oss;                            \
        hdf5_err_oss << "HDF5 Error code"                           \
            <<  hdf5_err                                            \
            << " " << msg;                                          \
        CONDUIT_ERROR( hdf5_err_oss.str());                         \
    }                                                               \
}                                                                   \


//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::io --
//-----------------------------------------------------------------------------
namespace io
{

//-----------------------------------------------------------------------------
// Helper methods that aren't part of public conduit::io
//  conduit_dtype_to_hdf5_dtype
//  hdf5_write_leaf
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
hid_t
conduit_dtype_to_hdf5_dtype(const DataType &dt)
{
    hid_t res = -1;
    // first check endianness
    if(dt.is_little_endian()) // we know we are little endian
    {
        switch(dt.id())
        {
            case DataType::INT8_ID:  res = H5T_STD_I8LE;
            case DataType::INT16_ID: res = H5T_STD_I16LE;
            case DataType::INT32_ID: res = H5T_STD_I32LE;
            case DataType::INT64_ID: res = H5T_STD_I64LE;
            
            case DataType::UINT8_ID:  res = H5T_STD_U8LE;
            case DataType::UINT16_ID: res = H5T_STD_U16LE;
            case DataType::UINT32_ID: res = H5T_STD_U32LE;
            case DataType::UINT64_ID: res = H5T_STD_U64LE;
            
            case DataType::FLOAT32_ID: res = H5T_IEEE_F32LE;
            case DataType::FLOAT64_ID: res = H5T_IEEE_F64LE;
            
            case DataType::CHAR8_STR_ID: res = H5T_C_S1;
            
            default:
                CONDUIT_ERROR("conduit::DataType to HDF5 Leaf DataType Conversion:"
                              << dt.to_json()
                              <<" is not a leaf data type");
        };
    }
    else // we know we are big endian
    {
        switch(dt.id())
        {
            case DataType::INT8_ID:  res = H5T_STD_I8BE;
            case DataType::INT16_ID: res = H5T_STD_I16BE;
            case DataType::INT32_ID: res = H5T_STD_I32BE;
            case DataType::INT64_ID: res = H5T_STD_I64BE;
            
            case DataType::UINT8_ID:  res = H5T_STD_U8BE;
            case DataType::UINT16_ID: res = H5T_STD_U16BE;
            case DataType::UINT32_ID: res = H5T_STD_U32BE;
            case DataType::UINT64_ID: res = H5T_STD_U64BE;
            
            case DataType::FLOAT32_ID: res = H5T_IEEE_F32BE;
            case DataType::FLOAT64_ID: res = H5T_IEEE_F64BE;
            
            case DataType::CHAR8_STR_ID: res = H5T_C_S1;
            
            default:
                CONDUIT_ERROR("conduit::DataType to HDF5 Leaf DataType Conversion:"
                              << dt.to_json()
                              <<" is not a leaf data type");
        };
    }
    
    return res;
}

//---------------------------------------------------------------------------//
void hdf5_write_leaf(const Node &node,
                     hid_t hdf5_id,
                     const std::string &hdf5_path)
{
    DataType dt = node.dtype();

    hid_t h5_dtype_id = conduit_dtype_to_hdf5_dtype(dt);

    herr_t h5_status;
    
    // create a data space to describe our data
    hsize_t num_eles  = dt.number_of_elements();
    hid_t   h5_dspace_id = H5Screate_simple(1,
                                            &num_eles,
                                            NULL);
    // TODO: Error check here?

    // create dataset to write to
    hid_t   h5_dset_id = H5Dcreate2(hdf5_id,
                                    hdf5_path.c_str(),
                                    h5_dtype_id,
                                    h5_dspace_id,
                                    H5P_DEFAULT,
                                    H5P_DEFAULT,
                                    H5P_DEFAULT);
    // TODO: Error check here?

    // if compact, we can write directly
    if(dt.is_compact()) 
    {
        // write data
        h5_status = H5Dwrite(h5_dset_id,
                             h5_dtype_id,
                             H5S_ALL,
                             H5S_ALL,
                             H5P_DEFAULT,
                             node.data_ptr());

        CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Failed to write to HDF5 Dataset " << h5_dset_id);
    }
    else 
    {
        // otherwise, we need to compact our data first
        Node n;
        node.compact_to(n);
        h5_status = H5Dwrite(h5_dset_id,
                             h5_dtype_id,
                             H5S_ALL,
                             H5S_ALL,
                             H5P_DEFAULT,
                             n.data_ptr());

        CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Failed to write to HDF5 Dataset " << h5_dset_id);
    }

    // close our dataset
    h5_status= H5Dclose(h5_dset_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Failed to close HDF5 Dataset " << h5_dset_id);

    // close our dataspace
    h5_status = H5Sclose(h5_dspace_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Failed to close HDF5 Data Space " << h5_dspace_id);
}


//---------------------------------------------------------------------------//
void 
hdf5_write(const  Node &node,
           const std::string &path)
{
    // check for ":" split
    std::string file_path;
    std::string hdf5_path;
    conduit::utils::split_string(path,
                                 std::string(":"),
                                 file_path,
                                 hdf5_path);

    /// If hdf5_path is empty, we have a problem ... 
    if(hdf5_path.size() == 0)
    {
        CONDUIT_ERROR("Invalid path for HDF5 write: " << path);
    }

    hdf5_write(node,
               file_path,
               hdf5_path);
}

//---------------------------------------------------------------------------//
void
hdf5_read(const std::string &path,
          Node &node)
{
    // check for ":" split    
    std::string file_path;
    std::string hdf5_path;
    conduit::utils::split_string(path,
                                 std::string(":"),
                                 file_path,
                                 hdf5_path);

    /// If hdf5_path is empty, we have a problem ... 
    if(hdf5_path.size() == 0)
    {
        CONDUIT_ERROR("Invalid path for HDF5 read: " << path);
    }

    hdf5_read(file_path,
              hdf5_path,
              node);
}


//---------------------------------------------------------------------------//
void
hdf5_write(const Node &node,
           const std::string &file_path,
           const std::string &hdf5_path)
{
    herr_t h5_status; // TODO, what to init to?

    // open the hdf5 file for writing
    hid_t h5_file_id = H5Fcreate(file_path.c_str(),
                                 H5F_ACC_TRUNC,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);
    
    if( h5_file_id >=0 )
    {  
        hdf5_write(node,
                   h5_file_id,
                   hdf5_path);
    }
    else
    {
        CONDUIT_ERROR("Error opening HDF5 file for writing: " << file_path);
    }
    
    // close the hdf5 file
    h5_status = H5Fclose(h5_file_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Error closing HDF5 file: " << file_path);
}

//---------------------------------------------------------------------------//
void
hdf5_read(const std::string &file_path,
          const std::string &hdf5_path,
          Node &node)
{
    herr_t h5_status; // TODO, what to init to?
    
    // open the hdf5 file for reading
    hid_t h5_file_id = H5Fopen(file_path.c_str(),
                               H5F_ACC_RDONLY,
                               H5P_DEFAULT);

    if( h5_file_id >=0 )
    {  
        hdf5_read(h5_file_id,
                  hdf5_path,
                  node);
    }
    else
    {
        CONDUIT_ERROR("Error opening HDF5 file for reading: " << file_path);
    }
    
    // close the hdf5 file
    h5_status = H5Fclose(h5_file_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Error closing HDF5 file: " << file_path);
}


//---------------------------------------------------------------------------//
void
hdf5_write(const Node &node,
           hid_t hdf5_id,
           const std::string &hdf5_path)
{
    DataType dt = node.dtype();
    if(dt.is_object())
    {
        // strong dose of evil casting, but it's ok b/c we are grownups here?
        // time we will tell ...
        NodeIterator itr = const_cast<Node*>(&node)->children();

        // call on each child with expanded path
        while(itr.has_next())
        {
            Node &child = itr.next();
            std::string chld_pth = hdf5_path + "/" + itr.path();
            hdf5_write(child,hdf5_id,chld_pth);
        }
    }
    else if(dt.is_number() || dt.is_string())
    {
        hdf5_write_leaf(node,hdf5_id,hdf5_path);
    }
    else
    {
        CONDUIT_ERROR("HDF5 write doesn't support LIST_ID or EMPTY_ID nodes.");
    }
}




//---------------------------------------------------------------------------//
void
hdf5_read(hid_t hdf5_id,
          const std::string &hdf5_path,
          Node &node)
{
    /// TODO
}


};
//-----------------------------------------------------------------------------
// -- end conduit::io --
//-----------------------------------------------------------------------------


};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
