//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
/// file: relay_hdf5.cpp
///
//-----------------------------------------------------------------------------

#include "relay_hdf5.hpp"

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

#define HDF5_STATUS_OK( hdf5_err )  hdf5_err >= 0


//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay --
//-----------------------------------------------------------------------------
namespace relay
{


//-----------------------------------------------------------------------------
// -- begin conduit::relay::io --
//-----------------------------------------------------------------------------
namespace io
{

//-----------------------------------------------------------------------------
// private class used to suppress HDF5 error messages.
//-----------------------------------------------------------------------------
class HDF5ErrorStackSupressor
{
public:
        HDF5ErrorStackSupressor()
        {
            disable_hdf5_error_func();
        }
        
       ~HDF5ErrorStackSupressor()
        {
            restore_hdf5_error_func();
        }

private:
    void disable_hdf5_error_func()
    {
            H5Eget_auto(H5E_DEFAULT,
                        &herr_func, 
                        &herr_func_client_data);
            
            H5Eset_auto(H5E_DEFAULT,
                        NULL,
                        NULL);
    }
        
    void restore_hdf5_error_func()
    {
        H5Eset_auto(H5E_DEFAULT,
                    herr_func,
                    herr_func_client_data);
    }

    H5E_auto2_t  herr_func;
    void         *herr_func_client_data; 
};


// //-----------------------------------------------------------------------------
// // private struct to hold HDF5 error state, so we can easily toggle display
// // of error messages.
// //-----------------------------------------------------------------------------
// typedef struct hdf5_error_handler_state
// {
//     H5E_auto2_t  herr_func;
//     void         *herr_func_client_data;
// } hdf5_error_state;
//
//
// //-----------------------------------------------------------------------------
// // helpers methods for that allow us to easily toggle display of
// // HDF5 error messages.
// //-----------------------------------------------------------------------------
//
// //-----------------------------------------------------------------------------
// void
// hdf5_get_error_handler(hdf5_error_handler_state &herr_state)
// {
//     H5Eget_auto(H5E_DEFAULT,
//                 &herr_state.herr_func,
//                 &herr_state.herr_func_client_data);
// }
//
// //-----------------------------------------------------------------------------
// void
// hdf5_disable_error_handler()
// {
//
// }
//
// //-----------------------------------------------------------------------------
// void
// hdf5_set_error_handler(hdf5_error_handler_state &herr_state)
// {
//     H5Eset_auto(H5E_DEFAULT,
//                 herr_state.herr_func,
//                 herr_state.herr_func_client_data);
// }

//-----------------------------------------------------------------------------
// helper method decls
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// helpers for data type conversions
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
hid_t    conduit_dtype_to_hdf5_dtype(const DataType &dt);

//-----------------------------------------------------------------------------
DataType hdf5_dtype_to_conduit_dtype(hid_t hdf5_dtype_id,
                                     index_t num_elems);

//-----------------------------------------------------------------------------
// helpers for checking if compatible 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool  check_if_conduit_leaf_is_compatible_with_hdf5_obj(const DataType &dtype,
                                                        hid_t hdf5_id);
 
//-----------------------------------------------------------------------------
bool  check_if_conduit_object_is_compatible_with_hdf5_tree(const Node &node,
                                                           hid_t hdf5_id);

//-----------------------------------------------------------------------------
bool  check_if_conduit_node_is_compatible_with_hdf5_tree(const Node &node,
                                                         hid_t hdf5_id);


//-----------------------------------------------------------------------------
// helpers for writing
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
hid_t create_hdf5_dataset_for_conduit_leaf(const DataType &dt,
                                           hid_t hdf5_group_id,
                                           const std::string &hdf5_dset_name);

//---------------------------------------------------------------------------//
void  write_conduit_leaf_to_hdf5_dataset(const Node &node,
                                         hid_t hdf5_dset_id);

//-----------------------------------------------------------------------------
void  write_conduit_leaf_to_hdf5_group(const Node &node,
                                       hid_t hdf5_group_id,
                                       const std::string &hdf5_dset_name);

//-----------------------------------------------------------------------------
void  write_conduit_object_to_hdf5_group(const Node &node,
                                         hid_t hdf5_group_id);


//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Data Type Helper methods that aren't part of public conduit::relay::io
//
//  conduit_dtype_to_hdf5_dtype
//  hdf5_dtype_to_conduit_dtype
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
            case DataType::INT8_ID:  res = H5T_STD_I8LE;  break;
            case DataType::INT16_ID: res = H5T_STD_I16LE; break;
            case DataType::INT32_ID: res = H5T_STD_I32LE; break;
            case DataType::INT64_ID: res = H5T_STD_I64LE; break;
            
            case DataType::UINT8_ID:  res = H5T_STD_U8LE;  break;
            case DataType::UINT16_ID: res = H5T_STD_U16LE; break;
            case DataType::UINT32_ID: res = H5T_STD_U32LE; break;
            case DataType::UINT64_ID: res = H5T_STD_U64LE; break;
            
            case DataType::FLOAT32_ID: res = H5T_IEEE_F32LE; break;
            case DataType::FLOAT64_ID: res = H5T_IEEE_F64LE; break;
            
            case DataType::CHAR8_STR_ID: res = H5T_C_S1; break;
            
            default:
                CONDUIT_ERROR("conduit::DataType to HDF5 Leaf DataType Conversion:"
                              << dt.to_json() << " is not a leaf data type");
        };
    }
    else // we know we are big endian
    {
        switch(dt.id())
        {
            case DataType::INT8_ID:  res = H5T_STD_I8BE;  break;
            case DataType::INT16_ID: res = H5T_STD_I16BE; break;
            case DataType::INT32_ID: res = H5T_STD_I32BE; break;
            case DataType::INT64_ID: res = H5T_STD_I64BE; break;
            
            case DataType::UINT8_ID:  res = H5T_STD_U8BE;  break;
            case DataType::UINT16_ID: res = H5T_STD_U16BE; break;
            case DataType::UINT32_ID: res = H5T_STD_U32BE; break;
            case DataType::UINT64_ID: res = H5T_STD_U64BE; break;
            
            case DataType::FLOAT32_ID: res = H5T_IEEE_F32BE; break;
            case DataType::FLOAT64_ID: res = H5T_IEEE_F64BE; break;
            
            case DataType::CHAR8_STR_ID: res = H5T_C_S1; break;
            
            default:
                CONDUIT_ERROR("conduit::DataType to HDF5 Leaf DataType Conversion:"
                              << dt.to_json() << " is not a leaf data type");
        };
    }
    
    return res;
}



//-----------------------------------------------------------------------------
DataType 
hdf5_dtype_to_conduit_dtype(hid_t hdf5_dtype_id,
                            index_t num_elems)
{
    // TODO: there may be a more straight forward way to do this using
    // hdf5's data type introspection methods
    
    DataType res;
    //-----------------------------------------------
    // signed ints
    //-----------------------------------------------
    // little endian
    if(H5Tequal(hdf5_dtype_id,H5T_STD_I8LE))
    {
        res = DataType::int8(num_elems);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I16LE))
    {
        res = DataType::int16(num_elems);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I32LE))
    {
        res = DataType::int32(num_elems);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I64LE))
    {
        res = DataType::int64(num_elems);
        res.set_id(DataType::UINT64_ID);
        res.set_endianness(Endianness::LITTLE_ID);
    }
     // big endian
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I8BE))
    {
        res = DataType::int8(num_elems);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I16BE))
    {
        res = DataType::int16(num_elems);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I32BE))
    {
        res = DataType::int32(num_elems);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I64BE))
    {
        res = DataType::int64(num_elems);
        res.set_endianness(Endianness::BIG_ID);
    }
    //-----------------------------------------------
    // unsigned ints
    //-----------------------------------------------
    // little endian
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U8LE))
    {
        res = DataType::uint8(num_elems);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U16LE))
    {
        res = DataType::uint16(num_elems);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U32LE))
    {
        res = DataType::uint32(num_elems);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U64LE))
    {
        res = DataType::uint64(num_elems);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    // big endian
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U8BE))
    {
        res = DataType::uint8(num_elems);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U16BE))
    {
        res = DataType::uint16(num_elems);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U32BE))
    {
        res = DataType::uint32(num_elems);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U64BE))
    {
        res = DataType::uint64(num_elems);
        res.set_endianness(Endianness::BIG_ID);
    }
    //-----------------------------------------------
    // floating point types
    //-----------------------------------------------
    // little endian
    else if(H5Tequal(hdf5_dtype_id,H5T_IEEE_F32LE))
    {
        res = DataType::float32(num_elems);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_IEEE_F64LE))
    {
        res = DataType::float64(num_elems);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    // big endian
    else if(H5Tequal(hdf5_dtype_id,H5T_IEEE_F32BE))
    {
        res = DataType::float32(num_elems);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_IEEE_F64BE))
    {
        res = DataType::float64(num_elems);
        res.set_endianness(Endianness::BIG_ID);
    }
    //-----------------------------------------------
    // String Types
    //-----------------------------------------------
    else if(H5Tequal(hdf5_dtype_id,H5T_C_S1))
    {
        res = DataType::char8_str(num_elems);
    }
    else
    {
        CONDUIT_ERROR("Error with HDF5 DataType to conduit::DataType Leaf Conversion");
    }

    // set proper number of elems from what was passed
    res.set_number_of_elements(num_elems);
    
    return res;
}


//---------------------------------------------------------------------------//
// Write Helpers
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
bool
check_if_conduit_leaf_is_compatible_with_hdf5_obj(const DataType &dtype,
                                                  hid_t hdf5_id)
{
    bool res = true;
    H5O_info_t h5_obj_info;

    herr_t h5_status = H5Oget_info(hdf5_id, &h5_obj_info);
    
    // make sure it is a dataset ...
    if(h5_status >= 0 && h5_obj_info.type == H5O_TYPE_DATASET)
    {
        // get the hdf5 datatype that matchs the conduit dtype
        hid_t h5_dtype = conduit_dtype_to_hdf5_dtype(dtype);
        
        // get the hdf5 datatype for the passed hdf5 obj
        hid_t h5_test_dtype  = H5Dget_type(hdf5_id);
        // get the hdf5 dataspace for the passed hdf5 obj
        hid_t h5_test_dspace = H5Dget_space(hdf5_id);
    
        // we will check the 1d-properties of the hdf5 dataspace
        hssize_t h5_test_num_ele = H5Sget_simple_extent_npoints(h5_test_dspace);
    
        // make sure we have the write dtype and the 1d size matches
        if( ! ( (H5Tequal(h5_dtype, h5_test_dtype) > 0) && 
                (dtype.number_of_elements() ==  h5_test_num_ele) ) )
        {
                res = false;
        }

        CONDUIT_CHECK_HDF5_ERROR(H5Tclose(h5_test_dtype),
                         "Failed to close HDF5 Datatype " << h5_test_dtype);

        CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_test_dspace),
                         "Failed to close HDF5 Dataspace " << h5_test_dspace);
    }
    else
    {
        // bad id, or not a dataset
        res = false;
    }

    return res;
}

//---------------------------------------------------------------------------//
bool
check_if_conduit_object_is_compatible_with_hdf5_tree(const Node &node,
                                                     hid_t hdf5_id)
{
    bool res = true;
    // make sure we have a group ... 
    
    H5O_info_t h5_obj_info;
    herr_t h5_status = H5Oget_info(hdf5_id, &h5_obj_info);
    
    // make sure it is a dataset ...
    if(h5_status >= 0 && h5_obj_info.type == H5O_TYPE_GROUP)
    {
        // strong dose of evil casting, but it's ok b/c we are grownups here?
        // time we will tell ...
        NodeIterator itr = const_cast<Node*>(&node)->children();

        // call on each child with expanded path
        while(itr.has_next() && res)
        {

            Node &child = itr.next();
            // check if the HDF5 group has child with same name 
            // as the node's child
        
            hid_t h5_child_id = H5Oopen(hdf5_id,
                                        itr.path().c_str(),
                                        H5P_DEFAULT);
        
            if(h5_child_id >= 0)
            {
                // if a child does exist, we need to make sure the child is 
                // compatible with the conduit node
                res = check_if_conduit_node_is_compatible_with_hdf5_tree(child,
                                                                  h5_child_id);
            }
        
            // no child exists with this name,  we are ok (it can be created 
            // to match) check the next child
        }
    }
    else // bad id or not a group
    {
        res = false;
    }
    
    return res;
}


//---------------------------------------------------------------------------//
bool
check_if_conduit_node_is_compatible_with_hdf5_tree(const Node &node,
                                                   hid_t hdf5_id)
{
    bool res = true;
    
    DataType dt = node.dtype();
    // check for leaf or group
    if(dt.is_number() || dt.is_string())
    {
        res = check_if_conduit_leaf_is_compatible_with_hdf5_obj(dt,
                                                                hdf5_id);
    }
    else if(dt.is_object())
    {
        res = check_if_conduit_object_is_compatible_with_hdf5_tree(node,
                                                                   hdf5_id);
    }
    else // not supported
    {
        res = false;
    }
    
    return res;
}



//---------------------------------------------------------------------------//
hid_t
create_hdf5_dataset_for_conduit_leaf(const DataType &dtype,
                                     hid_t hdf5_group_id,
                                     const std::string &hdf5_dset_name)
{
    hid_t res = -1;
    hid_t h5_dtype = conduit_dtype_to_hdf5_dtype(dtype);

    hsize_t num_eles = (hsize_t) dtype.number_of_elements();
    
    hid_t   h5_dspace_id = H5Screate_simple(1,
                                            &num_eles,
                                            NULL);

    CONDUIT_CHECK_HDF5_ERROR(h5_dspace_id,
                             "Failed to create HDF5 Dataspace");

    // create new dataset
    res = H5Dcreate(hdf5_group_id,
                    hdf5_dset_name.c_str(),
                    h5_dtype,
                    h5_dspace_id,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR(res,
                         "Failed to create HDF5 Dataset " 
                          << hdf5_group_id << " " << hdf5_dset_name);

    // close our dataspace
    CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                         "Failed to close HDF5 Dataspace " << h5_dspace_id);


    return res;
}



//---------------------------------------------------------------------------//
void 
write_conduit_leaf_to_hdf5_dataset(const Node &node,
                                   hid_t hdf5_dset_id)
{
    DataType dt = node.dtype();
    
    hid_t h5_dtype_id = conduit_dtype_to_hdf5_dtype(dt);
    herr_t h5_status = -1;

    // if the node is compact, we can write directly from its data ptr
    if(dt.is_compact()) 
    {
        // write data
        h5_status = H5Dwrite(hdf5_dset_id,
                             h5_dtype_id,
                             H5S_ALL,
                             H5S_ALL,
                             H5P_DEFAULT,
                             node.data_ptr());
    }
    else 
    {
        // otherwise, we need to compact our data first
        Node n;
        node.compact_to(n);
        h5_status = H5Dwrite(hdf5_dset_id,
                             h5_dtype_id,
                             H5S_ALL,
                             H5S_ALL,
                             H5P_DEFAULT,
                             n.data_ptr());
    }

    // check write result
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Failed to write to HDF5 Dataset " << hdf5_dset_id);

    // close our dataset
    CONDUIT_CHECK_HDF5_ERROR(H5Dclose(hdf5_dset_id),
                             "Failed to close HDF5 Dataset " << hdf5_dset_id);

}

//---------------------------------------------------------------------------//
void 
write_conduit_leaf_to_hdf5_group(const Node &node,
                                 hid_t hdf5_group_id,
                                 const std::string &hdf5_dset_name)
{

    // check if the dataset exists
    H5O_info_t h5_obj_info;
    herr_t h5_info_status =  H5Oget_info_by_name(hdf5_group_id,
                                                 hdf5_dset_name.c_str(),
                                                 &h5_obj_info,
                                                 H5P_DEFAULT);

    hid_t h5_child_id = -1;

    if(h5_info_status < 0)
    {
        // if the hdf5 dataset does not exist, we need to create it
        h5_child_id = create_hdf5_dataset_for_conduit_leaf(node.dtype(),
                                                           hdf5_group_id,
                                                           hdf5_dset_name.c_str());

        CONDUIT_CHECK_HDF5_ERROR(h5_child_id,
                             "Failed to create HDF5 Dataset "
                             << " parent: " << hdf5_group_id
                             << " name: "   << hdf5_dset_name);
    }
    else
    {
        // if it does exist, we assume it is compatible
        // (this private method will only be called after a 
        //  compatibility check)
        h5_child_id = H5Dopen(hdf5_group_id,
                              hdf5_dset_name.c_str(),
                              H5P_DEFAULT);

        CONDUIT_CHECK_HDF5_ERROR(h5_child_id,
                             "Failed to open HDF5 Dataset "
                             << " parent: " << hdf5_group_id
                             << " name: "   << hdf5_dset_name);
    }
    
    // write the data
    write_conduit_leaf_to_hdf5_dataset(node,
                                       h5_child_id);

}

//---------------------------------------------------------------------------//
// assume this is called only if we know the hdf5 state is compatible 
//---------------------------------------------------------------------------//
void
write_conduit_object_to_hdf5_group(const Node &node,
                                   hid_t hdf5_group_id)
{
    // strong dose of evil casting, but it's ok b/c we are grownups here?
    // time we will tell ...
    NodeIterator itr = const_cast<Node*>(&node)->children();

    // call on each child with expanded path
    while(itr.has_next())
    {
        Node &child = itr.next();
        DataType dt = child.dtype();

        if(dt.is_number() || dt. is_string())
        {
            write_conduit_leaf_to_hdf5_group(child,
                                             hdf5_group_id,
                                             itr.path().c_str());
        }
        else if(dt.is_object())
        {
            // check if the HDF5 group has child with same name 
            // as the node's child
            H5O_info_t h5_obj_info;
            herr_t h5_info_status =  H5Oget_info_by_name(hdf5_group_id,
                                                         itr.path().c_str(),
                                                         &h5_obj_info,
                                                         H5P_DEFAULT);
            
            hid_t h5_child_id = -1;
            
            // if the hdf5 group doesn't exist, we need to create it
            if(h5_info_status < 0)
            {
                
                h5_child_id = H5Gcreate(hdf5_group_id,
                                        itr.path().c_str(),
                                        H5P_DEFAULT,
                                        H5P_DEFAULT,
                                        H5P_DEFAULT);

                CONDUIT_CHECK_HDF5_ERROR(h5_child_id,
                                     "Failed to create HDF5 Group "
                                      << " parent: " << hdf5_group_id
                                      << " name: "   << itr.path());
            }
            else
            {
                h5_child_id = H5Gopen(hdf5_group_id,
                                      itr.path().c_str(),
                                      H5P_DEFAULT);

                CONDUIT_CHECK_HDF5_ERROR(h5_child_id,
                                         "Failed to open HDF5 Group "
                                         << " parent: " << hdf5_group_id
                                         << " name: "   << itr.path());
            }

            // traverse 
            write_conduit_object_to_hdf5_group(child,
                                               h5_child_id);

            CONDUIT_CHECK_HDF5_ERROR(H5Gclose(h5_child_id),
                                     "Failed to close HDF5 Group " << h5_child_id);
        }
    }
}



//---------------------------------------------------------------------------//
// assumes compatible, dispatches to proper specific write
//---------------------------------------------------------------------------//
void
write_conduit_node_to_hdf5_tree(const Node &node,
                                hid_t hdf5_id)
{

    DataType dt = node.dtype();
    // we support a leaf or a group 
    if(dt.is_number() || dt.is_string())
    {
        write_conduit_leaf_to_hdf5_dataset(node,hdf5_id);
        
    }
    else if(dt.is_object())
    {
        write_conduit_object_to_hdf5_group(node,hdf5_id);
    }
    else // not supported
    {
        CONDUIT_ERROR("HDF5 write doesn't support LIST_ID or EMPTY_ID nodes.");
    }
}



//---------------------------------------------------------------------------//
// Read Helpers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
h5_read_leaf(hid_t hdf5_id,
            const char *hdf5_path,
            Node &dest)
{
    hid_t h5_dset_id   = H5Dopen(hdf5_id,
                                 hdf5_path,
                                 H5P_DEFAULT);

    hid_t h5_dtype_id  = H5Dget_type(h5_dset_id);
    hid_t h5_dspace_id = H5Dget_space(h5_dset_id);
    index_t nelems     = H5Sget_simple_extent_npoints(h5_dspace_id);
    DataType dt        = hdf5_dtype_to_conduit_dtype(h5_dtype_id,nelems);
    hid_t h5_status    = 0;
    
    if(dest.dtype().is_compact() && 
       dest.dtype().compatible(dt) )
    {
        // we can read directly from hdf5 dataset if compact & compatible
        h5_status = H5Dread(h5_dset_id,
                            h5_dtype_id,
                            H5S_ALL,
                            H5S_ALL,
                            H5P_DEFAULT,
                            dest.data_ptr());
        CONDUIT_CHECK_HDF5_ERROR(h5_status,
                                  "Error reading HDF5 Dataset: " << h5_dset_id);
    }
    else
    {
        // we create a temp Node b/c we want read to work for strided data
        // 
        // the hdf5 data will always be compact, source node we are reading will 
        // not unless it's already compatible and compact.
        Node n_tmp(dt);
        h5_status = H5Dread(h5_dset_id,
                            h5_dtype_id,
                            H5S_ALL,
                            H5S_ALL,
                            H5P_DEFAULT,
                            n_tmp.data_ptr());
        CONDUIT_CHECK_HDF5_ERROR(h5_status,
                                  "Error reading HDF5 Dataset: " << h5_dset_id);
        
        // copy out to our dest
        dest.set(n_tmp);
    }
    
    h5_status = H5Tclose(h5_dtype_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Error reading HDF5 DataType: " << h5_dtype_id);

    h5_status = H5Sclose(h5_dspace_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Error closing HDF5 DataSpace: " << h5_dspace_id);

    h5_status = H5Dclose(h5_dset_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Error closing HDF5 Dataset: " << h5_dset_id);
}

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
/// Data structures and callbacks that allow us to read an HDF5 hierarchy 
/// via H5Literate 
/// (adapted from: h5ex_g_traverse)
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
/// 
/// I also reviewed / partially tried the seemingly more straight forward 
/// approach in:
///  https://www.hdfgroup.org/ftp/HDF5/examples/misc-examples/h5_info.c
///
/// But since H5Gget_objtype_by_idx and H5Gget_objname_by_idx are deprecated
/// there in favor of H5Lget_name_by_idx & H5Oget_info_by_idx, there isn't a
/// direct way to check for links which could create cycles.
/// 
/// It appears that the H5Literate method (as demonstrated in the
/// h5ex_g_traverse example) is the preferred way to read an hdf5 location
/// hierarchically, even if it seems overly complex.
///
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/// Operator struct type for our H5Literate callback.
/// (adapted from: h5ex_g_traverse)
//---------------------------------------------------------------------------//
//
// Define operator data structure type for H5Literate callback.
// During recursive iteration, these structures will form a
// linked list that can be searched for duplicate groups,
// preventing infinite recursion.
//
// Cyrus: I added a pointer to hold a conduit node for construction.
//
//---------------------------------------------------------------------------//
struct h5_read_opdata
{
    unsigned                recurs;      /* Recursion level.  0=root */
    struct h5_read_opdata   *prev;        /* Pointer to previous opdata */
    haddr_t                 addr;        /* Group address */

    // pointer to conduit node, anchors traversal to 
    Node            *node;
};

//---------------------------------------------------------------------------//
/// Recursive check for cycles.
/// (adapted from: h5ex_g_traverse)
//---------------------------------------------------------------------------//
//
//  This function recursively searches the linked list of
//  h5_read_opdata structures for one whose address matches
//  target_addr.  Returns 1 if a match is found, and 0
//  otherwise.
//---------------------------------------------------------------------------//
int
h5_group_check(h5_read_opdata *od,
               haddr_t target_addr)
{
    if (od->addr == target_addr)
    {
        /* Addresses match */
        return 1;
    }
    else if (!od->recurs)
    {
        /* Root group reached with no matches */
        return 0;
    }
    else
    {
        /* Recursively examine the next node */
        return h5_group_check(od->prev, target_addr);
    }
}

//---------------------------------------------------------------------------//
/// Our main callback for H5Literate.
/// (adapted from: h5ex_g_traverse)
//---------------------------------------------------------------------------//
//  Operator function.  This function prints the name and type
//  of the object passed to it.  If the object is a group, it
//  is first checked against other groups in its path using
//  the group_check function, then if it is not a duplicate,
//  H5Literate is called for that group.  This guarantees that
//  the program will not enter infinite recursion due to a
//  circular path in the file.
//---------------------------------------------------------------------------//
herr_t
h5_literate_traverse_op_func(hid_t hdf5_id,
                             const char *hdf5_path,
                             const H5L_info_t *,// hdf5_info -- unused
                             void *hdf5_operator_data)
{
    herr_t h5_status = 0;
    herr_t h5_return_val = 0;
    H5O_info_t h5_info_buf;
    
    struct h5_read_opdata   *h5_od = (struct h5_read_opdata *) hdf5_operator_data;
                                /* Type conversion */
    unsigned        spaces = 2*(h5_od->recurs+1);
                                /* Number of whitespaces to prepend
                                   to output */

    /*
     * Get type of the object and display its name and type.
     * The name of the object is passed to this function by
     * the Library.
     */
    h5_status = H5Oget_info_by_name(hdf5_id,
                                    hdf5_path,
                                    &h5_info_buf,
                                    H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Error fetching HDF5 Object info: " 
                             << hdf5_id << ":" << hdf5_path) ;

    printf ("%*s", spaces, "");     /* Format output */
    switch (h5_info_buf.type)
    {
        case H5O_TYPE_GROUP:
        {
            printf ("Group: %s {\n", hdf5_path);

            /*
             * Check group address against linked list of operator
             * data structures.  We will always run the check, as the
             * reference count cannot be relied upon if there are
             * symbolic links, and H5Oget_info_by_name always follows
             * symbolic links.  Alternatively we could use H5Lget_info
             * and never recurse on groups discovered by symbolic
             * links, however it could still fail if an object's
             * reference count was manually manipulated with
             * H5Odecr_refcount.
             */
            if ( h5_group_check (h5_od, h5_info_buf.addr) )
            {
                // skip cycles in the graph ...
            }
            else
            {
                /*
                 * Initialize new operator data structure and
                 * begin recursive iteration on the discovered
                 * group.  The new opdata structure is given a
                 * pointer to the current one.
                 */
                struct h5_read_opdata h5_next_od;
                h5_next_od.recurs = h5_od->recurs + 1;
                h5_next_od.prev   = h5_od;
                h5_next_od.addr   = h5_info_buf.addr;
                // point our new callback data to the proper node
                h5_next_od.node   = h5_od->node->fetch_ptr(hdf5_path);
                h5_return_val = H5Literate_by_name(hdf5_id,
                                                   hdf5_path,
                                                   H5_INDEX_NAME,
                                                   H5_ITER_NATIVE,
                                                   NULL,
                                                   h5_literate_traverse_op_func,
                                                   (void *) &h5_next_od,
                                                   H5P_DEFAULT);
            }
            printf ("%*s}\n", spaces, "");
            break;
        }
        case H5O_TYPE_DATASET:
        {
            printf ("Dataset: %s\n", hdf5_path);
            Node &leaf = h5_od->node->fetch(hdf5_path);
            h5_read_leaf(hdf5_id,hdf5_path,leaf);
            break;
        }
        default:
        {
            // unsupported
        }
    }

    return h5_return_val;
}


//---------------------------------------------------------------------------//
void
hdf5_read_traverse_group(hid_t hdf5_id,
                         const char *hdf5_path,
                         Node &dest)
{
    hid_t h5_group_id;
    // open the desired group
    h5_group_id = H5Gopen(hdf5_id,
                          hdf5_path,
                          H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR(h5_group_id,
                             "Error opening HDF5 Group: " 
                             << hdf5_id << ":" << hdf5_path);

    // get info, we need to get the obj addr for cycle tracking
    H5O_info_t h5_info_buf;
    herr_t h5_status = H5Oget_info(h5_group_id,
                                   &h5_info_buf);

    // setup the callback struct we will use for  H5Literate
    struct h5_read_opdata  h5_od;
    // setup linked list tracking that allows us to detect cycles
    h5_od.recurs = 0;
    h5_od.prev = NULL;
    h5_od.addr = h5_info_buf.addr;
    // attach the pointer to our node
    h5_od.node = &dest;

    // use H5Literate to traverse
    h5_status = H5Literate(h5_group_id,
                           H5_INDEX_NAME,
                           H5_ITER_NATIVE,
                           NULL,
                           h5_literate_traverse_op_func,
                           (void *) &h5_od);
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Error calling H5Literate to traverse and "
                             << "read HDF5 hierarchy: "
                             << hdf5_id << ":" << hdf5_path);

    // close the group
    h5_status = H5Gclose(h5_group_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                            "Error closing HDF5 Group: " << h5_group_id);
}



//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// Public interface: Write Methods
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

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

    // we will write to the root if no hdf5_path is given.
    // this should be fine for OBJECT_T, not sure about others ...
    if(hdf5_path.size() == 0)
    {
        hdf5_path = "/";
    }

    hdf5_write(node,
               file_path,
               hdf5_path);
}


//---------------------------------------------------------------------------//
void
hdf5_write(const Node &node,
           const std::string &file_path,
           const std::string &hdf5_path)
{
    herr_t h5_status = 0;

    // open the hdf5 file for writing
    hid_t h5_file_id = H5Fcreate(file_path.c_str(),
                                 H5F_ACC_TRUNC,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR(h5_file_id,
                             "Error opening HDF5 file for writing: " 
                             << file_path);
    
    hdf5_write(node,
               h5_file_id,
               hdf5_path);

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
    // TODO? if the path passed is empty, this is really a case for the
    // other public signature 
    // if(hdf5_path == "")
    // {
    //     return hdf5_write(node,hdf5_id);
    // }
    
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;
    
    
    // if we have a path (even ".", or "/")
    // the passed id must represent a hdf5  group
    

    // we only want to support abs paths if hdf5_id is a file
    // if ( (not hdf5 file) && 
    //      (hdf5_path.size() > 0) && 
    //      (hdf5_path[0] == "/") )
    //{
    //    CONDUIT_ERROR("HDF5 id must represent a file to use a HDF5 "
    ///                 "absolute path (a path starts with '/'");
    //}

    // after this check strip leading forward and trailing slashes
    // if the exist

    size_t pos = 0;
    size_t len = hdf5_path.size();

    if( (hdf5_path.size() > 0) && 
        (hdf5_path[0] == '/' ) )
    {
        pos = 1;
        len--;
    }
    
    // only trim right side if we are sure there is more than one char
    // (avoid "/" case, which would already have been trimmed )
    if( (hdf5_path.size() > 1 ) && 
        (hdf5_path[hdf5_path.size()-1] == '/') )
    {
        len--;
    }
    
    std::string path = hdf5_path.substr(pos,len);
    
    // TODO: Creating the external tree is inefficient but the compatibility 
    // checks and the writes handle node paths are there.
    // revisit if this is too slow
    
    Node n;
    if(path.size() > 0)
    {
        // strong dose of evil casting, but it's ok b/c we are grownups here?
        // time we will tell ...
        n.fetch(path).set_external(const_cast<Node&>(node));
    }
    else
    {
        // strong dose of evil casting, but it's ok b/c we are grownups here?
        // time we will tell ...
        n.set_external(const_cast<Node&>(node));
    }

    // check compat
    if(check_if_conduit_node_is_compatible_with_hdf5_tree(n,
                                                          hdf5_id))
    {
        // write if we are compat
        write_conduit_node_to_hdf5_tree(n,hdf5_id);
    }
    else
    {
        CONDUIT_ERROR("Failed to write node, existing HDF5 tree is "
                      "incompatible with the node.")
    }

    // restore hdf5 error stack
}


//---------------------------------------------------------------------------//
void
hdf5_write(const Node &node,
           hid_t hdf5_id)
{
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;
    
    // check compat
    if(check_if_conduit_node_is_compatible_with_hdf5_tree(node,
                                                          hdf5_id))
    {
        // write if we are compat
        write_conduit_node_to_hdf5_tree(node,hdf5_id);
    }
    else
    {
        CONDUIT_ERROR("Failed to write node, existing HDF5 tree is "
                      << "incompatible with the node.")
    }
    // restore hdf5 error stack
}


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// Public interface: Read Methods
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//


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

    // We will read the root if no hdf5_path is given.
    if(hdf5_path.size() == 0)
    {
        hdf5_path = "/";
    }
    
    hdf5_read(file_path,
              hdf5_path,
              node);
}


//---------------------------------------------------------------------------//
void
hdf5_read(const std::string &file_path,
          const std::string &hdf5_path,
          Node &node)
{
    herr_t h5_status = 0;
    
    // open the hdf5 file for reading
    hid_t h5_file_id = H5Fopen(file_path.c_str(),
                               H5F_ACC_RDONLY,
                               H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR(h5_file_id,
                             "Error opening HDF5 file for reading: " 
                              << file_path);

    hdf5_read(h5_file_id,
              hdf5_path,
              node);
    
    // close the hdf5 file
    h5_status = H5Fclose(h5_file_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Error closing HDF5 file: " << file_path);
}

//---------------------------------------------------------------------------//
void
hdf5_read(hid_t hdf5_id,
          const std::string &hdf5_path,
          Node &dest)
{
    herr_t     h5_status = 0;
    H5O_info_t h5_info_buf;
    
    h5_status = H5Oget_info_by_name(hdf5_id,
                                    hdf5_path.c_str(),
                                    &h5_info_buf,
                                    H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Error fetching HDF5 object info from: " 
                             << hdf5_id << ":" << hdf5_path);


    switch (h5_info_buf.type)
    {
        // if hdf5_id + hdf5_path points to a group,
        // use a H5Literate traversal
        case H5O_TYPE_GROUP:
        {
            hdf5_read_traverse_group(hdf5_id,
                                     hdf5_path.c_str(),
                                     dest);
            break;
        }
        // if hdf5_id + hdf5_path points directly to a dataset
        // skip the H5Literate traversal
        case H5O_TYPE_DATASET:
        {
            h5_read_leaf(hdf5_id,
                         hdf5_path.c_str(),
                         dest);
            break;
        }
        // unsupported types
        case H5O_TYPE_UNKNOWN:
        {
            CONDUIT_ERROR("Cannot read HDF5 Object (type == H5O_TYPE_UNKNOWN )");
            break;
        }
        case H5O_TYPE_NAMED_DATATYPE:
        {
            CONDUIT_ERROR("Cannot read HDF5 Object (type == H5O_TYPE_NAMED_DATATYPE )");
            break;
        }
        case H5O_TYPE_NTYPES:
        {
            CONDUIT_ERROR("Cannot read HDF5 Object "
                          << "(type == H5O_TYPE_NTYPES [This is an invalid HDF5 type!]");
            break;
        }
        default:
        {
            CONDUIT_ERROR("Cannot read HDF5 Object (type == Unknown )");
        }
    }
}


//---------------------------------------------------------------------------//
void
hdf5_read(hid_t hdf5_id,
          Node &dest)
{
    // FIX: this could be a leaf or a group ... 
   hdf5_read(hdf5_id, ".", dest);
}


}
//-----------------------------------------------------------------------------
// -- end conduit::relay::io --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
