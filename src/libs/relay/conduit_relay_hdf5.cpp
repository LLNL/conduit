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
/// file: conduit_relay_hdf5.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay_hdf5.hpp"

//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------
#include <iostream>

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <hdf5.h>

//-----------------------------------------------------------------------------
/// macro used to check if an HDF5 object id is valid
//-----------------------------------------------------------------------------
#define CONDUIT_HDF5_VALID_ID( hdf5_id )   hdf5_id  >= 0
//-----------------------------------------------------------------------------
/// macro used to check if an HDF5 return status is ok
//-----------------------------------------------------------------------------
#define CONDUIT_HDF5_STATUS_OK(hdf5_id )   hdf5_id  >= 0

//-----------------------------------------------------------------------------
/// The CONDUIT_HDF5_ERROR macro is used for errors with ref paths.
//-----------------------------------------------------------------------------
#define CONDUIT_HDF5_ERROR( ref_path, msg )                               \
{                                                                         \
    CONDUIT_ERROR( "HDF5 Error (reference path: \"" << ref_path           \
                    << ref_path << "\") " <<  msg);                       \
}

//-----------------------------------------------------------------------------
/// The CONDUIT_HDF5_WARN macro is used for warnings with ref paths.
//-----------------------------------------------------------------------------
#define CONDUIT_HDF5_WARN( ref_path, msg )                                \
{                                                                         \
    CONDUIT_WARN( "HDF5 Warning (reference path: \"" << ref_path          \
                  << ref_path << "\") " <<  msg);                         \
}

//-----------------------------------------------------------------------------
/// The CONDUIT_CHECK_HDF5_ERROR macro is used to check error codes from HDF5.
//-----------------------------------------------------------------------------
#define CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH( hdf5_err, ref_path, msg ) \
{                                                                         \
    if( hdf5_err < 0 )                                                    \
    {                                                                     \
        std::ostringstream hdf5_err_oss;                                  \
        hdf5_err_oss << "HDF5 Error (error code: "                        \
            <<  hdf5_err                                                  \
            <<  ", reference path: \""                                    \
            <<  ref_path << "\""                                          \
            <<  ") " << msg;                                              \
        CONDUIT_ERROR( hdf5_err_oss.str());                               \
    }                                                                     \
}


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
// Private class used to hold options that control hdf5 i/o params.
// 
// These values are read by about(), and are set by io::hdf5_set_options()
// 
//
//-----------------------------------------------------------------------------

class HDF5Options
{
public:
    static bool chunking_enabled;
    static int  chunk_threshold;
    static int  chunk_size;

    static bool compact_storage_enabled;
    static int  compact_storage_threshold;

    static std::string compression_method;
    static int         compression_level;

public:
    
    //------------------------------------------------------------------------
    static void set(const Node &opts)
    {
        
        if(opts.has_child("compact_storage"))
        {
            const Node &compact = opts["compact_storage"];
            
            if(compact.has_child("enabled"))
            {
                std::string enabled = compact["enabled"].as_string();
                if(enabled == "false")
                {
                    compact_storage_enabled = false;
                }
                else
                {
                    compact_storage_enabled = true;
                }
            }

            if(compact.has_child("threshold"))
            {
                compact_storage_threshold = compact["threshold"].to_value();
            }
        }

        if(opts.has_child("chunking"))
        {
            const Node &chunking = opts["chunking"];

            if(chunking.has_child("enabled"))
            {
                std::string enabled = chunking["enabled"].as_string();
                if(enabled == "false")
                {
                    chunking_enabled = false;
                }
                else
                {
                    chunking_enabled = true;
                }
                
            }
        
            if(chunking.has_child("threshold"))
            {
                chunk_threshold = chunking["threshold"].to_value();
            }
        
            if(chunking.has_child("chunk_size"))
            {
                chunk_size = chunking["chunk_size"].to_value();
            }
        
            if(chunking.has_child("compression"))
            {
                const Node &comp = chunking["compression"];
                
                if(comp.has_child("method"))
                {
                    compression_method = comp["method"].as_string();
                }
                if(comp.has_path("level"))
                {
                    compression_level = comp["level"].to_value();
                }
            }
        }
    }

    //------------------------------------------------------------------------
    static void about(Node &opts)
    {
        opts.reset();

        if(compact_storage_enabled)
        {
            opts["compact_storage/enabled"] = "true";
        }
        else
        {
            opts["compact_storage/enabled"] = "false";
        }
        
        opts["compact_storage/threshold"] = compact_storage_threshold;

        if(chunking_enabled)
        {
            opts["chunking/enabled"] = "true";
        }
        else
        {
            opts["chunking/enabled"] = "false";
        }
        
        opts["chunking/threshold"] = chunk_threshold;
        opts["chunking/chunk_size"] = chunk_size;

        opts["chunking/compression/method"] = compression_method;
        if(compression_method == "gzip")
        {
            opts["chunking/compression/level"] = compression_level;
        }
    }
};

// default hdf5 i/o settings

bool HDF5Options::compact_storage_enabled   = true;
int  HDF5Options::compact_storage_threshold = 1024;

bool        HDF5Options::chunking_enabled   = true;
int         HDF5Options::chunk_size         = 1000000; // 1 mb 
int         HDF5Options::chunk_threshold    = 2000000; // 2 mb

std::string HDF5Options::compression_method = "gzip";
int         HDF5Options::compression_level  = 5;


//-----------------------------------------------------------------------------
void
hdf5_set_options(const Node &opts)
{
    HDF5Options::set(opts);
}

//-----------------------------------------------------------------------------
void
hdf5_options(Node &opts)
{
    HDF5Options::about(opts);
}

//-----------------------------------------------------------------------------
// Private class used to suppress HDF5 error messages.
// 
// Creating an instance of this class will disable the current HDF5 error 
// callbacks.  The default HDF5 callback print error messages  we probing 
// properties of the HDF5 tree. When the instance is destroyed, the previous
// error state is restored.
//-----------------------------------------------------------------------------
class HDF5ErrorStackSupressor
{
public:
        HDF5ErrorStackSupressor()
        :  herr_func(NULL),
           herr_func_client_data(NULL)
        {
            disable_hdf5_error_func();
        }
        
       ~HDF5ErrorStackSupressor()
        {
            restore_hdf5_error_func();
        }

private:
    // saves current error func.
    // for hdf5's default setup this disable printed error messages 
    // that occur when we are probing properties of the hdf5 tree
    void disable_hdf5_error_func()
    {
            H5Eget_auto(H5E_DEFAULT,
                        &herr_func, 
                        &herr_func_client_data);
            
            H5Eset_auto(H5E_DEFAULT,
                        NULL,
                        NULL);
    }

    // restores saved error func
    void restore_hdf5_error_func()
    {
        H5Eset_auto(H5E_DEFAULT,
                    herr_func,
                    herr_func_client_data);
    }

    // callback used for hdf5 error interface
    H5E_auto2_t  herr_func;
    // data container for hdf5 error interface callback
    void         *herr_func_client_data; 
};

//-----------------------------------------------------------------------------
// helper method decls
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// helpers for data type conversions
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// helpers for checking if compatible 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool  check_if_conduit_leaf_is_compatible_with_hdf5_obj(const DataType &dtype,
                                                  const std::string &ref_path,
                                                                hid_t hdf5_id);
 
//-----------------------------------------------------------------------------
bool  check_if_conduit_object_is_compatible_with_hdf5_tree(const Node &node,
                                                const std::string &ref_path,
                                                             hid_t hdf5_id);

//-----------------------------------------------------------------------------
bool  check_if_conduit_node_is_compatible_with_hdf5_tree(const Node &node,
                                              const std::string &ref_path,
                                                            hid_t hdf5_id);


//-----------------------------------------------------------------------------
// helpers for writing
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
hid_t create_hdf5_dataset_for_conduit_leaf(const DataType &dt,
                                           const std::string &ref_path,
                                           hid_t hdf5_group_id,
                                           const std::string &hdf5_dset_name);

//-----------------------------------------------------------------------------
void  write_conduit_leaf_to_hdf5_dataset(const Node &node,
                                         const std::string &ref_path,
                                         hid_t hdf5_dset_id);

//-----------------------------------------------------------------------------
void  write_conduit_leaf_to_hdf5_group(const Node &node,
                                       const std::string &ref_path,
                                       hid_t hdf5_group_id,
                                       const std::string &hdf5_dset_name);

//-----------------------------------------------------------------------------
void  write_conduit_object_to_hdf5_group(const Node &node,
                                         const std::string &ref_path,
                                         hid_t hdf5_group_id);


//-----------------------------------------------------------------------------
// helpers for reading
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void read_hdf5_dataset_into_conduit_node(hid_t hdf5_dset_id,
                                         const std::string &ref_path,
                                         Node &dest);

//-----------------------------------------------------------------------------
void read_hdf5_group_into_conduit_node(hid_t hdf5_group_id,
                                       const std::string &ref_path,
                                       Node &dest);

//-----------------------------------------------------------------------------
void read_hdf5_tree_into_conduit_node(hid_t hdf5_id,
                                      const std::string &ref_path,
                                      Node &dest);




//-----------------------------------------------------------------------------
// helper used to properly create a new ref_path for a child 
std::string
join_ref_paths(const std::string &parent, const std::string &child)
{
    if(parent.size() > 0)
    {
        return parent + "/" + child;
    }
    else
    {
        return child;
    }
}

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Data Type Helper methods that are a part of public conduit::relay::io
//
//  conduit_dtype_to_hdf5_dtype
//  hdf5_dtype_to_conduit_dtype
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
hid_t
conduit_dtype_to_hdf5_dtype(const DataType &dt,
                            const std::string &ref_path)
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
                CONDUIT_HDF5_ERROR(ref_path,
                                  "conduit::DataType to HDF5 Leaf DataType "
                                  << "Conversion:"
                                  << dt.to_json() 
                                  << " is not a leaf data type");
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
                CONDUIT_HDF5_ERROR(ref_path,
                                  "conduit::DataType to HDF5 Leaf DataType "
                                  << "Conversion:"
                                  << dt.to_json() 
                                  << " is not a leaf data type");
        };
    }
    
    return res;
}



//-----------------------------------------------------------------------------
DataType 
hdf5_dtype_to_conduit_dtype(hid_t hdf5_dtype_id,
                            index_t num_elems,
                            const std::string &ref_path)
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
    //-----------------------------------------------
    // Unsupported
    //-----------------------------------------------
    else
    {
        CONDUIT_HDF5_ERROR(ref_path,
                           "Error with HDF5 DataType to conduit::DataType "
                           << "Leaf Conversion");
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
                                                  const std::string &ref_path,
                                                  hid_t hdf5_id)
{
    bool res = true;
    H5O_info_t h5_obj_info;

    herr_t h5_status = H5Oget_info(hdf5_id, &h5_obj_info);
    
    // make sure it is a dataset ...
    if( CONDUIT_HDF5_STATUS_OK(h5_status) && 
        ( h5_obj_info.type == H5O_TYPE_DATASET ) )
    {
        // get the hdf5 dataspace for the passed hdf5 obj
        hid_t h5_test_dspace = H5Dget_space(hdf5_id);
        // a dataset with H5S_NULL data space is only compatible with 
        // conduit empty
        if( H5Sget_simple_extent_type(h5_test_dspace) == H5S_NULL &&
            !dtype.is_empty())
        {
            res = false;
        }
        else
        {
            // get the hdf5 datatype that matchs the conduit dtype
            hid_t h5_dtype = conduit_dtype_to_hdf5_dtype(dtype,
                                                         ref_path);
        
            // get the hdf5 datatype for the passed hdf5 obj
            hid_t h5_test_dtype  = H5Dget_type(hdf5_id);
    
            // we will check the 1d-properties of the hdf5 dataspace
            hssize_t h5_test_num_ele = H5Sget_simple_extent_npoints(h5_test_dspace);
    
            // make sure we have the write dtype and the 1d size matches
            if( ! ( (H5Tequal(h5_dtype, h5_test_dtype) > 0) && 
                    (dtype.number_of_elements() ==  h5_test_num_ele) ) )
            {
                    res = false;
            }

            CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Tclose(h5_test_dtype),
                                                   ref_path,
                                     "Failed to close HDF5 Datatype " 
                                     << h5_test_dtype);
        }

        CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Sclose(h5_test_dspace),
                                               ref_path,
                         "Failed to close HDF5 Dataspace " << h5_test_dspace);
    }
    else
    {
        // bad id, or not a dataset
        res = false;
    }
    
    if(res == false)
    {
        CONDUIT_INFO("leaf in Conduit Node at path " << ref_path <<
                     " is not compatible with given HDF5 tree at path"
                     << ref_path);
    }

    return res;
}

//---------------------------------------------------------------------------//
bool
check_if_conduit_object_is_compatible_with_hdf5_tree(const Node &node,
                                                     const std::string &ref_path,
                                                     hid_t hdf5_id)
{
    bool res = true;
    // make sure we have a group ... 
    
    H5O_info_t h5_obj_info;
    herr_t h5_status = H5Oget_info(hdf5_id, &h5_obj_info);
    
    // make sure it is a dataset ...
    if( CONDUIT_HDF5_STATUS_OK(h5_status) &&
        (h5_obj_info.type == H5O_TYPE_GROUP) )
    {
        NodeConstIterator itr = node.children();

        // call on each child with expanded path
        while(itr.has_next() && res)
        {

            const Node &child = itr.next();
            // check if the HDF5 group has child with same name 
            // as the node's child
        
            hid_t h5_child_obj = H5Oopen(hdf5_id,
                                        itr.name().c_str(),
                                        H5P_DEFAULT);
        
            std::string chld_ref_path = join_ref_paths(ref_path,itr.name());
            if( CONDUIT_HDF5_VALID_ID(h5_child_obj) )
            {
                // if a child does exist, we need to make sure the child is 
                // compatible with the conduit node
                res = check_if_conduit_node_is_compatible_with_hdf5_tree(child,
                                                                 chld_ref_path,
                                                                  h5_child_obj);
            
                CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Oclose(h5_child_obj),
                                                       ref_path,
                             "Failed to close HDF5 Object: " << h5_child_obj);
            }
            // no child exists with this name,  we are ok (it can be created 
            // to match) check the next child
        }
    }
    else // bad id or not a group
    {
        CONDUIT_INFO("object in Conduit Node at path " << ref_path << 
                     " is not compatible with given HDF5 tree at path"
                     << ref_path );
        res = false;
    }
    
    return res;
}


//---------------------------------------------------------------------------//
bool
check_if_conduit_node_is_compatible_with_hdf5_tree(const Node &node,
                                                   const std::string &ref_path,
                                                   hid_t hdf5_id)
{
    bool res = true;
    
    DataType dt = node.dtype();
    // check for leaf or group
    if(dt.is_number() || dt.is_string())
    {
        res = check_if_conduit_leaf_is_compatible_with_hdf5_obj(dt,
                                                                ref_path,
                                                                hdf5_id);
    }
    else if(dt.is_object())
    {
        res = check_if_conduit_object_is_compatible_with_hdf5_tree(node,
                                                                   ref_path,
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
create_hdf5_compact_plist_for_conduit_leaf()
{
    hid_t h5_cprops_id = H5Pcreate(H5P_DATASET_CREATE);

    H5Pset_layout(h5_cprops_id,H5D_COMPACT);
    
    return h5_cprops_id;
}


//---------------------------------------------------------------------------//
hid_t
create_hdf5_chunked_plist_for_conduit_leaf(const DataType &dtype)
{
    hid_t h5_cprops_id = H5Pcreate(H5P_DATASET_CREATE);

    // Turn on chunking
    
    // hdf5 sets chunking in elements, not bytes, 
    // our options are in bytes, so convert to # of elems
    hsize_t h5_chunk_size =  (hsize_t) (HDF5Options::chunk_size / dtype.element_bytes()); 

    H5Pset_chunk(h5_cprops_id, 1, &h5_chunk_size);

    if(HDF5Options::compression_method == "gzip" )
    {
        // Turn on compression
        H5Pset_shuffle(h5_cprops_id);
        H5Pset_deflate(h5_cprops_id, HDF5Options::compression_level);
    }

    return h5_cprops_id;
}


//---------------------------------------------------------------------------//
hid_t
create_hdf5_dataset_for_conduit_leaf(const DataType &dtype,
                                     const std::string &ref_path,
                                     hid_t hdf5_group_id,
                                     const std::string &hdf5_dset_name)
{
    hid_t res = -1;
    
    hid_t h5_dtype = conduit_dtype_to_hdf5_dtype(dtype,ref_path);

    hsize_t num_eles = (hsize_t) dtype.number_of_elements();
    
    
    hid_t h5_cprops_id = H5P_DEFAULT;
    

    if( HDF5Options::compact_storage_enabled &&
        dtype.bytes_compact() <= HDF5Options::compact_storage_threshold)
    {
        h5_cprops_id = create_hdf5_compact_plist_for_conduit_leaf();
    }
    else if( HDF5Options::chunking_enabled &&
             dtype.bytes_compact() > HDF5Options::chunk_threshold)
    {
        h5_cprops_id = create_hdf5_chunked_plist_for_conduit_leaf(dtype);
    }

    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_cprops_id,
                                           ref_path,
                                           "Failed to create HDF5 property list");

    hid_t h5_dspace_id = H5Screate_simple(1,
                                          &num_eles,
                                          NULL);

    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_dspace_id,
                                           ref_path,
                                           "Failed to create HDF5 Dataspace");

    // create new dataset
    res = H5Dcreate(hdf5_group_id,
                    hdf5_dset_name.c_str(),
                    h5_dtype,
                    h5_dspace_id,
                    H5P_DEFAULT,
                    h5_cprops_id,
                    H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(res,
                                           ref_path,
                                           "Failed to create HDF5 Dataset " 
                                           << hdf5_group_id << " " 
                                           << hdf5_dset_name);

    // close plist used for compression
    if(h5_cprops_id != H5P_DEFAULT)
    {
        CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Pclose(h5_cprops_id),
                                                       ref_path,
                                           "Failed to close HDF5 compression "
                                           "property list " 
                                                       << h5_cprops_id);
    }

    // close our dataspace
    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Sclose(h5_dspace_id),
                                           ref_path,
                                           "Failed to close HDF5 Dataspace " 
                                           << h5_dspace_id);


    return res;
}

//---------------------------------------------------------------------------//
hid_t
create_hdf5_dataset_for_conduit_empty(hid_t hdf5_group_id,
                                      const std::string &ref_path,
                                      const std::string &hdf5_dset_name)
{
    hid_t res = -1;
    // for conduit empty, use an opaque data type with zero size;
    hid_t h5_dtype_id  = H5Tcreate(H5T_OPAQUE, 1);
    hid_t h5_dspace_id = H5Screate(H5S_NULL);
    
    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_dspace_id,
                                           ref_path,
                                           "Failed to create HDF5 Dataspace");

    // create new dataset
    res = H5Dcreate(hdf5_group_id,
                    hdf5_dset_name.c_str(),
                    h5_dtype_id,
                    h5_dspace_id,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(res,
                                           ref_path,
                                           "Failed to create HDF5 Dataset " 
                                           << hdf5_group_id 
                                           << " " << hdf5_dset_name);
    // close our datatype
    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Tclose(h5_dtype_id),
                                           ref_path,
                                           "Failed to close HDF5 Datatype");
    // close our dataspace
    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Sclose(h5_dspace_id),
                                           ref_path,
                                           "Failed to close HDF5 Dataspace " 
                                           << h5_dspace_id);

    return res;
}




//---------------------------------------------------------------------------//
void 
write_conduit_leaf_to_hdf5_dataset(const Node &node,
                                   const std::string &ref_path,
                                   hid_t hdf5_dset_id)
{
    DataType dt = node.dtype();
    
    hid_t h5_dtype_id = conduit_dtype_to_hdf5_dtype(dt,ref_path);
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
    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_status,
                                           ref_path,
                                           "Failed to write to HDF5 Dataset "
                                           << hdf5_dset_id);

}

//---------------------------------------------------------------------------//
void 
write_conduit_leaf_to_hdf5_group(const Node &node,
                                 const std::string &ref_path,
                                 hid_t hdf5_group_id,
                                 const std::string &hdf5_dset_name)
{
    // data set case ...

    // check if the dataset exists
    H5O_info_t h5_obj_info;
    herr_t h5_info_status =  H5Oget_info_by_name(hdf5_group_id,
                                                 hdf5_dset_name.c_str(),
                                                 &h5_obj_info,
                                                 H5P_DEFAULT);

    hid_t h5_child_id = -1;

    if( CONDUIT_HDF5_STATUS_OK(h5_info_status) )
    {
        // if it does exist, we assume it is compatible
        // (this private method will only be called after a 
        //  compatibility check)
        h5_child_id = H5Dopen(hdf5_group_id,
                              hdf5_dset_name.c_str(),
                              H5P_DEFAULT);

        CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_child_id,
                                               ref_path,
                                               "Failed to open HDF5 Dataset "
                                               << " parent: "
                                               << hdf5_group_id
                                               << " name: "
                                               << hdf5_dset_name);
    }
    else
    {
        // if the hdf5 dataset does not exist, we need to create it
        h5_child_id = create_hdf5_dataset_for_conduit_leaf(node.dtype(),
                                                           ref_path,
                                                           hdf5_group_id,
                                                           hdf5_dset_name);

        CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_child_id,
                                               ref_path,
                                               "Failed to create HDF5 Dataset "
                                               << " parent: "
                                               << hdf5_group_id
                                               << " name: "
                                               << hdf5_dset_name);
    }
    
    std::string chld_ref_path = join_ref_paths(ref_path,hdf5_dset_name);
    // write the data
    write_conduit_leaf_to_hdf5_dataset(node,
                                       chld_ref_path,
                                       h5_child_id);
    
    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Dclose(h5_child_id),
                                           ref_path,
                                           "Failed to close HDF5 Dataset: "
                                           << h5_child_id);
}

//---------------------------------------------------------------------------//
void 
write_conduit_empty_to_hdf5_group(hid_t hdf5_group_id,
                                  const std::string &ref_path,
                                  const std::string &hdf5_dset_name)
{

    // check if the dataset exists
    H5O_info_t h5_obj_info;
    herr_t h5_info_status =  H5Oget_info_by_name(hdf5_group_id,
                                                 hdf5_dset_name.c_str(),
                                                 &h5_obj_info,
                                                 H5P_DEFAULT);

    hid_t h5_child_id = -1;

    if( CONDUIT_HDF5_STATUS_OK(h5_info_status) )
    {
        // if it does exist, we assume it is compatible
        // (this private method will only be called after a 
        //  compatibility check)
    }
    else
    {
        // if the hdf5 dataset does not exist, we need to create it
        h5_child_id = create_hdf5_dataset_for_conduit_empty(hdf5_group_id,
                                                            ref_path,
                                                            hdf5_dset_name);

        CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_child_id,
                                               ref_path,
                                               "Failed to create HDF5 Dataset "
                                               << " parent: " << hdf5_group_id
                                               << " name: "   << hdf5_dset_name);

        CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Dclose(h5_child_id), 
                                              ref_path,
                                              "Failed to close HDF5 Dataset: "
                                              << h5_child_id);
    }
    

}

//---------------------------------------------------------------------------//
// assume this is called only if we know the hdf5 state is compatible 
//---------------------------------------------------------------------------//
void
write_conduit_object_to_hdf5_group(const Node &node,
                                   const std::string &ref_path,
                                   hid_t hdf5_group_id)
{
    NodeConstIterator itr = node.children();

    // call on each child with expanded path
    while(itr.has_next())
    {
        const Node &child = itr.next();
        DataType dt = child.dtype();

        if(dt.is_number() || dt. is_string())
        {
            write_conduit_leaf_to_hdf5_group(child,
                                             ref_path,
                                             hdf5_group_id,
                                             itr.name().c_str());
        }
        else if(dt.is_empty())
        {
            // if we have an empty node, it will become
            // a dataset with an null shape
            write_conduit_empty_to_hdf5_group(hdf5_group_id,
                                              ref_path,
                                              itr.name().c_str());
        }
        else if(dt.is_object())
        {
            // check if the HDF5 group has child with same name 
            // as the node's child
            H5O_info_t h5_obj_info;
            herr_t h5_info_status =  H5Oget_info_by_name(hdf5_group_id,
                                                         itr.name().c_str(),
                                                         &h5_obj_info,
                                                         H5P_DEFAULT);
            
            hid_t h5_child_id = -1;
            
            if( CONDUIT_HDF5_STATUS_OK(h5_info_status) )
            {
                // if the hdf5 group exists, open it
                h5_child_id = H5Gopen(hdf5_group_id,
                                      itr.name().c_str(),
                                      H5P_DEFAULT);

                CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_child_id,
                                                       ref_path,
                                             "Failed to open HDF5 Group "
                                             << " parent: " << hdf5_group_id
                                             << " name: "   << itr.name());
            }
            else
            {
                // if the hdf5 group doesn't exist, we need to create it
                hid_t h5_gc_plist = H5Pcreate(H5P_GROUP_CREATE);
        
                CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_gc_plist,
                                                       ref_path,
                                 "Failed to create H5P_GROUP_CREATE property "
                                 << " list");
        
                // track creation order
                herr_t h5_status = H5Pset_link_creation_order(h5_gc_plist, 
                        ( H5P_CRT_ORDER_TRACKED |  H5P_CRT_ORDER_INDEXED) );

                CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_status,
                                                       ref_path,
                                 "Failed to set group link creation property");

                // prefer compact group storage 
                // https://support.hdfgroup.org/HDF5/doc/RM/RM_H5G.html#Group-GroupStyles
                h5_status = H5Pset_link_phase_change(h5_gc_plist,
                                                     32,  // max for compact storage
                                                     32); // min for dense storage

                CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_status,
                                                       ref_path,
                            "Failed to set group link phase change property ");

                // calc hints for meta data about link names
                NodeConstIterator chld_itr = itr.node().children();
                
                index_t chld_names_avg_size = 0;
                index_t num_children = itr.node().number_of_children();
                
                while(chld_itr.has_next())
                {
                    chld_itr.next();
                    chld_names_avg_size +=chld_itr.name().size();
                }
                
                if(chld_names_avg_size > 0 && num_children > 0 )
                {
                    chld_names_avg_size = chld_names_avg_size / num_children;
                }

                // set hints for meta data about link names
                h5_status = H5Pset_est_link_info(h5_gc_plist,
                                                 num_children, // number of children
                                                 chld_names_avg_size); // est name size

                CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_status,
                                                       ref_path,
                            "Failed to set group est link info property ");

                h5_child_id = H5Gcreate(hdf5_group_id,
                                        itr.name().c_str(),
                                        H5P_DEFAULT,
                                        h5_gc_plist,
                                        H5P_DEFAULT);

                CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_child_id,
                                                       ref_path,
                                      "Failed to create HDF5 Group "
                                      << " parent: " << hdf5_group_id
                                      << " name: "   << itr.name());

                CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Pclose(h5_gc_plist),
                                                       ref_path,
                                     "Failed to close HDF5 H5P_GROUP_CREATE "
                                     << "property list: " 
                                     << h5_gc_plist);
            }


            // traverse 
            write_conduit_object_to_hdf5_group(child,
                                               ref_path,
                                               h5_child_id);

            CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Gclose(h5_child_id),
                                                   ref_path,
                                     "Failed to close HDF5 Group " 
                                     << h5_child_id);
        }
        else
        {
            CONDUIT_HDF5_WARN(ref_path,
                               "DataType \'" 
                               << DataType::id_to_name(dt.id())
                               <<"\' not supported for relay HDF5 I/O");
        }
    }
}



//---------------------------------------------------------------------------//
// assumes compatible, dispatches to proper specific write
//---------------------------------------------------------------------------//
void
write_conduit_node_to_hdf5_tree(const Node &node,
                                const std::string &ref_path,
                                hid_t hdf5_id)
{

    DataType dt = node.dtype();
    // we support a leaf or a group 
    if(dt.is_number() || dt.is_string())
    {
        write_conduit_leaf_to_hdf5_dataset(node,
                                           ref_path,
                                           hdf5_id);
    }
    else if(dt.is_object())
    {
        write_conduit_object_to_hdf5_group(node,
                                           ref_path,
                                           hdf5_id);
    }
    else // not supported
    {
        CONDUIT_HDF5_ERROR(ref_path,
                   "HDF5 write doesn't support LIST_ID or EMPTY_ID nodes.");
    }
}



//---------------------------------------------------------------------------//
// Read Helpers
//---------------------------------------------------------------------------//


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
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// For conduit+relay: Added a pointer to hold a conduit node for construction.
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
//---------------------------------------------------------------------------//
struct h5_read_opdata
{
    unsigned                recurs;      /* Recursion level.  0=root */
    struct h5_read_opdata   *prev;        /* Pointer to previous opdata */
    haddr_t                 addr;        /* Group address */

    // pointer to conduit node, anchors traversal to 
    Node            *node;
    std::string      ref_path;
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
h5l_iterate_traverse_op_func(hid_t hdf5_id,
                             const char *hdf5_path,
                             const H5L_info_t *,// hdf5_info -- unused
                             void *hdf5_operator_data)
{
    herr_t h5_status = 0;
    herr_t h5_return_val = 0;
    H5O_info_t h5_info_buf;

    /* Type conversion */
    struct h5_read_opdata *h5_od = (struct h5_read_opdata*)hdf5_operator_data;


    /*
     * Get type of the object and display its name and type.
     * The name of the object is passed to this function by
     * the Library.
     */
    h5_status = H5Oget_info_by_name(hdf5_id,
                                    hdf5_path,
                                    &h5_info_buf,
                                    H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_status,
                                           h5_od->ref_path,
                                           "Error fetching HDF5 Object info: "
                                           << " parent: " << hdf5_id 
                                           << " path:"    << hdf5_path) ;

    std::string chld_ref_path = h5_od->ref_path  + 
                                std::string("/") + 
                                std::string(hdf5_path);
    switch (h5_info_buf.type)
    {
        case H5O_TYPE_GROUP:
        {
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
                hid_t h5_group_id = H5Gopen(hdf5_id,
                                           hdf5_path,
                                           H5P_DEFAULT);
                CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_group_id,
                                                       h5_od->ref_path,
                                                       "Error opening HDF5 "
                                                       << "Group: " 
                                                       << " parent: "
                                                       << hdf5_id 
                                                       << " path:"
                                                       << hdf5_path);

                // execute traversal for this group
                Node &chld_node = h5_od->node->fetch(hdf5_path);

                read_hdf5_group_into_conduit_node(h5_group_id,
                                                  chld_ref_path,
                                                  chld_node);

                // close the group
                CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Gclose(h5_group_id),
                                                       h5_od->ref_path,
                                                       "Error closing HDF5 "
                                                       << "Group: "
                                                       << h5_group_id);
            }
            break;
        }
        case H5O_TYPE_DATASET:
        {
            Node &leaf = h5_od->node->fetch(hdf5_path);
            // open hdf5 dataset at path
            hid_t h5_dset_id = H5Dopen(hdf5_id,
                                       hdf5_path,
                                       H5P_DEFAULT);

            CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_dset_id,
                                                   h5_od->ref_path,
                                                   "Error opening HDF5 "
                                                   << " Dataset: " 
                                                   << " parent: "
                                                   << hdf5_id 
                                                   << " path:"
                                                   << hdf5_path);

            read_hdf5_dataset_into_conduit_node(h5_dset_id,
                                                chld_ref_path,
                                                leaf);
            
            // close the dataset
            CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Dclose(h5_dset_id),
                                                   h5_od->ref_path,
                                                   "Error closing HDF5 "
                                                   << " Dataset: " 
                                                   << h5_dset_id);
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
read_hdf5_group_into_conduit_node(hid_t hdf5_group_id,
                                  const std::string &ref_path,
                                  Node &dest)
{
    // we want to make sure this is a conduit object
    // even if it doesn't have any children
    dest.set(DataType::object());
    
    
    // get info, we need to get the obj addr for cycle tracking
    H5O_info_t h5_info_buf;
    herr_t h5_status = H5Oget_info(hdf5_group_id,
                                   &h5_info_buf);

    // setup the callback struct we will use for  H5Literate
    struct h5_read_opdata  h5_od;
    // setup linked list tracking that allows us to detect cycles
    h5_od.recurs = 0;
    h5_od.prev = NULL;
    h5_od.addr = h5_info_buf.addr;
    // attach the pointer to our node
    h5_od.node = &dest;
    // keep ref path
    h5_od.ref_path = ref_path;

    H5_index_t h5_grp_index_type = H5_INDEX_NAME;
    
    // check for creation order index using propertylist

    hid_t h5_gc_plist = H5Gget_create_plist(hdf5_group_id);

    if( CONDUIT_HDF5_VALID_ID(h5_gc_plist) )
    {
        unsigned int h5_gc_flags = 0;
        h5_status = H5Pget_link_creation_order(h5_gc_plist,
                                           &h5_gc_flags);

        // first make sure we have the link creation order plist
        if( CONDUIT_HDF5_STATUS_OK(h5_status) )
        {
            // check that we have both order_tracked and order_indexed
            if( h5_gc_flags & (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED) )
            {
                // if so, we can use creation order in h5literate
                h5_grp_index_type = H5_INDEX_CRT_ORDER;
            }
        }
    
        CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Pclose(h5_gc_plist),
                                               ref_path,
                                               "Failed to close HDF5 "
                                               << "H5P_GROUP_CREATE "
                                               << "property list: " 
                                               << h5_gc_plist);
    }
    
    

    // use H5Literate to traverse
    h5_status = H5Literate(hdf5_group_id,
                           h5_grp_index_type,
                           H5_ITER_INC,
                           NULL,
                           h5l_iterate_traverse_op_func,
                           (void *) &h5_od);

    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_status,
                                           ref_path,
                                           "Error calling H5Literate to "
                                           << "traverse and read HDF5 "
                                           << "hierarchy: "
                                           << hdf5_group_id);
}

//---------------------------------------------------------------------------//
void
read_hdf5_dataset_into_conduit_node(hid_t hdf5_dset_id,
                                    const std::string &ref_path,
                                    Node &dest)
{
    hid_t h5_dspace_id = H5Dget_space(hdf5_dset_id);
    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_dspace_id,
                                           ref_path,
                                           "Error reading HDF5 Dataspace: " 
                                           << hdf5_dset_id);

    // check for empty case
    if(H5Sget_simple_extent_type(h5_dspace_id) == H5S_NULL)
    {
        // change to empty
        dest.reset();
    }
    else
    {
        hid_t h5_dtype_id  = H5Dget_type(hdf5_dset_id); 
    
        CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_dtype_id,
                                               ref_path,
                                               "Error reading HDF5 Datatype: "
                                               << hdf5_dset_id);


    
        index_t nelems     = H5Sget_simple_extent_npoints(h5_dspace_id);
        DataType dt        = hdf5_dtype_to_conduit_dtype(h5_dtype_id,
                                                         nelems,
                                                         ref_path);
        // if the endianness of the dset in the file doesn't
        // match the current machine we always want to convert it
        // on read.

        // check endianness
        if(!dt.endianness_matches_machine())
        {
            // if they don't match, modify the dt
            // and get the proper hdf5 data type handle
            dt.set_endianness(Endianness::machine_default());
            
            // clean up our old handle
            CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Tclose(h5_dtype_id),
                                                   ref_path,
                                        "Error closing HDF5 Datatype: "
                                        << h5_dtype_id);
            // get ref to standard variant of this dtype
            h5_dtype_id  = conduit_dtype_to_hdf5_dtype(dt,
                                                       ref_path);

            CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_dtype_id,
                                                   ref_path,
                                        "Error creating HDF5 Datatype");

            // copy this handle, b/c clean up code later will close it
            h5_dtype_id  = H5Tcopy(h5_dtype_id);
            CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_dtype_id,
                                                   ref_path,
                                        "Error copying HDF5 Datatype");
        }
        
        hid_t h5_status    = 0;
    
        if(dest.dtype().is_compact() && 
           dest.dtype().compatible(dt) )
        {
            // we can read directly from hdf5 dataset if compact 
            // & compatible
            h5_status = H5Dread(hdf5_dset_id,
                                h5_dtype_id,
                                H5S_ALL,
                                H5S_ALL,
                                H5P_DEFAULT,
                                dest.data_ptr());
        }
        else
        {
            // we create a temp Node b/c we want read to work for 
            // strided data
            // 
            // the hdf5 data will always be compact, source node we are 
            // reading will not unless it's already compatible and compact.
            Node n_tmp(dt);
            h5_status = H5Dread(hdf5_dset_id,
                                h5_dtype_id,
                                H5S_ALL,
                                H5S_ALL,
                                H5P_DEFAULT,
                                n_tmp.data_ptr());
        
            // copy out to our dest
        dest.set(n_tmp);
        }

        CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_status,
                                               ref_path,
                                               "Error reading HDF5 Dataset: "
                                                << hdf5_dset_id);
    
        CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Tclose(h5_dtype_id),
                                               ref_path,
                                               "Error closing HDF5 Datatype: "
                                               << h5_dtype_id);

    }

    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Sclose(h5_dspace_id),
                                           ref_path,
                                           "Error closing HDF5 Dataspace: "
                                           << h5_dspace_id);

}

//---------------------------------------------------------------------------//
void
read_hdf5_tree_into_conduit_node(hid_t hdf5_id,
                                 const std::string  &ref_path,
                                 Node &dest)
{
    herr_t     h5_status = 0;
    H5O_info_t h5_info_buf;
    
    h5_status = H5Oget_info(hdf5_id,&h5_info_buf);

    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_status,
                                           ref_path,
                                           "Error fetching HDF5 object "
                                           << "info from: " 
                                           << hdf5_id);


    switch (h5_info_buf.type)
    {
        // if hdf5_id + hdf5_path points to a group,
        // use a H5Literate traversal
        case H5O_TYPE_GROUP:
        {
            read_hdf5_group_into_conduit_node(hdf5_id,
                                              ref_path,
                                              dest);
            break;
        }
        // if hdf5_id + hdf5_path points directly to a dataset
        // skip the H5Literate traversal
        case H5O_TYPE_DATASET:
        {
            read_hdf5_dataset_into_conduit_node(hdf5_id,
                                                ref_path,
                                                dest);
            break;
        }
        // unsupported types
        case H5O_TYPE_UNKNOWN:
        {
            CONDUIT_HDF5_ERROR(ref_path,
                               "Cannot read HDF5 Object : "
                                << "(type == H5O_TYPE_UNKNOWN )");
            break;
        }
        case H5O_TYPE_NAMED_DATATYPE:
        {
            CONDUIT_HDF5_ERROR(ref_path,
                               "Cannot read HDF5 Object "
                               << "(type == H5O_TYPE_NAMED_DATATYPE )");
            break;
        }
        case H5O_TYPE_NTYPES:
        {
            CONDUIT_HDF5_ERROR(ref_path,
              "Cannot read HDF5 Object "
               << "(type == H5O_TYPE_NTYPES [This is an invalid HDF5 type!]");
            break;
        }
        default:
        {
            CONDUIT_HDF5_ERROR(ref_path,
                               "Cannot read HDF5 Object (type == Unknown )");
        }
    }
}



//---------------------------------------------------------------------------//
hid_t
create_hdf5_file_access_plist()
{
    // create property list and set use latest lib ver settings 
    hid_t h5_fa_props = H5Pcreate(H5P_FILE_ACCESS);
    
    CONDUIT_CHECK_HDF5_ERROR(h5_fa_props,
                             "Failed to create H5P_FILE_ACCESS "
                             << " property list");
    
    
    herr_t h5_status = H5Pset_libver_bounds(h5_fa_props,
                                            H5F_LIBVER_LATEST,
                                            H5F_LIBVER_LATEST);

    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Failed to set libver options for "
                             << "property list " << h5_fa_props);
    return h5_fa_props;
}

//---------------------------------------------------------------------------//
hid_t
create_hdf5_file_create_plist()
{
    // create property list and set it to preserve creation order
    hid_t h5_fc_props = H5Pcreate(H5P_FILE_CREATE);

    CONDUIT_CHECK_HDF5_ERROR(h5_fc_props,
                             "Failed to create H5P_FILE_CREATE "
                             << " property list");

    herr_t h5_status = H5Pset_link_creation_order(h5_fc_props, 
            ( H5P_CRT_ORDER_TRACKED |  H5P_CRT_ORDER_INDEXED) );

    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Failed to set creation order options for "
                             << "property list " << h5_fc_props);
    return h5_fc_props;
}

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// Public interface: Write Methods
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
hid_t
hdf5_open_file_for_write(const std::string &file_path)
{
    hid_t h5_fc_plist = create_hdf5_file_create_plist();
    hid_t h5_fa_plist = create_hdf5_file_access_plist();

    // open the hdf5 file for writing
    hid_t h5_file_id = H5Fcreate(file_path.c_str(),
                                 H5F_ACC_TRUNC,
                                 h5_fc_plist,
                                 h5_fa_plist);

    CONDUIT_CHECK_HDF5_ERROR(h5_file_id,
                             "Error opening HDF5 file for writing: " 
                             << file_path);

    CONDUIT_CHECK_HDF5_ERROR(H5Pclose(h5_fc_plist),
                             "Failed to close HDF5 H5P_GROUP_CREATE "
                             << "property list: " << h5_fc_plist);

    CONDUIT_CHECK_HDF5_ERROR(H5Pclose(h5_fa_plist),
                             "Failed to close HDF5 H5P_FILE_ACCESS "
                             << "property list: " << h5_fa_plist);
    
    return h5_file_id;
}

//---------------------------------------------------------------------------//
void 
hdf5_write(const  Node &node,
           const std::string &path)
{
    // check for ":" split
    std::string file_path;
    std::string hdf5_path;

    io::split_path(path,
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
    // open the hdf5 file for writing
    hid_t h5_file_id = hdf5_open_file_for_write(file_path);

    hdf5_write(node,
               h5_file_id,
               hdf5_path);

    // close the hdf5 file
    CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                             "Error closing HDF5 file: " << file_path);
}


//---------------------------------------------------------------------------//
void
hdf5_write(const Node &node,
           hid_t hdf5_id,
           const std::string &hdf5_path)
{
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;

    // TODO: we only want to support abs paths if hdf5_id is a file
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
    // checks and write methods handle node paths easily handle this case.
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
                                                          "",
                                                          hdf5_id))
    {
        // write if we are compat
        write_conduit_node_to_hdf5_tree(n,"",hdf5_id);
    }
    else
    {
        CONDUIT_ERROR("Failed to write node, existing HDF5 tree is "
                      "incompatible with the Conduit Node.")
    }

    // restore hdf5 error stack
}


//---------------------------------------------------------------------------//
void
hdf5_write(const Node &node,
           hid_t hdf5_id)
{
    // disable hdf5 error stack
    // TODO: we may only need to use this in an outer level variant
    // of check_if_conduit_node_is_compatible_with_hdf5_tree
    HDF5ErrorStackSupressor supress_hdf5_errors;
    
    // check compat
    if(check_if_conduit_node_is_compatible_with_hdf5_tree(node,
                                                          "",
                                                          hdf5_id))
    {
        // write if we are compat
        write_conduit_node_to_hdf5_tree(node,
                                        "",
                                        hdf5_id);
    }
    else
    {
        CONDUIT_ERROR("Failed to write node, existing HDF5 tree is "
                      << "incompatible with the Conduit Node.")
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
hid_t
hdf5_open_file_for_read(const std::string &file_path)
{
    hid_t h5_fa_plist = create_hdf5_file_access_plist();
    
    // open the hdf5 file for reading
    hid_t h5_file_id = H5Fopen(file_path.c_str(),
                               H5F_ACC_RDONLY,
                               h5_fa_plist);

    CONDUIT_CHECK_HDF5_ERROR(h5_file_id,
                             "Error opening HDF5 file for reading: " 
                              << file_path);

    CONDUIT_CHECK_HDF5_ERROR(H5Pclose(h5_fa_plist),
                             "Failed to close HDF5 H5P_FILE_ACCESS "
                             << "property list: " << h5_fa_plist);
    
    return h5_file_id;
}

//---------------------------------------------------------------------------//
void
hdf5_read(const std::string &path,
          Node &node)
{
    // check for ":" split
    std::string file_path;
    std::string hdf5_path;
    
    io::split_path(path,
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
    // open the hdf5 file for reading
    hid_t h5_file_id = hdf5_open_file_for_read(file_path);

    hdf5_read(h5_file_id,
              hdf5_path,
              node);
    
    // close the hdf5 file
    CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                             "Error closing HDF5 file: " << file_path);
}
//---------------------------------------------------------------------------//
void
hdf5_read(hid_t hdf5_id,
          const std::string &hdf5_path,
          Node &dest)
{
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;
    
    // get hdf5 object at path, then call read_hdf5_tree_into_conduit_node
    hid_t h5_child_obj  = H5Oopen(hdf5_id,
                                  hdf5_path.c_str(),
                                  H5P_DEFAULT);
    
    CONDUIT_CHECK_HDF5_ERROR(h5_child_obj,
                             "Failed to fetch HDF5 object from: "
                             << hdf5_id << ":" << hdf5_path);

    read_hdf5_tree_into_conduit_node(h5_child_obj,
                                     hdf5_path,
                                     dest);
    
    CONDUIT_CHECK_HDF5_ERROR(H5Oclose(h5_child_obj),
                             "Failed to close HDF5 Object: "
                             << h5_child_obj);
    
    // enable hdf5 error stack
}


//---------------------------------------------------------------------------//
void
hdf5_read(hid_t hdf5_id,
          Node &dest)
{
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;
    
    read_hdf5_tree_into_conduit_node(hdf5_id,
                                     "",
                                     dest);
    
    // enable hdf5 error stack
}


//---------------------------------------------------------------------------//
bool
hdf5_has_path(hid_t hdf5_id,
              const std::string &hdf5_path)
{
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;
    
    int res = H5Lexists(hdf5_id,hdf5_path.c_str(),H5P_DEFAULT);
    
    
    // H5Lexists returns:
    //  a positive value if the link exists.
    //
    //  0 if it doesn't exist
    //
    //  a negative # in some cases when it doesn't exist, and in some cases
    //    where there is an error. 
    // For our cases, we treat 0 and negative as does not exist. 

    return (res > 0);
    // enable hdf5 error stack
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
