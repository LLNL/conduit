//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
/// file: conduit_relay_io_hdf5.cpp
///
//-----------------------------------------------------------------------------

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    #include "conduit_relay_mpi_io_hdf5.hpp"
#else
    #include "conduit_relay_io_hdf5.hpp"
#endif

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
/// The CONDUIT_CHECK_HDF5_ERROR macro is used to check error codes from HDF5.
//-----------------------------------------------------------------------------
#define CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH( hdf5_err, hdf5_obj_id, ref_path, msg ) \
{                                                                         \
    if( hdf5_err < 0 )                                                    \
    {                                                                     \
        /* Try to find the file system path from hdf5_obj */              \
        ssize_t hdf5_f_sz = H5Fget_name(hdf5_obj_id, NULL, 0 );           \
        std::ostringstream hdf5_err_oss;                                  \
        hdf5_err_oss << "HDF5 Error (error code: "                        \
            <<  hdf5_err                                                  \
            <<  ", reference path: \"";                                   \
        if(hdf5_f_sz > 0)                                                 \
        {                                                                 \
            std::vector<char>hdf5_f_buff(hdf5_f_sz+1, 0);                 \
            H5Fget_name(hdf5_obj_id, &hdf5_f_buff[0], hdf5_f_sz+1);       \
            hdf5_err_oss << std::string(&hdf5_f_buff[0]) << ":";          \
        }                                                                 \
            hdf5_err_oss <<  ref_path << "\""                             \
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

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
//-----------------------------------------------------------------------------
// -- begin conduit::relay::mpi --
//-----------------------------------------------------------------------------
namespace mpi
{
#endif

//-----------------------------------------------------------------------------
// -- begin conduit::relay::<mpi>::io --
//-----------------------------------------------------------------------------
namespace io
{


static std::string conduit_hdf5_list_attr_name = "__conduit_list";

    
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
// helpers for finding hdf5 object filename and constructing ref paths for 
// errors
//-----------------------------------------------------------------------------
void  hdf5_filename_from_hdf5_obj_id(hid_t hdf5_id,
                                     std::string &result);

void  hdf5_ref_path_with_filename(hid_t hdf5_id,
                                  const std::string &ref_path,
                                  std::string &result);

//-----------------------------------------------------------------------------
// helpers for data type conversions
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// helpers for checking if compatible 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// if incompatible, incompat_details contains human readable details
// about why
//-----------------------------------------------------------------------------
bool check_if_conduit_leaf_is_compatible_with_hdf5_obj(const DataType &dtype,
                                                  const std::string &ref_path,
                                                                hid_t hdf5_id,
                                                std::string &incompat_details);

//-----------------------------------------------------------------------------
bool check_if_conduit_object_is_compatible_with_hdf5_tree(const Node &node,
                                                const std::string &ref_path,
                                                             hid_t hdf5_id,
                                                std::string &incompat_details);

//-----------------------------------------------------------------------------
bool check_if_conduit_node_is_compatible_with_hdf5_tree(const Node &node,
                                              const std::string &ref_path,
                                                            hid_t hdf5_id,
                                                std::string &incompat_details);

//-----------------------------------------------------------------------------
bool check_if_conduit_list_is_compatible_with_hdf5_tree(const Node &node,
                                                   const std::string &ref_path,
                                                        hid_t hdf5_id,
                                                std::string &incompat_details);

//-----------------------------------------------------------------------------
bool check_if_hdf5_group_has_conduit_list_attribute(hid_t hdf5_group_id);

//-----------------------------------------------------------------------------
// helpers for writing
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
hid_t create_hdf5_dataset_for_conduit_leaf(const DataType &dt,
                                           const std::string &ref_path,
                                           hid_t hdf5_group_id,
                                           const std::string &hdf5_dset_name);

//-----------------------------------------------------------------------------
hid_t create_hdf5_group_for_conduit_node(const Node &node,
                                         const std::string &ref_path,
                                         hid_t hdf5_parent_group_id,
                                         const std::string &hdf5_child_group_name);

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
void  write_conduit_node_children_to_hdf5_group(const Node &node,
                                                const std::string &ref_path,
                                                hid_t hdf5_group_id);

//-----------------------------------------------------------------------------
void write_conduit_hdf5_list_attribute(hid_t hdf5_group_id,
                                       const std::string &ref_path);

//-----------------------------------------------------------------------------
void remove_conduit_hdf5_list_attribute(hid_t hdf5_group_id,
                                        const std::string &ref_path);

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
//  conduit_dtype_to_hdf5_dtype_cleanup
//  hdf5_dtype_to_conduit_dtype
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
hid_t
conduit_dtype_to_hdf5_dtype(const DataType &dt,
                            const std::string &ref_path)
{
    hid_t res = -1;

    // // This code path enables writing strings in a way that is friendlier 
    // // to hdf5 command line tools like hd5dump and h5ls. However
    // // using this path we *cannot* compress that string data, so
    // // is currently disabled
    //
    // if(dt.is_string())
    // {
    //
    //     // modify the default hdf5 type to include string length info,
    //     // so hdf5 tools display the string contents in a human friendly way
    //
    //     // create a copy of the default type
    //     res = H5Tcopy(H5T_C_S1);
    //     CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(res,
    //                                            ref_path,
    //                                     "Failed to copy HDF5 type for string");
    //
    //     // set the size
    //     CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(
    //                                 H5Tset_size(res,
    //                                             // string size + null
    //                                             dt.number_of_elements()),
    //                                 ref_path,
    //                                 "Failed to set size in HDF5 string type");
    //
    //     // set term
    //     CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(
    //                                 H5Tset_strpad(res, H5T_STR_NULLTERM),
    //                                 ref_path,
    //                                 "Failed to set strpad in HDF5 string type");
    // }

    // strings are special, check for them first
    if( dt.is_string() )
    {
        res = H5T_C_S1;
    }
    // next check endianness
    else if(dt.is_little_endian()) // we know we are little endian
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
            
            case DataType::CHAR8_STR_ID: 
                CONDUIT_HDF5_ERROR(ref_path,
                              "conduit::DataType to HDF5 Leaf DataType "
                              << "Conversion:"
                              << dt.to_json() 
                              << " needs to be handled with string logic");
                        break;
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
            
            case DataType::CHAR8_STR_ID: 
                CONDUIT_HDF5_ERROR(ref_path,
                              "conduit::DataType to HDF5 Leaf DataType "
                              << "Conversion:"
                              << dt.to_json() 
                              << " needs to be handled with string logic");
                        break;
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
// cleanup conduit created hdf5 dtype
// (effectively a noop, except for the string case)
// TODO: This could be a macro ... ?
//-----------------------------------------------------------------------------
void
conduit_dtype_to_hdf5_dtype_cleanup(hid_t hdf5_dtype_id,
                            const std::string &ref_path)
{
    // NOTE: This cleanup won't be triggered when we use thee
    // based H5T_C_S1 with a data space that encodes # of elements
    // (Our current path, given our logic to encode string size in the 
    //  hdf5 type is disabled )
    
    // if this is a string using a custom type we need to cleanup 
    // the conduit_dtype_to_hdf5_dtype result
    if( (! H5Tequal(hdf5_dtype_id, H5T_C_S1) ) && 
        (H5Tget_class(hdf5_dtype_id) == H5T_STRING ) )
    {
        CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(H5Tclose(hdf5_dtype_id),
                                                        ref_path,
                                    "Failed to close HDF5 string Type "
                                                        << hdf5_dtype_id);
    }
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
        // string as array case (old way of writing)
        res = DataType::char8_str(num_elems);
    }
    // extended string reps
    else if( H5Tget_class(hdf5_dtype_id) == H5T_STRING )
    {
        // for strings of this type, the length 
        // is encoded in the hdf5 type not the hdf5 data space
        index_t hdf5_strlen = H5Tget_size(hdf5_dtype_id);
        // check for variable type first
        if( H5Tis_variable_str(hdf5_dtype_id) )
        {
            res = DataType::char8_str(-1);
        }
        else
        {
            res = DataType::char8_str(hdf5_strlen);
        }
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

    return res;
}

//---------------------------------------------------------------------------//
// General Helper
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
hdf5_filename_from_hdf5_obj_id(hid_t hdf5_id,
                               std::string &hdf5_filename)
{
    hdf5_filename.clear();
    ssize_t hdf5_f_sz = H5Fget_name(hdf5_id, NULL, 0 );
    if(hdf5_f_sz > 0)
    {
        std::vector<char>hdf5_f_buff(hdf5_f_sz+1, 0);
        H5Fget_name(hdf5_id, &hdf5_f_buff[0], hdf5_f_sz+1);
        hdf5_filename = std::string(&hdf5_f_buff[0]);
    }
}

//---------------------------------------------------------------------------//
void
hdf5_ref_path_with_filename(hid_t hdf5_id,
                            const std::string &ref_path,
                            std::string &result)
{
    hdf5_filename_from_hdf5_obj_id(hdf5_id, result);
    if(result.empty())
    {
        result = ref_path;
    }
    else
    {
        if(!ref_path.empty())
        {
            result +=  ":" + ref_path;
        }
    }
}


//---------------------------------------------------------------------------//
// Write Helpers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
bool
check_if_conduit_leaf_is_compatible_with_hdf5_obj(const DataType &dtype,
                                                  const std::string &ref_path,
                                                  hid_t hdf5_id,
                                                  std::string &incompat_details)
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
            std::ostringstream oss;
            oss << "Conduit Node (leaf) at path '" << ref_path << "'"
                << " is not compatible with given HDF5 Dataset at path"
                << " '" << ref_path << "'"
                << "\nHDF5 dataset has a H5S_NULL Dataspace which"
                << " only compatible with an empty Conduit Node";

            incompat_details = oss.str();

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
    
            
            // string case is special, check it first
            
            // if the dataset in the file is a custom string type
            // check the type's size vs the # of elements
            if(   ( ! H5Tequal(h5_test_dtype, H5T_C_S1) &&
                  ( H5Tget_class(h5_test_dtype) == H5T_STRING ) &&
                  ( H5Tget_class(h5_dtype) == H5T_STRING ) ) &&  
                 // if not shorted out, we have a string w/ custom type
                 // check length to see if compat
                 // note: both hdf5 and conduit dtypes include null term in string size
                 (dtype.number_of_elements() !=  (index_t)H5Tget_size(h5_test_dtype) ) )
            {
                std::ostringstream oss;
                oss << "Conduit Node (string leaf) at path '" << ref_path << "'"
                    << " is not compatible with given HDF5 Dataset at path"
                    << " '" << ref_path << "'"
                    << "\nConduit leaf String Node length (" 
                    << dtype.number_of_elements() << ")"
                    << " != HDF5 Dataset size (" << H5Tget_size(h5_test_dtype) << ")";

                incompat_details = oss.str();

                res = false;
            }
            else if( ! ( (H5Tequal(h5_dtype, h5_test_dtype) > 0) && 
                         (dtype.number_of_elements() ==  h5_test_num_ele) ) )
            {
                std::ostringstream oss;
                oss << "Conduit Node (leaf) at path '" << ref_path << "'"
                    << " is not compatible with given HDF5 Dataset at path"
                    << " '" << ref_path << "'"
                    << "\nConduit leaf Node number of elements (" 
                    << dtype.number_of_elements() << ")"
                    << " != HDF5 Dataset size (" << H5Tget_size(h5_test_dtype) << ")";

                incompat_details = oss.str();

                res = false;
            }

            CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Tclose(h5_test_dtype),
                                                            hdf5_id,
                                                            ref_path,
                                     "Failed to close HDF5 Datatype "
                                     << h5_test_dtype);
            // clean up when necessary
            conduit_dtype_to_hdf5_dtype_cleanup(h5_dtype);
        }

        CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Sclose(h5_test_dspace),
                                                        hdf5_id,
                                                        ref_path,
                         "Failed to close HDF5 Dataspace " << h5_test_dspace);
    }
    else
    {
        // bad id, or not a dataset
        std::ostringstream oss;
        oss << "Conduit Node (leaf) at path '" << ref_path << "'"
            << " is not compatible with given HDF5 Dataset at path"
            << "'" << ref_path << "'"
            << "\nConduit leaf vs HDF5 Dataset: Bad HDF5 Leaf ID"
            << " or HDF5 ID is not a HDF5 Group";

        incompat_details = oss.str();
        res = false;
    }

    return res;
}

//---------------------------------------------------------------------------//
bool
check_if_conduit_object_is_compatible_with_hdf5_tree(const Node &node,
                                                     const std::string &ref_path,
                                                     hid_t hdf5_id,
                                                  std::string &incompat_details)
{
    bool res = true;
    
    // make sure we have a group ... 
    
    H5O_info_t h5_obj_info;
    herr_t h5_status = H5Oget_info(hdf5_id, &h5_obj_info);
    
    // make sure it is a group ...
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
                                                                  h5_child_obj,
                                                                  incompat_details);
            
                CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Oclose(h5_child_obj),
                                                                hdf5_id,
                                                                ref_path,
                             "Failed to close HDF5 Object: " << h5_child_obj);
            }
            // no child exists with this name,  we are ok (it can be created 
            // to match) check the next child
        }
    }
    else // bad id or not a group
    {
        std::ostringstream oss;
        oss << "Conduit Node (object) at path '" << ref_path << "'"
            << " is not compatible with given HDF5 tree at path"
            << "'" << ref_path << "'"
            << "\nConduit Object vs HDF5 Group: Bad HDF5 Group ID "
            << "or HDF5 ID is not a HDF5 Group";

        incompat_details = oss.str();

        res = false;
    }
    
    return res;
}

//---------------------------------------------------------------------------//
bool
check_if_conduit_list_is_compatible_with_hdf5_tree(const Node &node,
                                                   const std::string &ref_path,
                                                   hid_t hdf5_id,
                                                   std::string &incompat_details)
{
    bool res = true;
    
    // make sure we have a group ... 
    
    H5O_info_t h5_obj_info;
    herr_t h5_status = H5Oget_info(hdf5_id, &h5_obj_info);

    // make sure it is a group ...
    if( CONDUIT_HDF5_STATUS_OK(h5_status) &&
        (h5_obj_info.type == H5O_TYPE_GROUP) )
    {
        // TODO: should we force the group should have our att that signals a 
        //       list ?
        
        // if(!check_if_hdf5_group_has_conduit_list_attribute(hdf5_id,ref_path))
        // {
        //     // we don't have a list
        //         std::ostringstream oss;
        //         oss << "Conduit Node (list) at path '" << ref_path << "'"
        //             << " is not compatible with given HDF5 tree at path"
        //             << " '" << ref_path << "'"
        //             << "\nHDF5 Group missing '" << conduit_hdf5_list_attr_name
        //             << "' attribute";
        //         incompat_details = oss.str();
        //         res = false;
        //         return res;
        // }

        NodeConstIterator itr = node.children();

        // call on each child with expanded path
        while(itr.has_next() && res)
        {

            const Node &child = itr.next();
            // check if the HDF5 group has ordered children
            // that are compat with our node.

            // for a list, we check the group's children by index (not name)

            hid_t h5_child_obj = H5Oopen_by_idx(hdf5_id, ".",
                                                H5_INDEX_CRT_ORDER,
                                                H5_ITER_INC,
                                                itr.index(),
                                                H5P_DEFAULT);
        
            std::string chld_ref_path = join_ref_paths(ref_path,itr.name());
            if( CONDUIT_HDF5_VALID_ID(h5_child_obj) )
            {
                // if a child does exist, we need to make sure the child is 
                // compatible with the conduit node
                res = check_if_conduit_node_is_compatible_with_hdf5_tree(child,
                                                                  chld_ref_path,
                                                                  h5_child_obj,
                                                                  incompat_details);
            
                CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Oclose(h5_child_obj),
                                                                hdf5_id,
                                                                ref_path,
                             "Failed to close HDF5 Object: " << h5_child_obj);
            }
            // no child exists with this index,  we are ok (it can be created 
            // to match)
        }
    }
    else // bad id or not a group
    {
        std::ostringstream oss;
        oss << "Conduit Node (list) at path '" << ref_path << "'"
            << " is not compatible with given HDF5 tree at path"
            << "'" << ref_path << "'"
            << "\nConduit List vs HDF5 Group: Bad HDF5 Group ID "
            << "or HDF5 ID is not a HDF5 Group";

        incompat_details = oss.str();

        res = false;
    }
    
    return res;
}



//---------------------------------------------------------------------------//
bool
check_if_conduit_node_is_compatible_with_hdf5_tree(const Node &node,
                                                   const std::string &ref_path,
                                                   hid_t hdf5_id,
                                                   std::string &incompat_details)
{
    bool res = true;

    DataType dt = node.dtype();
    // check for leaf or group
    if(dt.is_number() || dt.is_string())
    {
        res = check_if_conduit_leaf_is_compatible_with_hdf5_obj(dt,
                                                                ref_path,
                                                                hdf5_id,
                                                                incompat_details);
    }
    else if(dt.is_object())
    {
        res = check_if_conduit_object_is_compatible_with_hdf5_tree(node,
                                                                   ref_path,
                                                                   hdf5_id,
                                                                   incompat_details);
    }
    else if(dt.is_list())
    {
        res = check_if_conduit_list_is_compatible_with_hdf5_tree(node,
                                                                 ref_path,
                                                                 hdf5_id,
                                                                 incompat_details);
    }
    else // not supported
    {
        std::ostringstream oss;
        oss << "Conduit Node at path '" << ref_path << "'"
            << " has an unsupported dtype (" << dt.name() << ")"
            << " for HDF5 i/o and cannot be written to HDF5 path"
            << " '" << ref_path  << "'";

        incompat_details = oss.str();
        res = false;
    }

    return res;
}

//---------------------------------------------------------------------------//
bool
check_if_hdf5_group_has_conduit_list_attribute(hid_t hdf5_group_id,
                                               const std::string &ref_path)
{
    htri_t h5_att_status = H5Aexists_by_name(hdf5_group_id, ".",
                                          conduit_hdf5_list_attr_name.c_str(),
                                             H5P_DEFAULT);
    //
    // https://support.hdfgroup.org/HDF5/doc/RM/RM_H5A.html#Annot-ExistsByName
    // > 0 exists, 0 doesn't exist, < 0 error
    //

    CONDUIT_CHECK_HDF5_ERROR_WITH_REF_PATH(h5_att_status,
                                           ref_path,
                                           "Failed call to H5Aexists_by_name"
                                           << " to check for '"
                                           << conduit_hdf5_list_attr_name
                                           << "' attribute of HDF5 Group ID "
                                           << " " << hdf5_group_id);
    return h5_att_status > 0;
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

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_cprops_id,
                                                    hdf5_group_id,
                                                    ref_path,

                                         "Failed to create HDF5 property list");
    hid_t h5_dspace_id = -1;
    
    // string a scalar with size embedded in type is disabled
    // b/c this path undermines compression 
    // if(dtype.is_string())
    // {
    //     h5_dspace_id = H5Screate(H5S_SCALAR);
    // }
    // else
    // {
    //     h5_dspace_id = H5Screate_simple(1,
    //                                     &num_eles,
    //                                     NULL);
    // }

    h5_dspace_id = H5Screate_simple(1,
                                    &num_eles,
                                    NULL);


    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_dspace_id,
                                                    hdf5_group_id,
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

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(res,
                                                    hdf5_group_id,
                                                    ref_path,
                                           "Failed to create HDF5 Dataset " 
                                           << hdf5_group_id << " " 
                                           << hdf5_dset_name);

    // cleanup if custom data type was used
    conduit_dtype_to_hdf5_dtype_cleanup(h5_dtype);

    // close plist used for compression
    if(h5_cprops_id != H5P_DEFAULT)
    {
        CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Pclose(h5_cprops_id),
                                                        hdf5_group_id,
                                                        ref_path,
                                           "Failed to close HDF5 compression "
                                           "property list " 
                                                       << h5_cprops_id);
    }

    // close our dataspace
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Sclose(h5_dspace_id),
                                                    hdf5_group_id,
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
    
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_dspace_id,
                                                    hdf5_group_id,
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

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(res,
                                                    hdf5_group_id,
                                                    ref_path,
                                           "Failed to create HDF5 Dataset " 
                                           << hdf5_group_id 
                                           << " " << hdf5_dset_name);
    // close our datatype
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Tclose(h5_dtype_id),
                                                    hdf5_group_id,
                                                    ref_path,
                                           "Failed to close HDF5 Datatype");
    // close our dataspace
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Sclose(h5_dspace_id),
                                                    hdf5_group_id,
                                                    ref_path,
                                           "Failed to close HDF5 Dataspace " 
                                           << h5_dspace_id);

    return res;
}

//-----------------------------------------------------------------------------
hid_t
create_hdf5_group_for_conduit_node(const Node &node,
                                   const std::string &ref_path,
                                   hid_t hdf5_parent_group_id,
                                   const std::string &hdf5_new_group_name)
{
    hid_t h5_gc_plist = H5Pcreate(H5P_GROUP_CREATE);

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_gc_plist,
                                                    hdf5_parent_group_id,
                                                    ref_path,
                     "Failed to create H5P_GROUP_CREATE property "
                     << " list");

    // track creation order
    herr_t h5_status = H5Pset_link_creation_order(h5_gc_plist, 
            ( H5P_CRT_ORDER_TRACKED |  H5P_CRT_ORDER_INDEXED) );

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_status,
                                                    hdf5_parent_group_id,
                                                    ref_path,
                     "Failed to set group link creation property");

    // prefer compact group storage 
    // https://support.hdfgroup.org/HDF5/doc/RM/RM_H5G.html#Group-GroupStyles
    h5_status = H5Pset_link_phase_change(h5_gc_plist,
                                         32,  // max for compact storage
                                         32); // min for dense storage

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_status,
                                                    hdf5_parent_group_id,
                                                    ref_path,
                "Failed to set group link phase change property ");

    // calc hints for meta data about link names
    NodeConstIterator chld_itr = node.children();

    index_t chld_names_avg_size = 0;
    index_t num_children = node.number_of_children();

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
                                     // number of children
                                     (unsigned int)num_children,
                                     // est name size
                                     (unsigned int)chld_names_avg_size);

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_status,
                                                    hdf5_parent_group_id,
                                                    ref_path,
                "Failed to set group est link info property ");

    hid_t h5_child_id = H5Gcreate(hdf5_parent_group_id,
                                  hdf5_new_group_name.c_str(),
                                  H5P_DEFAULT,
                                  h5_gc_plist,
                                  H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_child_id,
                                                    hdf5_parent_group_id,
                                                    ref_path,
                          "Failed to create HDF5 Group "
                          << " parent: " << hdf5_parent_group_id
                          << " name: "   << hdf5_new_group_name);

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Pclose(h5_gc_plist),
                                                    hdf5_parent_group_id,
                                                    ref_path,
                         "Failed to close HDF5 H5P_GROUP_CREATE "
                         << "property list: " 
                         << h5_gc_plist);

    return h5_child_id;
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
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_status,
                                                    hdf5_dset_id,
                                                    ref_path,
                                           "Failed to write to HDF5 Dataset "
                                           << hdf5_dset_id);

    conduit_dtype_to_hdf5_dtype_cleanup(h5_dtype_id);
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

        CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_child_id,
                                                        hdf5_group_id,
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

        CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_child_id,
                                                        hdf5_group_id,
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
    
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Dclose(h5_child_id),
                                                    hdf5_group_id,
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

        CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_child_id,
                                                        hdf5_group_id,
                                                        ref_path,
                                               "Failed to create HDF5 Dataset "
                                               << " parent: " << hdf5_group_id
                                               << " name: "   << hdf5_dset_name);

        CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Dclose(h5_child_id), 
                                                        hdf5_group_id,
                                                        ref_path,
                                              "Failed to close HDF5 Dataset: "
                                              << h5_child_id);
    }
    

}

//---------------------------------------------------------------------------//
void
setup_hdf5_group_atts_for_conduit_node(const Node &node,
                                       const std::string &ref_path,
                                       hid_t hdf5_group_id)
{
    bool has_list_attr = check_if_hdf5_group_has_conduit_list_attribute(hdf5_group_id,
                                                                        ref_path);

    if( !has_list_attr && node.dtype().is_list() )
    {
        write_conduit_hdf5_list_attribute(hdf5_group_id,
                                          ref_path);
    }

    if( has_list_attr && node.dtype().is_object() )
    {
        std::cout << " remove_conduit_hdf5_list_attribute " << std::endl;
        remove_conduit_hdf5_list_attribute(hdf5_group_id,
                                           ref_path);
    }
}


//---------------------------------------------------------------------------//
// assume this is called only if we know the hdf5 state is compatible 
//---------------------------------------------------------------------------//
void
write_conduit_node_children_to_hdf5_group(const Node &node,
                                          const std::string &ref_path,
                                          hid_t hdf5_group_id)
{
    // make sure our special atts are setup correctly
    setup_hdf5_group_atts_for_conduit_node(node,
                                           ref_path,
                                           hdf5_group_id);

    NodeConstIterator itr = node.children();

    // call on each child with expanded path
    while(itr.has_next())
    {
        const Node &child = itr.next();
        DataType dt = child.dtype();
        std::string child_name = itr.name();

        if(dt.is_number() || dt. is_string())
        {
            write_conduit_leaf_to_hdf5_group(child,
                                             ref_path,
                                             hdf5_group_id,
                                             child_name.c_str());
        }
        else if(dt.is_empty())
        {
            // if we have an empty node, it will become
            // a dataset with an null shape
            write_conduit_empty_to_hdf5_group(hdf5_group_id,
                                              ref_path,
                                              child_name.c_str());
        }
        else if( dt.is_object() || dt.is_list() )
        {
            // check if the HDF5 group has child with same name
            // as the node's child
            H5O_info_t h5_obj_info;
            herr_t h5_info_status =  H5Oget_info_by_name(hdf5_group_id,
                                                         child_name.c_str(),
                                                         &h5_obj_info,
                                                         H5P_DEFAULT);
            
            hid_t h5_child_id = -1;
            
            if( CONDUIT_HDF5_STATUS_OK(h5_info_status) )
            {
                // if the hdf5 group exists, open it
                h5_child_id = H5Gopen(hdf5_group_id,
                                      child_name.c_str(),
                                      H5P_DEFAULT);

                CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_child_id,
                                                                hdf5_group_id,
                                                                ref_path,
                                             "Failed to open HDF5 Group "
                                             << " parent: " << hdf5_group_id
                                             << " name: "   << itr.name());
            }
            else
            {
                // if the hdf5 group doesn't exist, we need to create it
                h5_child_id = create_hdf5_group_for_conduit_node(child,
                                                                 ref_path,
                                                                 hdf5_group_id,
                                                                 child_name);

            }

            // traverse 
            write_conduit_node_children_to_hdf5_group(child,
                                                      ref_path,
                                                      h5_child_id);

            CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Gclose(h5_child_id),
                                                            hdf5_group_id,
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
    if( dt.is_number() || dt.is_string() )
    {
        write_conduit_leaf_to_hdf5_dataset(node,
                                           ref_path,
                                           hdf5_id);
    }
    else if( dt.is_object() || dt.is_list() )
    {
        write_conduit_node_children_to_hdf5_group(node,
                                                  ref_path,
                                                  hdf5_id);
    }
    else // not supported
    {
        std::string hdf5_err_ref_path;
        hdf5_ref_path_with_filename(hdf5_id, ref_path, hdf5_err_ref_path);
        CONDUIT_HDF5_ERROR(hdf5_err_ref_path,
                   "HDF5 write doesn't support EMPTY_ID nodes.");
    }
}
//---------------------------------------------------------------------------//
void
write_conduit_hdf5_list_attribute(hid_t hdf5_group_id,
                                  const std::string &ref_path)
{
    //  We really just use the presence of the attribute, we don't need
    //  data associated with it. 
    //
    //  I tried to write a null att (null hdf5 dt, etc) but that didn't work.
    //  H5Awrite fails with message about null data. I could't find any 
    //  examples that demoed this either -- it may not be supported.
    //
    //  So, we write a single meaningless int as the attribute data.
    //  Perhaps someone could breath meaning into this int in the future
    //  or find a way to eliminate it.

    int att_value = 1;
    
    hid_t h5_dspace_id = H5Screate(H5S_SCALAR);
    
    hid_t h5_attr_id  = H5Acreate(hdf5_group_id,
                                  conduit_hdf5_list_attr_name.c_str(),
                                  H5T_NATIVE_INT,
                                  h5_dspace_id,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_attr_id,
                                                    hdf5_group_id,
                                                    ref_path,
                                           "Failed to create HDF5 Attribute " 
                                           << hdf5_group_id 
                                           << " "
                                           << conduit_hdf5_list_attr_name.c_str());

    
    hid_t h5_status = H5Awrite(h5_attr_id,
                               H5T_NATIVE_INT,
                               &att_value);
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_status,
                                                    hdf5_group_id,
                                                    ref_path,
                                           "Failed to write HDF5 Attribute " 
                                           << hdf5_group_id 
                                           << " "
                                           << conduit_hdf5_list_attr_name.c_str());

    // close our dataspace
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Sclose(h5_dspace_id),
                                                    hdf5_group_id,
                                                    ref_path,
                                           "Failed to close HDF5 Dataspace " 
                                           << h5_dspace_id);

    // close our attribute
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Aclose(h5_attr_id),
                                                    hdf5_group_id,
                                                    ref_path,
                                           "Failed to close HDF5 Attribute " 
                                           << h5_attr_id);
}

//---------------------------------------------------------------------------//
void
remove_conduit_hdf5_list_attribute(hid_t hdf5_group_id,
                                   const std::string &ref_path)
{
    // cleanup group attached att, just in case a group changes roles
    // and is still compatible otherwise
    herr_t h5_status = H5Adelete(hdf5_group_id,
                                 conduit_hdf5_list_attr_name.c_str());

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_status,
                                                    hdf5_group_id,
                                                    ref_path,
                                           "Failed to remove HDF5 Attribute " 
                                           << hdf5_group_id 
                                           << " "
                                           << conduit_hdf5_list_attr_name.c_str());
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
Node *
h5l_iterate_traverse_op_func_get_child(Node &node,
                                       const std::string &hdf5_path)
{
    Node *chld_node_ptr = NULL;
    // if( h5_od->node->dtype().is_object() )
    if( node.dtype().is_object() )
    {
        // execute traversal for this group
        chld_node_ptr = &node.fetch(hdf5_path);
    }
    else if( node.dtype().is_list() )
    {
        // we need the child index, use name to index for now
        // not sure if it is possible to get iteration index
        // from h5literate
        
        // Either the child already exists in conduit 
        // (compat case), or we need to append to add
        // a new child

        // parse index for this list entry
        std::istringstream iss(hdf5_path);
        int child_idx = -1;
        iss >> child_idx;

        if(node.number_of_children() <= child_idx )
        {
            node.append();             
        }

        chld_node_ptr = &node.child(child_idx);
    }
    else
    {
        // TODO Error: We should not land here, H5Literate should
        // only be called on groups, which will correspond
        // to either objects or lists
    }
    
    return chld_node_ptr;
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

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_status,
                                                    hdf5_id,
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
                CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_group_id,
                                                                hdf5_id,
                                                                h5_od->ref_path,
                                                       "Error opening HDF5 "
                                                       << "Group: " 
                                                       << " parent: "
                                                       << hdf5_id 
                                                       << " path:"
                                                       << hdf5_path);

                Node *chld_node_ptr = h5l_iterate_traverse_op_func_get_child(
                                                   *h5_od->node,
                                                   std::string(hdf5_path));

                read_hdf5_group_into_conduit_node(h5_group_id,
                                                  chld_ref_path,
                                                  *chld_node_ptr);

                // close the group
                CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Gclose(h5_group_id),
                                                                hdf5_id,
                                                                h5_od->ref_path,
                                                       "Error closing HDF5 "
                                                       << "Group: "
                                                       << h5_group_id);
            }
            break;
        }
        case H5O_TYPE_DATASET:
        {
            Node *chld_node_ptr = h5l_iterate_traverse_op_func_get_child(
                                                   *h5_od->node,
                                                   std::string(hdf5_path));

            // open hdf5 dataset at path
            hid_t h5_dset_id = H5Dopen(hdf5_id,
                                       hdf5_path,
                                       H5P_DEFAULT);

            CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_dset_id,
                                                            hdf5_id,
                                                            h5_od->ref_path,
                                                   "Error opening HDF5 "
                                                   << " Dataset: " 
                                                   << " parent: "
                                                   << hdf5_id 
                                                   << " path:"
                                                   << hdf5_path);

            read_hdf5_dataset_into_conduit_node(h5_dset_id,
                                                chld_ref_path,
                                                *chld_node_ptr);
            
            // close the dataset
            CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Dclose(h5_dset_id),
                                                            hdf5_id,
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
    // get info, we need to get the obj addr for cycle tracking
    H5O_info_t h5_info_buf;
    herr_t h5_status = H5Oget_info(hdf5_group_id,
                                   &h5_info_buf);

    // Check if this is a list or an object case
    if(check_if_hdf5_group_has_conduit_list_attribute(hdf5_group_id,
                                                      ref_path))
    {
        // special att: we have a list
        dest.set(DataType::list());
    }
    else
    {
        // no special att: we have an object
        dest.set(DataType::object());
    }

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
    
        CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Pclose(h5_gc_plist),
                                                        hdf5_group_id,
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

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_status,
                                                    hdf5_group_id,
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
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_dspace_id,
                                                    hdf5_dset_id,
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
    
        CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_dtype_id,
                                                        hdf5_dset_id,
                                                        ref_path,
                                               "Error reading HDF5 Datatype: "
                                               << hdf5_dset_id);

        index_t nelems     = H5Sget_simple_extent_npoints(h5_dspace_id);

        // Note: string case is handed properly in hdf5_dtype_to_conduit_dtype
        DataType dt        = hdf5_dtype_to_conduit_dtype(h5_dtype_id,
                                                         nelems,
                                                         ref_path);

        // if the endianness of the dset in the file doesn't
        // match the current machine we always want to convert it
        // on read.

        // check endianness
        // Note: string cases never land here b/c they are 
        // created with default endianness
        if(!dt.endianness_matches_machine())
        {
            // if they don't match, modify the dt
            // and get the proper hdf5 data type handle
            dt.set_endianness(Endianness::machine_default());
            
            // clean up our old handle
            CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Tclose(h5_dtype_id),
                                                            hdf5_dset_id,
                                                            ref_path,
                                        "Error closing HDF5 Datatype: "
                                        << h5_dtype_id);

            // get ref to standard variant of this dtype
            h5_dtype_id  = conduit_dtype_to_hdf5_dtype(dt,
                                                       ref_path);

            CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_dtype_id,
                                                            hdf5_dset_id,
                                                            ref_path,
                                        "Error creating HDF5 Datatype");

            // copy since the logic after read will cleanup
            h5_dtype_id  = H5Tcopy(h5_dtype_id);
            CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_dtype_id,
                                                            hdf5_dset_id,
                                                            ref_path,
                                        "Error copying HDF5 Datatype");
            // cleanup our ref from conduit_dtype_to_hdf5_dtype if necessary
            conduit_dtype_to_hdf5_dtype_cleanup(h5_dtype_id);
        }


        hid_t h5_status    = 0;

        // check for string special case, H5T_VARIABLE string
        if( H5Tis_variable_str(h5_dtype_id) )
        {
            //special case for reading variable string data
            // hdf5 reads the data onto its heap, and 
            // gives us a pointer to that location

            char *read_ptr[1] = {NULL};
            h5_status = H5Dread(hdf5_dset_id,
                                h5_dtype_id,
                                H5S_ALL,
                                H5S_ALL,
                                H5P_DEFAULT,
                                read_ptr);

            // copy the data out to the conduit node
            dest.set_string(read_ptr[0]);
        }
        // check for bad # of elements
        else if( dt.number_of_elements() < 0 )
        {
            CONDUIT_HDF5_ERROR(ref_path,
                                "Cannot read dataset with # of elements < 0");
        }
        else if(dest.dtype().is_compact() &&
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

        CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_status,
                                                        hdf5_dset_id,
                                                        ref_path,
                                               "Error reading HDF5 Dataset: "
                                                << hdf5_dset_id);
    
        CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Tclose(h5_dtype_id),
                                                        hdf5_dset_id,
                                                        ref_path,
                                               "Error closing HDF5 Datatype: "
                                               << h5_dtype_id);

    }

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Sclose(h5_dspace_id),
                                                    hdf5_dset_id,
                                                    ref_path,
                                           "Error closing HDF5 Dataspace: "
                                           << h5_dspace_id);

}

//---------------------------------------------------------------------------//
void
read_hdf5_tree_into_conduit_node(hid_t hdf5_id,
                                 const std::string &ref_path,
                                 Node &dest)
{
    herr_t     h5_status = 0;
    H5O_info_t h5_info_buf;
    
    h5_status = H5Oget_info(hdf5_id,&h5_info_buf);

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_status,
                                                    hdf5_id,
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
            // we only construct these strings 
            // when an error occurs, to avoid overhead
            // for healthy fetches
            std::string hdf5_err_ref_path;
            hdf5_ref_path_with_filename(hdf5_id,
                                        ref_path,
                                        hdf5_err_ref_path);
            CONDUIT_HDF5_ERROR(hdf5_err_ref_path,
                               "Cannot read HDF5 Object : "
                                << "(type == H5O_TYPE_UNKNOWN )");
            break;
        }
        case H5O_TYPE_NAMED_DATATYPE:
        {
            std::string hdf5_err_ref_path;
            hdf5_ref_path_with_filename(hdf5_id,
                                        ref_path,
                                        hdf5_err_ref_path);
            CONDUIT_HDF5_ERROR(hdf5_err_ref_path,
                               "Cannot read HDF5 Object "
                               << "(type == H5O_TYPE_NAMED_DATATYPE )");
            break;
        }
        case H5O_TYPE_NTYPES:
        {
            std::string hdf5_err_ref_path;
            hdf5_ref_path_with_filename(hdf5_id,
                                        ref_path,
                                        hdf5_err_ref_path);
            CONDUIT_HDF5_ERROR(hdf5_err_ref_path,
              "Cannot read HDF5 Object "
               << "(type == H5O_TYPE_NTYPES [This is an invalid HDF5 type!]");
            break;
        }
        default:
        {
            std::string hdf5_err_ref_path;
            hdf5_ref_path_with_filename(hdf5_id,
                                        ref_path,
                                        hdf5_err_ref_path);
            CONDUIT_HDF5_ERROR(hdf5_err_ref_path,
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
    
    unsigned int major_num=0;
    unsigned int minor_num=0;
    unsigned int release_num=0;
    
    herr_t h5_status = H5get_libversion(&major_num, &minor_num,&release_num);
    
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                             "Failed to fetch HDF5 library version info ");

    // most of our use cases are still using 1.8.
    // to allow hdf5 1.8 readers to read from hdf5 1.10 writers,
    // we want to pin to hdf5 1.8 features for now. 
    // There isn't a way to select 1.8, 
    // https://forum.hdfgroup.org/t/seconding-the-request-for-h5pset-libver-bounds-1-8-x-file-compat-option/4056
    // so only enable H5F_LIBVER_LATEST if we are using hdf5 1.8

    if(major_num == 1 && minor_num == 8)
    {
        h5_status = H5Pset_libver_bounds(h5_fa_props,
                                         H5F_LIBVER_LATEST,
                                         H5F_LIBVER_LATEST);

        CONDUIT_CHECK_HDF5_ERROR(h5_status,
                                 "Failed to set libver options for "
                                 << "property list " << h5_fa_props);

    }
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
hdf5_create_file(const std::string &file_path)
{
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;
        
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
    
    // enable hdf5 error stack
}

//-----------------------------------------------------------------------------
void
hdf5_close_file(hid_t hdf5_id)
{
    // close the hdf5 file
    CONDUIT_CHECK_HDF5_ERROR(H5Fclose(hdf5_id),
                             "Error closing HDF5 file handle: " << hdf5_id);
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

    std::string incompat_details;
    // check compat
    if(check_if_conduit_node_is_compatible_with_hdf5_tree(n,
                                                          "",
                                                          hdf5_id,
                                                          incompat_details))
    {
        // write if we are compat
        write_conduit_node_to_hdf5_tree(n,"",hdf5_id);
    }
    else
    {
        std::string hdf5_error_ref_path;
        hdf5_ref_path_with_filename(hdf5_id,
                                    hdf5_path,
                                    hdf5_error_ref_path);

        CONDUIT_ERROR("Failed to write node to " 
                      << "\"" << hdf5_error_ref_path << "\", "
                      << "existing HDF5 tree is "
                      << "incompatible with the Conduit Node."
                      << "\nDetails:\n" << incompat_details);
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
    
    std::string incompat_details;
    
    // check compat
    if(check_if_conduit_node_is_compatible_with_hdf5_tree(node,
                                                          "",
                                                          hdf5_id,
                                                          incompat_details))
    {
        // write if we are compat
        write_conduit_node_to_hdf5_tree(node,
                                        "",
                                        hdf5_id);
    }
    else
    {
        std::string hdf5_fname;
        hdf5_filename_from_hdf5_obj_id(hdf5_id, hdf5_fname);

        CONDUIT_ERROR("Failed to write node to " 
                      << "\"" << hdf5_fname << "\", "
                      << "existing HDF5 tree is "
                      << "incompatible with the Conduit Node."
                      << " Details: " << incompat_details);
    }
    // restore hdf5 error stack
}

//---------------------------------------------------------------------------//
void 
hdf5_save(const Node &node,
          const std::string &path)
{
    hdf5_write(node,path,false);
}

//---------------------------------------------------------------------------//
void
hdf5_save(const Node &node,
          const std::string &file_path,
          const std::string &hdf5_path)
{
    hdf5_write(node,file_path,hdf5_path,false);
}


//---------------------------------------------------------------------------//
void 
hdf5_append(const Node &node,
            const std::string &path)
{
    hdf5_write(node,path,true);
}

//---------------------------------------------------------------------------//
void
hdf5_append(const Node &node,
            const std::string &file_path,
            const std::string &hdf5_path)
{
    hdf5_write(node,file_path,hdf5_path,true);
}


//---------------------------------------------------------------------------//
void 
hdf5_write(const  Node &node,
           const std::string &path,
           bool append)
{
    // check for ":" split
    std::string file_path;
    std::string hdf5_path;

    conduit::utils::split_file_path(path,
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
               hdf5_path,
               append);
}


//---------------------------------------------------------------------------//
void
hdf5_write(const Node &node,
           const std::string &file_path,
           const std::string &hdf5_path,
           bool append)
{
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;

    hid_t h5_file_id = -1;
    if(append && utils::is_file(file_path))
    {
        // open existing hdf5 file for read + write
        h5_file_id = hdf5_open_file_for_read_write(file_path);
    }
    else // trunc
    {
        // open the hdf5 file for writing
        h5_file_id = hdf5_create_file(file_path);
    }

    hdf5_write(node,
               h5_file_id,
               hdf5_path);

    // close the hdf5 file
    CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                             "Error closing HDF5 file: " << file_path);

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
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;
    
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

    // restore hdf5 error stack
}

//---------------------------------------------------------------------------//
hid_t
hdf5_open_file_for_read_write(const std::string &file_path)
{
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;

    hid_t h5_fa_plist = create_hdf5_file_access_plist();
    
    // open the hdf5 file for read + write
    hid_t h5_file_id = H5Fopen(file_path.c_str(),
                               H5F_ACC_RDWR,
                               h5_fa_plist);

    CONDUIT_CHECK_HDF5_ERROR(h5_file_id,
                             "Error opening HDF5 file for reading: " 
                              << file_path);

    CONDUIT_CHECK_HDF5_ERROR(H5Pclose(h5_fa_plist),
                             "Failed to close HDF5 H5P_FILE_ACCESS "
                             << "property list: " << h5_fa_plist);
    
    return h5_file_id;

    // restore hdf5 error stack
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
    
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_child_obj,
                                                    hdf5_id,
                                                    hdf5_path,
                            "Failed to fetch HDF5 object from: "
                             << hdf5_id << ":" << hdf5_path);

    read_hdf5_tree_into_conduit_node(h5_child_obj,
                                     hdf5_path,
                                     dest);
    
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Oclose(h5_child_obj),
                                                    hdf5_id,
                                                    hdf5_path,
                             "Failed to close HDF5 Object: "
                             << h5_child_obj);
    
    // restore hdf5 error stack
}
//---------------------------------------------------------------------------//
void
hdf5_read(const std::string &file_path,
          const std::string &hdf5_path,
          Node &node)
{
    // note: hdf5 error stack is suppressed in these calls
    
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
hdf5_read(const std::string &path,
          Node &node)
{
    // check for ":" split
    std::string file_path;
    std::string hdf5_path;
    
    conduit::utils::split_file_path(path,
                                    std::string(":"),
                                    file_path,
                                    hdf5_path);

    // We will read the root if no hdf5_path is given.
    if(hdf5_path.size() == 0)
    {
        hdf5_path = "/";
    }
    
    // note: hdf5 error stack is suppressed in this call
    hdf5_read(file_path,
              hdf5_path,
              node);
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
    
    // restore hdf5 error stack
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
    // restore hdf5 error stack
}

//---------------------------------------------------------------------------//
void
hdf5_remove_path(hid_t hdf5_id,
                 const std::string &hdf5_path)
{
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;
    
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Ldelete(hdf5_id,
                                                              hdf5_path.c_str(),
                                                              H5P_DEFAULT),
                                                    hdf5_id,
                                                    hdf5_path,
                             "Error deleting HDF5 path: "
                              << hdf5_id << ":" << hdf5_path);

    // restore hdf5 error stack
}




//---------------------------------------------------------------------------//
bool
is_hdf5_file(const std::string &file_path)
{
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;
    
    bool res = false;
    // open the file for read to check if it is valid hdf5
    //
    // don't use H5F_ACC_RDWR, b/c if we already have a file handle open 
    // that is RDONLY, the open will fail
    //
    // use H5F_ACC_RDONLY b/c it will work with open file handles
    hid_t h5_file_id = H5Fopen(file_path.c_str(),
                               H5F_ACC_RDONLY,
                               H5P_DEFAULT);
    
    if( h5_file_id >= 0)
    {
        res = true;
        H5Fclose(h5_file_id);
    }

    return res;
    // restore hdf5 error stack
}

//---------------------------------------------------------------------------//
void hdf5_group_list_child_names(hid_t hdf5_id,
                                 const std::string &hdf5_path,
                                 std::vector<std::string> &res)
{   
    // disable hdf5 error stack
    HDF5ErrorStackSupressor supress_hdf5_errors;
    
    res.clear();
    // first, hdf5_id + path must be a group in order to have children

    H5O_info_t h5_info_buf;

    // Get type of the object
    herr_t h5_status = H5Oget_info_by_name(hdf5_id,
                                           hdf5_path.c_str(),
                                           &h5_info_buf,
                                           H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_status,
                                                    hdf5_id,
                                                    "",
                                           "Error fetching HDF5 Object info: "
                                           << " parent: " << hdf5_id 
                                           << " path:"    << hdf5_path) ;

    if( h5_info_buf.type != H5O_TYPE_GROUP )
    {    
        //  not a group, child names will be empty
        // we could also choose to throw an error in the future 
        return;
    }

    // we have a group    
    // we don't care about links in this case, we want
    // the child names regardless, so we don't have to use H5Literate
    //
    // we can use H5Lget_name_by_idx, as demoed in 
    // https://support.hdfgroup.org/ftp/HDF5/examples/examples-by-api/hdf5-examples/1_10/C/H5G/h5ex_g_corder.c
    // 
    

    hid_t h5_group_id = H5Gopen(hdf5_id,
                                hdf5_path.c_str(),
                                H5P_DEFAULT);
    
    CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(h5_group_id,
                                                    hdf5_id,
                                           "",
                                           "Error opening HDF5 "
                                           << "Group: " 
                                           << " parent: "
                                           << hdf5_id 
                                           << " path:"
                                           << hdf5_path);
    
    // get group info so we can find the # of children
    H5G_info_t  h5_group_info;
    h5_status = H5Gget_info(h5_group_id, &h5_group_info);

    // buffer for child names, if the names are bigger than this
    // buffer can hold, we will fall back to a malloc
    char name_buff[512];
    
    for (hsize_t i=0; i < h5_group_info.nlinks; i++)
    {
        char *name_buff_ptr = name_buff;
        char *name_buff_tmp = NULL;
        // Get size of name,
        int name_size = H5Lget_name_by_idx(h5_group_id, ".",
                                           H5_INDEX_CRT_ORDER,
                                           H5_ITER_INC,
                                           i,
                                           NULL,
                                           0,
                                           H5P_DEFAULT);

        if(name_size <= 0)
        {
            // error, not valid
            std::string hdf5_err_ref_path;
            hdf5_ref_path_with_filename(h5_group_id,
                                        hdf5_path,
                                        hdf5_err_ref_path);

            CONDUIT_HDF5_ERROR(hdf5_err_ref_path,
                               "Failed to fetch group child name at index " << i);
        }
        // now that we know the name exists, add 1 for null terminator.
        name_size++;

        if(name_size > 512)
        {
            // we dont have room with our built in buffer
            // fall back to malloc
            name_buff_tmp = (char*)malloc(sizeof(char)*name_size);
            name_buff_ptr = name_buff_tmp;
        }
        
        name_size = H5Lget_name_by_idx(h5_group_id, ".",
                                       H5_INDEX_CRT_ORDER,
                                       H5_ITER_INC,
                                       i,
                                       name_buff_ptr,
                                       name_size,
                                       H5P_DEFAULT);
        
        res.push_back(std::string(name_buff_ptr));

        if(name_buff_tmp)
        {
            free(name_buff_tmp);
            name_buff_tmp = NULL;
        }
   }
   
   CONDUIT_CHECK_HDF5_ERROR_WITH_FILE_AND_REF_PATH(H5Gclose(h5_group_id),
                                                   hdf5_id,
                                          "",
                                          "Failed to close HDF5 Group " 
                                          << h5_group_id);

   // restore hdf5 error stack
}



}
//-----------------------------------------------------------------------------
// -- end conduit::relay::<mpi>::io --
//-----------------------------------------------------------------------------

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
}
//-----------------------------------------------------------------------------
// -- begin conduit::relay::mpi --
//-----------------------------------------------------------------------------
#endif

}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
