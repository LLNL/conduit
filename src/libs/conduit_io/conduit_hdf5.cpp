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
// * Neither the name of the LLNS/LL    set(ENABLE_FORTRAN ON CACHE PATH "")NL nor the names of its contributors may
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
// Data Type Helper methods that aren't part of public conduit::io
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
        res.set_id(DataType::UINT8_ID);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I16LE))
    {
        res.set_id(DataType::UINT16_ID);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I32LE))
    {
        res.set_id(DataType::UINT32_ID);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I64LE))
    {
        res.set_id(DataType::UINT64_ID);
        res.set_endianness(Endianness::LITTLE_ID);
    }
     // big endian
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I8BE))
    {
        res.set_id(DataType::UINT8_ID);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I16BE))
    {
        res.set_id(DataType::UINT16_ID);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I32BE))
    {
        res.set_id(DataType::UINT32_ID);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_I64BE))
    {
        res.set_id(DataType::UINT64_ID);
        res.set_endianness(Endianness::BIG_ID);
    }
    //-----------------------------------------------
    // unsigned ints
    //-----------------------------------------------
    // little endian
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U8LE))
    {
        res.set_id(DataType::UINT8_ID);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U16LE))
    {
        res.set_id(DataType::UINT16_ID);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U32LE))
    {
        res.set_id(DataType::UINT32_ID);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U64LE))
    {
        res.set_id(DataType::UINT64_ID);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    // big endian
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U8BE))
    {
        res.set_id(DataType::UINT8_ID);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U16BE))
    {
        res.set_id(DataType::UINT16_ID);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U32BE))
    {
        res.set_id(DataType::UINT32_ID);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_STD_U64BE))
    {
        res.set_id(DataType::UINT64_ID);
        res.set_endianness(Endianness::BIG_ID);
    }
    //-----------------------------------------------
    // floating point types
    //-----------------------------------------------
    // little endian
    else if(H5Tequal(hdf5_dtype_id,H5T_IEEE_F32LE))
    {
        res.set_id(DataType::FLOAT32_ID);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_IEEE_F64LE))
    {
        res.set_id(DataType::FLOAT64_ID);
        res.set_endianness(Endianness::LITTLE_ID);
    }
    // big endian
    else if(H5Tequal(hdf5_dtype_id,H5T_IEEE_F32BE))
    {
        res.set_id(DataType::FLOAT32_ID);
        res.set_endianness(Endianness::BIG_ID);
    }
    else if(H5Tequal(hdf5_dtype_id,H5T_IEEE_F64BE))
    {
        res.set_id(DataType::FLOAT64_ID);
        res.set_endianness(Endianness::BIG_ID);
    }
    //-----------------------------------------------
    // String Types
    //-----------------------------------------------
    else if(H5Tequal(hdf5_dtype_id,H5T_C_S1))
    {
        res.set_id(DataType::CHAR8_STR_ID);
        res.set_endianness(Endianness::DEFAULT_ID);
    }
    else
    {
        CONDUIT_ERROR("Error with HDF5 Leaf DataType to conduit::DataType Conversion");
    }

    // set proper number of elems from what was passed
    res.set_number_of_elements(num_elems);
    
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
hdf5_write(const Node &node,
           hid_t hdf5_id,
           const std::string &hdf5_path)
{
    DataType dt = node.dtype();
    if(dt.is_object())
    {
        
        // Create a group named with the path name in the file.
        hid_t h5_group_id = H5Gcreate2(hdf5_id,
                                       hdf5_path.c_str(),
                                       H5P_DEFAULT,
                                       H5P_DEFAULT,
                                       H5P_DEFAULT);

        CONDUIT_CHECK_HDF5_ERROR(h5_group_id,
                                 "Error creating HDF5 Group: " << hdf5_path);
        // strong dose of evil casting, but it's ok b/c we are grownups here?
        // time we will tell ...
        NodeIterator itr = const_cast<Node*>(&node)->children();

        // call on each child with expanded path
        while(itr.has_next())
        {
            Node &child = itr.next();
            hdf5_write(child,
                       h5_group_id,
                       itr.path());
        }
        
        // close the group.
        herr_t h5_status = H5Gclose(h5_group_id);
        CONDUIT_CHECK_HDF5_ERROR(h5_status,
                                 "Error closing HDF5 Group: " << hdf5_path);
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
    // open the desired group
    hid_t h5_group_id = H5Gopen(hdf5_id,
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

};
//-----------------------------------------------------------------------------
// -- end conduit::io --
//-----------------------------------------------------------------------------


};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
