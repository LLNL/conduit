//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
/// file: conduit_relay_hdf5.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_HDF5_HPP
#define CONDUIT_RELAY_HDF5_HPP

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <hdf5.h>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_relay_io.hpp"


//-----------------------------------------------------------------------------
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
//
/// When writing a node to a HDF5 hierarchy there are two steps:
/// (1) Given a destination path, find (or create) the proper HDF5 parent group
/// (2) Write the node's data into the parent group
/// 
///
/// 1) To find (or create) the parent group there are two cases to finalize
///    the desired parent's HDF5 path:
///
///  A) If the input node is a leaf type, the last part of the path 
///     (the portion after the last slash '/', or the entire path if 
///     the path doesn't contain any slashes '/') is reserved for the
///     name of the HDF5 dataset that will be used to store the node's data. 
///     The rest of the path will be used to find or create the proper parent 
///     group.
/// 
///  B) If the input node is an Object, the full input path will be used to 
///     find or create the proper parent group.
///
///  Given the desired parent path: 
/// 
///  If the desired parent path exists in the HDF5 tree and resolves to 
///  a group, that group will be used as the parent.
///
///  If the desired parent path does not exist, or partially exists 
///  relay will attempt to create a hierarchy of HDF5 groups to represent it.
///
///  During this process, if any part of the path exists and is not a HDF5 
///  group, an error is thrown and nothing will be modified.
///  (This error check happens before anything is modified in HDF5)
///
///
/// 2) For writing the data, there are two cases:
///   A) If the input node is an Object, the children of the node will be
///      written to the parent group. If children correspond to existing 
///      HDF5 entires and they are incompatible (e.g. a group vs a dataset)
//       an error is thrown and nothing will be written.
///      (This error check happens before anything is modified in HDF5)
///
///   B) If the input node is a leaf type, the last part of the path will be 
///      used as the name for a HDF5 dataset that will hold the node's data. 
///      If a child with this name already exists in in the parent group, and
///      it is not compatible (e.g. a group vs a dataset) an error is thrown 
///     and nothing will be written.
///
//-----------------------------------------------------------------------------
///  Note: HDF5 I/O is not implemented for Conduit Nodes in the List role.
///        We believe this will require a non-standard HDF5 convention, and
///        we want to focus on the Object and Leave cases since they are 
///        compatible with HDF5's data model.
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Create a hdf5 file for read and write using conduit's selected hdf5 plists.
//-----------------------------------------------------------------------------
hid_t hdf5_create_file(const std::string &file_path);

//-----------------------------------------------------------------------------
/// Close hdf5 file handle
//-----------------------------------------------------------------------------
void hdf5_close_file(hid_t hdf5_id);

//-----------------------------------------------------------------------------
/// Write node data to a given path
///
/// This methods supports a file system and hdf5 path, joined using a ":"
///  ex: "/path/on/file/system.hdf5:/path/inside/hdf5/file"
/// 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API hdf5_write(const Node &node,
                                  const std::string &path);

//-----------------------------------------------------------------------------
/// Write node data to given file system path and internal hdf5 path
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API hdf5_write(const Node &node,
                                  const std::string &file_path,
                                  const std::string &hdf5_path);

//-----------------------------------------------------------------------------
/// Write node data to the hdf5_path relative to group represented  by 
/// hdf5_id 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API hdf5_write(const Node &node,
                                  hid_t hdf5_id,
                                  const std::string &hdf5_path);

//-----------------------------------------------------------------------------
/// Write node data to group represented by hdf5_id
/// 
/// Note: this only works for Conduit Nodes in the Object role.
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API hdf5_write(const Node &node,
                                  hid_t hdf5_id);


//-----------------------------------------------------------------------------
/// Open a hdf5 file for reading, using conduit's selected hdf5 plists.
//-----------------------------------------------------------------------------
hid_t hdf5_open_file_for_read(const std::string &file_path);
hid_t hdf5_open_file_for_read_write(const std::string &file_path);

//-----------------------------------------------------------------------------
/// Read hdf5 data from given path into the output node 
/// 
/// This methods supports a file system and hdf5 path, joined using a ":"
///  ex: "/path/on/file/system.hdf5:/path/inside/hdf5/file"
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API hdf5_read(const std::string &path,
                                 Node &node);

//-----------------------------------------------------------------------------
/// Read hdf5 data from given file system path and internal hdf5 path into 
/// the output node
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API hdf5_read(const std::string &file_path,
                                 const std::string &hdf5_path,
                                 Node &node);

//-----------------------------------------------------------------------------
/// Read from hdf5 path relative to the hdf5 id into the output node
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API hdf5_read(hid_t hdf5_id,
                                 const std::string &hdf5_path,
                                 Node &node);
//-----------------------------------------------------------------------------
/// Read from hdf5 id into the output node
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API hdf5_read(hid_t hdf5_id,
                                 Node &node);

//-----------------------------------------------------------------------------
/// Read from hdf5 id into the output node
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API hdf5_read(hid_t hdf5_id,
                                 Node &node);

//-----------------------------------------------------------------------------
/// Helpers for converting between hdf5 dtypes and conduit dtypes
/// 
///  Throughout the relay hdf5 implementation, we use DataType::Empty when
///  the hdf5 data space is H5S_NULL, regardless of what the hdf5 data type is.
///  That isn't reflected in these helper functions,they handle
///  mapping of endianness and leaf types other than empty.
///
///
///  Note: In these functions, ref_path is used to provide context about the
///  hdf5 tree when an error occurs. Using it is recommend but not required.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
hid_t CONDUIT_RELAY_API    conduit_dtype_to_hdf5_dtype(const DataType &dt,
                                                const std::string &ref_path="");

//-----------------------------------------------------------------------------
DataType CONDUIT_RELAY_API hdf5_dtype_to_conduit_dtype(hid_t hdf5_dtype_id,
                                                       index_t num_elems,
                                                const std::string &ref_path="");


//-----------------------------------------------------------------------------
/// Check if path exists relative to hdf5 id
//-----------------------------------------------------------------------------
bool CONDUIT_RELAY_API hdf5_has_path(hid_t hdf5_id, const std::string &path);

//-----------------------------------------------------------------------------
/// Pass a Node to set hdf5 i/o options.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API hdf5_set_options(const Node &opts);

//-----------------------------------------------------------------------------
/// Get a Node that contains hdf5 i/o options.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API hdf5_options(Node &opts);


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


#endif

