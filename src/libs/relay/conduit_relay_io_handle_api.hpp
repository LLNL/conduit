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
/// file: conduit_relay_io_handle_api.hpp
///
//-----------------------------------------------------------------------------


#ifndef CONDUIT_RELAY_IO_HANDLE_API_HPP
#define CONDUIT_RELAY_IO_HANDLE_API_HPP


//-----------------------------------------------------------------------------
///
/// class: conduit::relay::{mpi}::io::IOHandle
///
/// Contract: Changes to backing (file on disk, etc) aren't guaranteed to 
//  be reflected until a call to close
//-----------------------------------------------------------------------------
class CONDUIT_API IOHandle
{

public:
     IOHandle();
    ~IOHandle();

    /// establish a handle
    void open(const std::string &path);
    void open(const std::string &path,
              const std::string &protocol);

    void open(const std::string &path,
              const std::string &protocol,
              const Node &options);

    /// check if the handle is currently open
              
    bool is_open() const;

    /// read contents starting at the root of the handle
    void read(Node &node);
    /// read contents starting at given subpath
    void read(const std::string &path,
              Node &node);

    /// write contents of passed node to the root of the handle
    void write(const Node &node);
    /// write contents of passed node to given subpath
    void write(const Node &node,
               const std::string &path);

    /// list child names at root of handle
    void list_child_names(std::vector<std::string> &res) const;
    /// list child names at subpath
    void list_child_names(const std::string &path,
                          std::vector<std::string> &res) const;

    // TODO: options variants for read and write above? with update of 
    // above options with passed?

    /// remove contents at given path
    void remove(const std::string &path);

    /// check if given path exists
    bool has_path(const std::string &path) const;

    // FUTURE: also provide access to read schema
    // void read_schema(Schema &schema);
    // void read_schema(const std::string &path,
    //                  Schema &schema);

    /// close the handle
    void close();

    class HandleInterface;
private:
    HandleInterface *m_handle;

};


#endif

