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
// For details, see: http://cyrush.github.io/conduit.
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
/// file: Generator.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_GENERATOR_HPP
#define CONDUIT_GENERATOR_HPP

//-----------------------------------------------------------------------------
// -- conduit includes -- 
//-----------------------------------------------------------------------------
#include "Core.hpp"
#include "Endianness.hpp"
#include "DataType.hpp"
#include "Node.hpp"
#include "Schema.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{
//-----------------------------------------------------------------------------
// -- begin conduit::Generator --
//-----------------------------------------------------------------------------
///
/// class: conduit::Generator
///
/// description:
///  The Generator class implements parsing logic for json schemas.
///
//-----------------------------------------------------------------------------
class CONDUIT_API Generator
{
public:
    
//-----------------------------------------------------------------------------
// -- friends of Generator --
//-----------------------------------------------------------------------------
    friend class Node;
    friend class Schema;
    
//-----------------------------------------------------------------------------
//
// -- conduit::Generator public members --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Generator Construction and Destruction
//-----------------------------------------------------------------------------
    /// create a generator from json
    Generator(const std::string &json_schema);
    /// create a generator from json, which can be applied to a data pointer
    Generator(const std::string &json_schema,
              void *data);
    /// create a generator from json, using a given protocol name, which can 
    /// optionally be applied to a data pointer
    Generator(const std::string &json_schema,
              const std::string &protocol,
              void *data = NULL);

//-----------------------------------------------------------------------------
// JSON Parsing interface
//-----------------------------------------------------------------------------
    /// parse a json schema to a Schema object.
    void walk(Schema &sdest) const;

    /// parse a json schema to a Node object.
    void walk(Node &ndest) const;
    void walk_external(Node &ndest) const;

    // private class used to encapsulate RapidJSON logic. 
    class Parser;


private:
//-----------------------------------------------------------------------------
//
// -- conduit::Generator private data members --
//
//-----------------------------------------------------------------------------
    
    
    /// holds the json based schema text
    std::string  m_json_schema;
    /// holds the parsing protocol
    std::string  m_protocol;
    /// optional external data pointer
    void        *m_data;

};
//-----------------------------------------------------------------------------
// -- end conduit::Generator --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif
