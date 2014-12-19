//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: Generator.h
///
//-----------------------------------------------------------------------------

#ifndef __CONDUIT_GENERATOR_H
#define __CONDUIT_GENERATOR_H

//-----------------------------------------------------------------------------
// -- conduit library includes -- 
//-----------------------------------------------------------------------------
#include "Core.h"
#include "Endianness.h"
#include "DataType.h"
#include "Node.h"
#include "Schema.h"

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
    void walk(Schema &) const;
    /// parse a json schema to a Node object.
    void walk(Node &)   const;

private:
//-----------------------------------------------------------------------------
//
// -- conduit::Error private data members --
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
