/*****************************************************************************
* Copyright (c) 2014, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory. 
* 
* All rights reserved.
* 
* This source code cannot be distributed without further review from 
* Lawrence Livermore National Laboratory.
*****************************************************************************/

///
/// file: Generator.h
///

#ifndef __CONDUIT_GENERATOR_H
#define __CONDUIT_GENERATOR_H

#include "Core.h"
#include "Endianness.h"
#include "DataType.h"

#include "Node.h"
#include "Schema.h"

namespace conduit
{

class CONDUIT_API Generator
{
public:

    friend class Node;
    friend class Schema;
    
    /* Constructors */
    Generator(const std::string &json_schema);
    
    Generator(const std::string &json_schema,
              void *data);
    
    Generator(const std::string &json_schema,
              const std::string &protocol,
              void *data = NULL);
    
    
    /* Parsing Interface */
    void walk(Schema &) const;
    void walk(Node &)   const;

private:
    std::string  m_json_schema;
    std::string  m_protocol;
    void        *m_data;

};

}


#endif
