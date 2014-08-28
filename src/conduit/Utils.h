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
/// file: Utils.h
///

#ifndef __CONDUIT_UTILS_H
#define __CONDUIT_UTILS_H

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <Core.h>

namespace conduit
{

namespace utils
{
     void     split_path(const std::string &path,
                         std::string &curr,
                         std::string &next);

     std::string json_sanitize(const std::string &json);
     
     
     template< typename T >
     std::string to_hex_string(T value)
     {
           std::stringstream oss;
           oss << std::hex << value;
           return  oss.str();
     }
     
     void indent(std:: ostringstream &oss,
                 index_t indent,
                 index_t depth,
                 const std::string &pad);
     
}
}

#endif
