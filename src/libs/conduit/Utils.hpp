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
// For details, see https://lc.llnl.gov/conduit/.
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
/// file: Utils.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_UTILS_HPP
#define CONDUIT_UTILS_HPP

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>


//-----------------------------------------------------------------------------
// -- conduit includes -- 
//-----------------------------------------------------------------------------
#include "Core.hpp"


//-----------------------------------------------------------------------------
//
/// The CONDUIT_INFO macro is the primary mechanism used to log basic messages.
/// It currently simply prints the message to std::out.
///
//-----------------------------------------------------------------------------
#define CONDUIT_INFO( msg )                                         \
{                                                                   \
    std::ostringstream conduit_oss_info;                            \
    conduit_oss_info << "[" << std::string(__FILE__);               \
    conduit_oss_info << " : " << __LINE__  << "]";                  \
    conduit_oss_info << "\n " << msg;                               \
    std::cout << conduit_oss_info.str() << std::endl;               \
}                                                                   \


//-----------------------------------------------------------------------------
//
/// The CONDUIT_ERROR macro is the primary mechanism used to capture errors  
/// in conduit. It calls conduit::utils::handle_error() which invokes 
/// the currently configured error handler. 
///
/// The default error handler throws a c++ exception, in the form of a
/// conduit::Error class. You can change the error handler via
/// conduit::utils::set_error_handler().
//
//-----------------------------------------------------------------------------
#define CONDUIT_ERROR( msg )                                        \
{                                                                   \
    std::ostringstream conduit_oss_error;                           \
    conduit_oss_error << msg;                                       \
    conduit::utils::handle_error( conduit_oss_error.str(),          \
                                  std::string(__FILE__),            \
                                  __LINE__);                        \
}                                                                   \

//-----------------------------------------------------------------------------
//
/// The CONDUIT_ASSERT macro is the primary mechanism used to capture assert
/// failures in conduit. It calls conduit::utils::handle_error() which invokes 
/// the currently configured error handler. 
///
/// The default error handler throws a c++ exception, in the form of a
/// conduit::Error class. You can change the error handler via
/// conduit::utils::set_error_handler().
//
//-----------------------------------------------------------------------------
#define CONDUIT_ASSERT( cond, msg)                                   \
{                                                                    \
    if(!cond)                                                        \
    {                                                                \
        std::ostringstream conduit_oss_assert;                       \
        conduit_oss_assert << msg;                                   \
        conduit::utils::handle_error( conduit_oss_assert.str(),      \
                                      std::string(__FILE__),         \
                                      __LINE__);                     \
    }                                                                \
}                                                                    \

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::utils --
//-----------------------------------------------------------------------------
namespace utils
{

//-----------------------------------------------------------------------------
/// Primary interface used by the conduit API when an error occurs. 
/// This simply dispatches the error to the currently configured error handler.
/// The default error handler throws a conduit::Error exception.
//-----------------------------------------------------------------------------
    void CONDUIT_API handle_error(const std::string &msg,
                                  const std::string &file,
                                  int line);

//-----------------------------------------------------------------------------
/// Allows other libraries to provide an alternate error handler.
//-----------------------------------------------------------------------------
    void CONDUIT_API set_error_handler( void(*on_error)
                                        (const std::string &,
                                         const std::string &,
                                         int));

//-----------------------------------------------------------------------------
/// Default error handler, which throws a conduit::Error exception.
//-----------------------------------------------------------------------------
   void CONDUIT_API default_error_handler(const std::string &msg,
                                          const std::string &file,
                                          int line);

//-----------------------------------------------------------------------------
/// Helpers for common string splitting operations. 
//-----------------------------------------------------------------------------
    void CONDUIT_API split_string(const std::string &path,
                                  const std::string &sep,
                                  std::string &curr,
                                  std::string &next);

    void CONDUIT_API rsplit_string(const std::string &path,
                                   const std::string &sep,
                                   std::string &curr,
                                   std::string &next);

    void  CONDUIT_API split_path(const std::string &path,
                                 std::string &curr,
                                 std::string &next);


//-----------------------------------------------------------------------------
/// Base64 Encoding of Buffers 
//-----------------------------------------------------------------------------
    void CONDUIT_API base64_encode(const void *src,
                                   index_t src_nbytes,
                                   void *dest);

    void CONDUIT_API base64_decode(const void *src,
                                   index_t src_nbytes,
                                   void *dest);

//-----------------------------------------------------------------------------
     std::string CONDUIT_API json_sanitize(const std::string &json);
     
//----------------------------------------------------------------------------- 
     template< typename T >
     std::string to_hex_string(T value)
     {
           std::stringstream oss;
           oss << std::hex << value;
           return  oss.str();
     }

//-----------------------------------------------------------------------------
     void CONDUIT_API indent(std:: ostringstream &oss,
                             index_t indent,
                             index_t depth,
                             const std::string &pad);
     
}
//-----------------------------------------------------------------------------
// -- end conduit::utils --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
