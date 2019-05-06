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
/// file: conduit_utils.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_UTILS_HPP
#define CONDUIT_UTILS_HPP

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>


//-----------------------------------------------------------------------------
// -- conduit includes -- 
//-----------------------------------------------------------------------------
#include "conduit_core.hpp"


//-----------------------------------------------------------------------------
//
/// The CONDUIT_INFO macro is the primary mechanism used to log basic messages.
/// It calls conduit::utils::handle_info() which invokes 
///
/// The default info handler prints the message to std::out. 
/// You can change the info handler via conduit::utils::set_info_handler().
///
//-----------------------------------------------------------------------------
#define CONDUIT_INFO( msg )                                         \
{                                                                   \
    std::ostringstream conduit_oss_info;                            \
    conduit_oss_info << msg;                                        \
    conduit::utils::handle_info( conduit_oss_info.str(),            \
                                  std::string(__FILE__),            \
                                  __LINE__);                        \
}                                                                   \

//-----------------------------------------------------------------------------
//
/// The CONDUIT_WARN macro is the primary mechanism used to capture warnings
/// in conduit. It calls conduit::utils::handle_warning() which invokes 
/// the currently configured warning handler. 
///
/// The default warning handler throws a c++ exception, in the form of a
/// conduit::Error instance. You can change the error handler via
/// conduit::utils::set_warning_handler().
//
//-----------------------------------------------------------------------------
#define CONDUIT_WARN( msg )                                          \
{                                                                    \
    std::ostringstream conduit_oss_warn;                             \
    conduit_oss_warn << msg;                                         \
    conduit::utils::handle_warning( conduit_oss_warn.str(),          \
                                    std::string(__FILE__),           \
                                    __LINE__);                       \
}                                                                    \

//-----------------------------------------------------------------------------
//
/// The CONDUIT_ERROR macro is the primary mechanism used to capture errors  
/// in conduit. It calls conduit::utils::handle_error() which invokes 
/// the currently configured error handler. 
///
/// The default error handler throws a c++ exception, in the form of a
/// conduit::Error instance. You can change the error handler via
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
/// The CONDUIT_ASSERT macro is the primary mechanism used to for asserts
/// in conduit. It calls conduit::utils::handle_error() which invokes 
/// the currently configured error handler. 
///
/// The default error handler throws a c++ exception, in the form of a
/// conduit::Error instance. You can change the error handler via
/// conduit::utils::set_error_handler().
//
//-----------------------------------------------------------------------------
#define CONDUIT_ASSERT( cond, msg )                                  \
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
//
/// The CONDUIT_CHECK macro is the mechanism used for checks in conduit.
/// It calls conduit::utils::handle_warning() which invokes 
/// the currently configured warning handler. 
///
/// The default warning handler throws a c++ exception, in the form of a
/// conduit::Error instance. You can change the error handler via
/// conduit::utils::set_warning_handler().
//
//-----------------------------------------------------------------------------
#define CONDUIT_CHECK( cond, msg )                                   \
{                                                                    \
    if(!cond)                                                        \
    {                                                                \
        std::ostringstream conduit_oss_check;                        \
        conduit_oss_check << msg;                                    \
        conduit::utils::handle_warning( conduit_oss_check.str(),     \
                                        std::string(__FILE__),       \
                                        __LINE__);                   \
    }                                                                \
}                                                                    \

//-----------------------------------------------------------------------------
//
/// The CONDUIT_EPSILON macro defines the default machine epsilon
/// value used when comparing floating-point values. This value is used
/// by default in all of the Conduit comparison operations (e.g.
/// 'conduit::Node::diff' and 'conduit::Node::diff_compatible').
//
//-----------------------------------------------------------------------------
#define CONDUIT_EPSILON 1e-12


//-----------------------------------------------------------------------------
//
/// The CONDUIT_UNUSED macro is used to identify unused variables 
/// in cases where it is difficult to avoid defining in the method signature 
/// for methods that use optional features. 
///
//-----------------------------------------------------------------------------
#define CONDUIT_UNUSED( var ) (void)(var)

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
/// Primary interface used by the conduit API when an info message is issued 
/// This simply dispatches the message to the currently configured info handler.
/// The default info handler prints a the message to std::cout;
//-----------------------------------------------------------------------------
    void CONDUIT_API handle_info(const std::string &msg,
                                 const std::string &file,
                                 int line);

//-----------------------------------------------------------------------------
/// Allows other libraries to provide an alternate info message handler.
//-----------------------------------------------------------------------------
    void CONDUIT_API set_info_handler( void(*on_info)
                                       (const std::string &,
                                        const std::string &,
                                        int));

//-----------------------------------------------------------------------------
/// Default info message handler, which prints the message to std::cout;
//-----------------------------------------------------------------------------
   void CONDUIT_API default_info_handler(const std::string &msg,
                                         const std::string &file,
                                         int line);

//-----------------------------------------------------------------------------
/// Primary interface used by the conduit API when a warning is issued. 
/// This simply dispatches the warning to the currently configured warning handler.
/// The default warning handler throws a conduit::Error exception.
//-----------------------------------------------------------------------------
    void CONDUIT_API handle_warning(const std::string &msg,
                                    const std::string &file,
                                    int line);

//-----------------------------------------------------------------------------
/// Allows other libraries to provide an alternate warning handler.
//-----------------------------------------------------------------------------
    void CONDUIT_API set_warning_handler( void(*on_error)
                                         (const std::string &,
                                          const std::string &,
                                          int));

//-----------------------------------------------------------------------------
/// Default warning handler, which throws a conduit::Error exception.
//-----------------------------------------------------------------------------
   void CONDUIT_API default_warning_handler(const std::string &msg,
                                            const std::string &file,
                                            int line);


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
    void CONDUIT_API split_string(const std::string &str,
                                  const std::string &sep,
                                  std::string &curr,
                                  std::string &next);

    void CONDUIT_API split_string(const std::string &str,
                                  char sep,
                                  std::vector<std::string> &sv);

    void CONDUIT_API rsplit_string(const std::string &str,
                                   const std::string &sep,
                                   std::string &curr,
                                   std::string &next);

//-----------------------------------------------------------------------------
/// Helpers for splitting and joining conduit paths (which always use "/")
//-----------------------------------------------------------------------------
    void  CONDUIT_API split_path(const std::string &path,
                                 std::string &curr,
                                 std::string &next);

    void  CONDUIT_API rsplit_path(const std::string &path,
                                  std::string &curr,
                                  std::string &next);

    std::string CONDUIT_API join_path(const std::string &left,
                                      const std::string &right);

//-----------------------------------------------------------------------------
/// Helpers for splitting and joining file system paths.
/// These use the proper platform specific separator (/ or \).
//-----------------------------------------------------------------------------
    std::string CONDUIT_API file_path_separator();

    void CONDUIT_API        split_file_path(const std::string &path,
                                            std::string &curr,
                                            std::string &next);

    void CONDUIT_API        rsplit_file_path(const std::string &path,
                                             std::string &curr,
                                             std::string &next);

     //------------------------------------------------------------------------
    /// `split_file_path` and `rsplit_file_path` are helpers that allows us to 
    ///  use  ":" for subpaths even on Windows when a drive letter including 
    ///  ":" is in the path. 
    //-------------------------------------------------------------------------
    void CONDUIT_API split_file_path(const std::string &str,
                                     const std::string &sep,
                                     std::string &curr,
                                     std::string &next);

    void CONDUIT_API rsplit_file_path(const std::string &str,
                                      const std::string &sep,
                                      std::string &curr,
                                      std::string &next);


    std::string CONDUIT_API join_file_path(const std::string &left,
                                           const std::string &right);




//-----------------------------------------------------------------------------
     bool CONDUIT_API is_file(const std::string &path);

//-----------------------------------------------------------------------------
     bool CONDUIT_API is_directory(const std::string &path);

//-----------------------------------------------------------------------------
     index_t CONDUIT_API file_size(const std::string &path);

//-----------------------------------------------------------------------------
/// Creates a new directory.
/// 
/// Does not recursively create parent directories if they do not already 
/// exist.
//-----------------------------------------------------------------------------
     bool CONDUIT_API create_directory(const std::string &path);

//-----------------------------------------------------------------------------
/// Remove files, or empty directories
//-----------------------------------------------------------------------------
     bool CONDUIT_API remove_file(const std::string &path);

     bool CONDUIT_API remove_directory(const std::string &path);

//-----------------------------------------------------------------------------
     int  CONDUIT_API system_execute(const std::string &cmd);


//-----------------------------------------------------------------------------
/// Helpers for escaping / unescaping special characters in strings.
///
/// Our main use case for escaping is json, so we support the escape rules 
/// outlined by the json standard (see: http://www.json.org/). 
///
/// List of supported special characters. 
///    " (quote)
///    \ (backward slash)
///    \n (newline)
///    \t (tab)
///    \b (backspace)
///    \f (form feed)
///    \r (carriage return)
///
/// Special chars that are not escaped, but are unescaped:
///    / (forward slash)
///
/// Special chars that are not escaped or unescaped:
///    \u (for hex escapes: \uFFFF)
/// 
//-----------------------------------------------------------------------------
    std::string CONDUIT_API escape_special_chars(const std::string &input);
    std::string CONDUIT_API unescape_special_chars(const std::string &input);



//-----------------------------------------------------------------------------
/// Base64 Encoding of Buffers 
//-----------------------------------------------------------------------------
    void CONDUIT_API base64_encode(const void *src,
                                   index_t src_nbytes,
                                   void *dest);

    index_t CONDUIT_API base64_encode_buffer_size(index_t src_nbytes);

    index_t CONDUIT_API base64_decode_buffer_size(index_t encoded_nbytes);


    void CONDUIT_API base64_decode(const void *src,
                                   index_t src_nbytes,
                                   void *dest);

//-----------------------------------------------------------------------------
     std::string CONDUIT_API json_sanitize(const std::string &json);
     
//----------------------------------------------------------------------------- 
     // declare then define to avoid icc warnings
     template< typename T >
     std::string to_hex_string(T value);

     template< typename T >
     std::string to_hex_string(T value)
     {
           std::stringstream oss;
           oss << std::hex << value;
           return  oss.str();
     }

//-----------------------------------------------------------------------------
// floating point to string helper, strikes a balance of what we want 
// for format-wise for debug printing and json.
//-----------------------------------------------------------------------------
    std::string CONDUIT_API float64_to_string(float64 value);

//-----------------------------------------------------------------------------
     void CONDUIT_API indent(std::ostream &os,
                             index_t indent,
                             index_t depth,
                             const std::string &pad);
                             
//-----------------------------------------------------------------------------
     void CONDUIT_API sleep(index_t milliseconds);

//----------------------------------------------------------------------------- 
// String hash functions
//----------------------------------------------------------------------------- 
     unsigned int CONDUIT_API hash(const char *k, 
                                   unsigned int length,
                                   unsigned int initval = 0);
     unsigned int CONDUIT_API hash(const char *k, 
                                   unsigned int initval = 0);
     unsigned int CONDUIT_API hash(const std::string &k, 
                                   unsigned int initval = 0);

}
//-----------------------------------------------------------------------------
// -- end conduit::utils --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
