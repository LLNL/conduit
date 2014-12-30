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
/// file: Error.h
///
//-----------------------------------------------------------------------------

#ifndef __CONDUIT_ERROR_H
#define __CONDUIT_ERROR_H

//-----------------------------------------------------------------------------
// -- conduit library includes -- 
//-----------------------------------------------------------------------------
#include "Core.h"

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <string>
#include <sstream>

//-----------------------------------------------------------------------------
//
/// The THROW_ERROR macro is the primary mechanism used to capture errors  
/// in conduit. It currently throws a c++ exception, in the form of a
/// conduit::Error class. If we decide to not use exceptions in the future
/// we can simply change the behavior of this macro.
///
//-----------------------------------------------------------------------------
#define THROW_ERROR( msg )                                          \
{                                                                   \
    std::ostringstream oss;                                         \
    oss << msg;                                                     \
    throw conduit::Error( oss.str(), __FILE__, __LINE__);           \
}                                                                   \

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::Error --
//-----------------------------------------------------------------------------
///
/// class: conduit::Error
///
/// description:
///  Class used to record a runtime error in conduit.
///
//-----------------------------------------------------------------------------
class CONDUIT_API Error : public std::exception
{
public:    
//-----------------------------------------------------------------------------
//
// -- conduit::Error public members --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Construction and Destruction
//-----------------------------------------------------------------------------
    /// Default constructor
    Error();
    /// Copy constructor
    Error(const Error &err);
    /// Main constructor, used by THROW_ERROR macro
    Error(const std::string &msg, 
          const std::string &file,
          index_t line);
    /// Destructor
    ~Error() throw();
    
//-----------------------------------------------------------------------------
// Methods used to access and display Error as a human readable string. 
//-----------------------------------------------------------------------------
    /// Print info about this error to stdout.
    void                print()   const;
    /// Return a human readable string that describes this error.
    std::string         message() const;
    /// Writes a human readable string that describes this error to the 
    /// passed output stream.
    void                message(std::ostringstream &oss) const;
    /// part of std::exception interface
    virtual const char* what() const throw() 
                        { return message().c_str(); }

private:
//-----------------------------------------------------------------------------
//
// -- conduit::Error private data members --
//
//-----------------------------------------------------------------------------
    /// holds the error message
    std::string m_msg;
    /// holds the source file name where the error was raised
    std::string m_file;
    /// holds the line number in the source file where the error was raised.
    index_t     m_line;
    
};
//-----------------------------------------------------------------------------
// -- end conduit::Error --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif
