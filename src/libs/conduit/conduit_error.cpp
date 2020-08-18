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
/// file: conduit_error.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_error.hpp"

//-----------------------------------------------------------------------------
// -- conduit includes -- 
//-----------------------------------------------------------------------------
#include "conduit_node.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
//
// -- conduit::Error public members --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Error Construction and Destruction
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Error::Error()
:m_msg(""),
 m_file(""),
 m_line(0),
 m_what("")
{
    m_what = message();
}

//---------------------------------------------------------------------------//
Error::Error(const Error &err)
:m_msg(err.m_msg),
 m_file(err.m_file),
 m_line(err.m_line),
 m_what("")
{
    m_what = message();
}

//---------------------------------------------------------------------------//
Error::Error(const std::string &msg,
             const std::string &file,
             index_t line)
:m_msg(msg),
 m_file(file),
 m_line(line),
 m_what("")
{
    m_what = message();
}

//---------------------------------------------------------------------------//
Error::~Error() throw()
{
    //empty
}

//-----------------------------------------------------------------------------
// Methods used to access and display Error as a human readable string. 
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
void
Error::print() const
{
    std::cout << message() << std::endl;
}


//---------------------------------------------------------------------------//
std::string
Error::message() const
{
    std::ostringstream oss;
    message(oss);
    return oss.str();
}

//---------------------------------------------------------------------------//
void
Error::message(std::ostringstream &oss) const
{
    oss << std::endl;
    oss << "file: " << m_file << std::endl;
    oss << "line: " << m_line << std::endl;
    oss << "message: " << std::endl;
    oss << m_msg << std::endl;
}

//---------------------------------------------------------------------------//
const char* 
Error::what() const throw()
{
    return m_what.c_str(); 
}


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

