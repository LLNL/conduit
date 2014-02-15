///
/// file: Error.cpp
///

#include "Error.h"

namespace conduit
{


///============================================
/// Error
///============================================

///============================================
Error::Error()
:m_msg(""),
 m_file(""),
 m_line(0)
{}

///============================================
Error::Error(const Error &err)
:m_msg(err.m_msg),
 m_file(err.m_file),
 m_line(err.m_line)

{
}

///============================================
Error::Error(const std::string &msg,
             const std::string &file,
             index_t line)
:m_msg(msg),
 m_file(file),
 m_line(line)
{
}

///============================================
Error::~Error()
{}


///============================================
void
Error::print() const
{
    std::string msg = m_msg;
    if(msg == "")
        msg = "<EMPTY>";
    std::cout << "[" << m_file << ":" << m_line <<"]Error: " << msg  << std::endl;
}

}

