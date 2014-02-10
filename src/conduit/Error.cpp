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
: m_msg("<empty>")
{}

///============================================
Error::Error(const Error &err)
: m_msg(err.m_msg)
{
    print();
}

///============================================
Error::Error(const std::string &msg)
: m_msg(msg)
{
    print();
}

///============================================
Error::Error(const std::ostringstream &oss)
: m_msg(oss.str())
{
    print();
}

///============================================
Error::~Error()
{

}

///============================================
void
Error::print() const
{
    std::cout << "Error: "  << m_msg << std::endl;
}

}

