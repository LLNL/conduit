///
/// file: Error.h
///

#ifndef __CONDUIT_ERROR_H
#define __CONDUIT_ERROR_H

#include "Core.h"
#include <string>
#include <sstream>

#define THROW_ERROR( msg )                                          \
{                                                                   \
    std::ostringstream oss;                                         \
    oss << msg;                                                     \
    throw conduit::Error( oss.str(), __FILE__, __LINE__);           \
}                                                                   \

namespace conduit
{

class Error : public std::exception
{
public:    
    /* Constructors */
    Error();
    Error(const std::string &msg, 
          const std::string &file,
          index_t line);
    Error(const Error &err);

    /* Destructor */
    virtual  ~Error() throw();

    void        print()   const;
    std::string message() const;
    void        message(std::ostringstream &oss) const;
    virtual const char* what() const throw() { return message().c_str();}

private:
    std::string m_msg;
    std::string m_file;
    index_t     m_line;
    
};


}


#endif
