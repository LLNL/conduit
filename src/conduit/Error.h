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

class Error 
{
public:    
    /* Constructors */
    Error();
    Error(const std::string &msg, 
          const std::string &file,
          index_t line);
    Error(const Error &err);

    /* Destructor */
    virtual  ~Error();

    void        print()   const;
    std::string message() const { return m_msg;}

private:
    std::string m_msg;
    std::string m_file;
    index_t     m_line;
    
};


}


#endif
