///
/// file: Error.h
///

#ifndef __CONDUIT_ERROR_H
#define __CONDUIT_ERROR_H

#include "Core.h"
#include <string>
#include <sstream>

namespace conduit
{

class Error
{
public:    
    /* Constructors */
    Error(); // empty error msg
    Error(const std::string &msg);
    Error(const std::ostringstream &msg);
    Error(const Error &err);

    /* Destructor */
    virtual  ~Error();

    void        print() const;
    std::string message() const { return m_msg;}

private:
    std::string m_msg;
      
};

}


#endif
