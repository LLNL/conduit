///
/// file: Utils.cpp
///

#include "Utils.h"

namespace conduit
{

namespace utils
{

void     
split_path(const std::string &path,
           std::string &curr,
           std::string &next)
{
    curr.clear();
    next.clear();
    std::size_t found = path.find("/");
    if (found != std::string::npos)
    {
        curr = path.substr(0,found);
        if(found != path.size()-1)
            next = path.substr(found+1,path.size()-(found-1));
    }
    else
    {
        curr = path;
    }
}



}
}
