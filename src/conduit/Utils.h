///
/// file: Utils.h
///

#ifndef __CONDUIT_UTILS_H
#define __CONDUIT_UTILS_H

#include <string>

namespace conduit
{

namespace utils
{
     void     split_path(const std::string &path,
                         std::string &curr,
                         std::string &next);

     std::string json_sanitize(const std::string &json);
}
}

#endif
