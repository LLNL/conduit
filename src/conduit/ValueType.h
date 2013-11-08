///
/// file: ValueArray.h
///

#ifndef __CONDUIT_VALUE_TYPE_H
#define __CONDUIT_VALUE_TYPE_H


#include "conduit.h"

namespace conduit
{

class ValueType
{
public:
    typedef enum
    {
        EMPTY_T = 0,
        UNKNOWN_T,
        NODE_T,
        LIST_T,
        UINT32_T,
        UINT64_T,
        FLOAT64_T,
        BYTESTR_T,
    } ValueTypeEnum;
    

    static index_t     id(const std::string &name);
    static std::string name(index_t dtype);

};

};

#endif