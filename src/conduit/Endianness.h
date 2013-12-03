///
/// file: Endianness.h
///

#ifndef __CONDUIT_ENDIANNESS_H
#define __CONDUIT_ENDIANNESS_H

#include "Core.h"
#include <vector>
#include <sstream>

namespace conduit
{

class Endianness
{
public:
    typedef enum
    {
        DEFAULT_T = 0, // default
        BIG_T,
        LITTLE_T,
    } EndianEnum;

    static index_t          machine_default();
    static index_t          name_to_id(const std::string &name);
    static std::string      id_to_name(index_t endianness);
    
    // basic swap routines
    static void             swap16(void *data);
    static void             swap16(void *src,void *dest);
    
    static void             swap32(void *data);
    static void             swap32(void *src,void *dest);
    
    static void             swap64(void *data);
    static void             swap64(void *src,void *dest);

};

};


#endif
