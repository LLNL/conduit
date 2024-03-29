// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_config.hpp
///
//-----------------------------------------------------------------------------

//
// C++ specific config values for conduit
//
// These settings are C++ only and require a C++11 compiler.
// They are here to avoid undermining c compilers from using our c header.
//

#ifndef CONDUIT_CONFIG_HPP
#define CONDUIT_CONFIG_HPP

#include "conduit_config.h"

//-----------------------------------------------------------------------------
// if built with c++14 support, make sure a c++14 compiler is used
//-----------------------------------------------------------------------------
#if defined(CONDUIT_USE_CXX14)
    #if defined(_MSC_VER)
        #if _MSC_VER < 1900
            #error Conduit was built with c++14 support, please use a c++14 compliant compiler
        #endif
    #elif __cplusplus < 201103L
        #error Conduit was built with c++14 support, please use a c++14 compliant compiler
    #endif
#endif

#endif
