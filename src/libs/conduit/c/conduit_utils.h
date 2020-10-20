// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_utils.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_UTILS_H
#define CONDUIT_UTILS_H

//-----------------------------------------------------------------------------
// -- includes for the public conduit c interface -- 
//-----------------------------------------------------------------------------

#include "conduit_node.h"
#include "conduit_utils.h"

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// -- general conduit c interface methods --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Allows other libraries to provide an alternate info message handler.
//-----------------------------------------------------------------------------
CONDUIT_API void conduit_utils_set_info_handler( void(*on_info)
                                                     (const char *,
                                                      const char *,
                                                      int));

//-----------------------------------------------------------------------------
/// Allows other libraries to provide an alternate warning handler.
//-----------------------------------------------------------------------------
CONDUIT_API void conduit_utils_set_warning_handler( void(*on_warning)
                                                         (const char *,
                                                          const char *,
                                                          int));

//-----------------------------------------------------------------------------
/// Allows other libraries to provide an alternate error handler.
//-----------------------------------------------------------------------------
CONDUIT_API void conduit_utils_set_error_handler( void(*on_error)
                                                       (const char *,
                                                        const char *,
                                                        int));

#ifdef __cplusplus
}
#endif
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------


#endif
