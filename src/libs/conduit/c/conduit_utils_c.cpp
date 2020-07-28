// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.
#include "conduit_utils.h"
#include <conduit_utils.hpp>

static void(*conduit_utils_on_info)(const char *, const char *, int) = NULL;
static void(*conduit_utils_on_warning)(const char *, const char *, int) = NULL;
static void(*conduit_utils_on_error)(const char *, const char *, int) = NULL;

static void conduit_utils_on_info_thunk(const std::string &v0,
                                        const std::string &v1,
                                        int v2)
{
    if(conduit_utils_on_info != NULL)
    {
        (*conduit_utils_on_info)(v0.c_str(), v1.c_str(), v2);
    }
}

static void conduit_utils_on_warning_thunk(const std::string &v0,
                                           const std::string &v1,
                                           int v2)
{
    if(conduit_utils_on_warning != NULL)
    {
        (*conduit_utils_on_warning)(v0.c_str(), v1.c_str(), v2);
    }
}

static void conduit_utils_on_error_thunk(const std::string &v0,
                                         const std::string &v1,
                                         int v2)
{
    if(conduit_utils_on_error != NULL)
    {
        (*conduit_utils_on_error)(v0.c_str(), v1.c_str(), v2);
    }
}

//-----------------------------------------------------------------------------
void
conduit_utils_set_info_handler( 
    void(*on_info)(const char *, const char *, int))
{
    conduit_utils_on_info = on_info;
    conduit::utils::set_info_handler(conduit_utils_on_info_thunk);
}

//-----------------------------------------------------------------------------
void
conduit_utils_set_warning_handler( 
    void(*on_warning)(const char *, const char *, int))
{
    conduit_utils_on_warning = on_warning;
    conduit::utils::set_warning_handler(conduit_utils_on_warning_thunk);
}

//-----------------------------------------------------------------------------
void
conduit_utils_set_error_handler( 
    void(*on_error)(const char *, const char *, int))
{
    conduit_utils_on_error = on_error;
    conduit::utils::set_error_handler(conduit_utils_on_error_thunk);
}

