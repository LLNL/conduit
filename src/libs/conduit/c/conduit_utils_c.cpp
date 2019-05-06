//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include <conduit_utils.h>
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

