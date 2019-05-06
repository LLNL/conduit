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

//-----------------------------------------------------------------------------
///
/// file: adios_smoke.cpp
///
//-----------------------------------------------------------------------------

// Serial test only.
#define _NOMPI

#include <adios.h>
#include <iostream>
#include "gtest/gtest.h"

// Adapted from ADIOS' global aggregate by color test.

//-----------------------------------------------------------------------------
TEST(adios_smoke, basic_use)
{
    int rank = 0;
    int comm = 0;
    int NX = 5;
    const char *filename = "adios_smoke.bp";
    int64_t m_adios_group;
    int64_t m_adios_file;

    int status = adios_init_noxml(comm);
    EXPECT_TRUE(status >= 0 );

    adios_set_max_buffer_size (10);

    status = adios_declare_group(&m_adios_group, "restart", "iter", adios_stat_default);
    EXPECT_TRUE(status >= 0 );

    status = adios_select_method(m_adios_group, "POSIX", "", "");
    EXPECT_TRUE(status >= 0 );

    adios_define_var(m_adios_group, "NX",
                     "", adios_integer,
                     0, 0, 0);

    status = adios_open(&m_adios_file, "restart", filename, "w", comm);

    status = adios_write(m_adios_file, "NX", (void *) &NX);
    EXPECT_TRUE(status >= 0 );

    status = adios_close(m_adios_file);
    EXPECT_TRUE(status >= 0 );

    status = adios_finalize(rank);
    EXPECT_TRUE(status >= 0 );
}
