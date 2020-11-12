// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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
