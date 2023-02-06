// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_c_conduit_datatype.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"

#include <stdio.h>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(c_conduit_datatype, sizeof_index_t)
{
    int sz_it = conduit_datatype_sizeof_index_t();

#ifdef CONDUIT_INDEX_32
    EXPECT_EQ(sz_it,4);
#else
    EXPECT_EQ(sz_it,8);
#endif

}

