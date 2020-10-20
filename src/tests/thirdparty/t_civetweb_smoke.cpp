// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_civetweb_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
// civetweb includes
//-----------------------------------------------------------------------------
#include "civetweb.h"

//-----------------------------------------------------------------------------
//
// Note: This just tests that we can compile and link with civetweb, it 
// does not start a web server
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int
dummy_civetweb_handler(struct mg_connection *conn, void *cbdata)
{
	mg_printf(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n");
    return 0;
}

TEST(civetweb_smoke, basic_use )
{
    EXPECT_TRUE(true);
}

