// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: silo_smoke.cpp
///
//-----------------------------------------------------------------------------

#include <silo.h>
#include <iostream>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(silo_smoke, basic_use)
{


    DBfile *dbfile = DBCreate("silo_smoke_test.silo", 0, DB_LOCAL, "test", DB_HDF5);
    std::string twrite = "test_string";
    int twrite_len = twrite.size()+1;
    DBWrite (dbfile, "tdata", twrite.c_str(), &twrite_len, 1, DB_CHAR);
    DBClose(dbfile);
    
    dbfile = DBOpen("silo_smoke_test.silo", DB_HDF5, DB_READ);
    
    
    int tread_len  = DBGetVarLength(dbfile, "tdata");
    char  *tread = new char[tread_len];
    DBReadVar(dbfile, "tdata", tread);
    DBClose(dbfile);
    
    EXPECT_EQ(twrite,std::string(tread));
    delete [] tread;
}
