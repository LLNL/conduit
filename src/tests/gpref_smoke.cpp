//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: gpref_smoke.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include "conduit.h"
#include "gperftools/heap-checker.h"
#include "gperftools/heap-profiler.h"
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(gperf_smoke, heap_check)
{
    //
    // To run successfully the env var HEAPCHECK  must be set to "local"
    //
   
    HeapLeakChecker checker_a("leak_check_000");  
    {
    float64 *vals_a = new float64[1000];
    delete [] vals_a;
    }
    
    EXPECT_TRUE(checker_a.NoLeaks());
    
    HeapLeakChecker checker_b("leak_check_001");  
    {
    float64 *vals_b = new float64[1000];
    }
    
    EXPECT_FALSE(checker_b.NoLeaks());    
    EXPECT_EQ(checker_b.BytesLeaked(),8000);
    EXPECT_EQ(checker_b.ObjectsLeaked(),1);
    
    HeapProfilerStart("heap_prof");
    {

    float64 *vals_c = new float64[1000];
    HeapProfilerDump("check");
    }
    HeapProfilerStop();
    
}

