//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://cyrush.github.io/conduit.
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
/// file: gpref_smoke.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include "conduit.hpp"
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

