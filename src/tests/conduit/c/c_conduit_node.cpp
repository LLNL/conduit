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
// For details, see: http://llnl.github.io/conduit/.
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
/// file: conduit_node.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"

#include <stdio.h>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(c_conduit_node, simple)
{
    conduit_node *n = conduit_node_create();
    
    EXPECT_EQ(conduit_node_is_root(n),1);
    
    conduit_node_set_int(n,10);
    
    EXPECT_EQ(conduit_node_as_int(n),10);
    int *int_ptr = conduit_node_as_int_ptr(n);
    EXPECT_EQ(int_ptr[0],10);
    
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}


//-----------------------------------------------------------------------------
TEST(c_conduit_node, simple_hier)
{
    int     a_val  = 10;
    int     b_val  = 20;
    double  c_val  = 30.0;
    
    conduit_node *n = conduit_node_create();
    
    conduit_node *a = conduit_node_fetch(n,"a");
    conduit_node *b = conduit_node_fetch(n,"b");
    conduit_node *c = conduit_node_fetch(n,"c");

    EXPECT_EQ(conduit_node_is_root(n),1);
    
    EXPECT_EQ(conduit_node_is_root(a),0);
    EXPECT_EQ(conduit_node_is_root(b),0);
    EXPECT_EQ(conduit_node_is_root(c),0);
    
    conduit_node_set_int(a,a_val);
    conduit_node_set_int(b,b_val);
    conduit_node_set_double(c,c_val);

    
    EXPECT_EQ(conduit_node_as_int(a),a_val);
    EXPECT_EQ(conduit_node_as_int(b),b_val);
    EXPECT_EQ(conduit_node_as_double(c),c_val);
        
    int    *a_ptr = conduit_node_as_int_ptr(a);
    int    *b_ptr = conduit_node_as_int_ptr(b);
    double *c_ptr = conduit_node_as_double_ptr(c);
    
    EXPECT_EQ(a_ptr[0],a_val);
    EXPECT_EQ(b_ptr[0],b_val);
    EXPECT_EQ(c_ptr[0],c_val);
    
    conduit_node_print(n);
    
    /// these are no-ops
    conduit_node_destroy(a);
    conduit_node_destroy(b);
    conduit_node_destroy(c);

    conduit_node_print(n);
    
    /// this actually deletes the node
    conduit_node_destroy(n);
}


