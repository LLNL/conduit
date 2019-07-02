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
    
    EXPECT_TRUE(conduit_node_is_root(n));
    
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

    EXPECT_TRUE(conduit_node_is_root(n));
    
    EXPECT_FALSE(conduit_node_is_root(a));
    EXPECT_FALSE(conduit_node_is_root(b));
    EXPECT_FALSE(conduit_node_is_root(c));
    
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

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_diff)
{
    int     a_val  = 10;
    int     b_val  = 20;
    double  c_val  = 30.0;
    
    conduit_node *ninfo = conduit_node_create();
    
    conduit_node *n1 = conduit_node_create();
    
    conduit_node *a1 = conduit_node_fetch(n1,"a");
    conduit_node *b1 = conduit_node_fetch(n1,"b");
    conduit_node *c1 = conduit_node_fetch(n1,"c");

    conduit_node_set_int(a1,a_val);
    conduit_node_set_int(b1,b_val);
    conduit_node_set_double(c1,c_val);

    conduit_node *n2 = conduit_node_create();
    
    conduit_node *a2 = conduit_node_fetch(n2,"a");
    conduit_node *b2 = conduit_node_fetch(n2,"b");
    conduit_node *c2 = conduit_node_fetch(n2,"c");

    conduit_node_set_int(a2,a_val);
    conduit_node_set_int(b2,b_val);
    conduit_node_set_double(c2,c_val);
    
    // no diff
    EXPECT_FALSE(conduit_node_diff(n1,n2,ninfo,1e-12));

    conduit_node_set_double(c2,32.0);

    // there is a diff
    EXPECT_TRUE(conduit_node_diff(n1,n2,ninfo,1e-12));
    
    conduit_node_set_double(c2,30.0);
    
    conduit_node *d2 = conduit_node_fetch(n2,"d");
    conduit_node_set_double(d2,42.0);

    // there is diff
    EXPECT_TRUE(conduit_node_diff(n1,n2,ninfo,1e-12));
    // no diff compat
    EXPECT_FALSE(conduit_node_diff_compatible(n1,n2,ninfo,1e-12));
    
    conduit_node_set_double(c2,32.0);
    
    // there is a diff compat
    EXPECT_TRUE(conduit_node_diff_compatible(n1,n2,ninfo,1e-12));
    
    conduit_node_destroy(n1);
    conduit_node_destroy(n2);
    conduit_node_destroy(ninfo);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_update)
{
    int     a_val  = 10;
    int     b_val  = 20;
    double  c_val  = 30.0;
    
    conduit_node *ninfo = conduit_node_create();
    
    conduit_node *n1 = conduit_node_create();
    
    conduit_node *a1 = conduit_node_fetch(n1,"a");
    conduit_node *b1 = conduit_node_fetch(n1,"b");
    conduit_node *c1 = conduit_node_fetch(n1,"c");

    conduit_node_set_int(a1,a_val);
    conduit_node_set_int(b1,b_val);
    conduit_node_set_double(c1,c_val);

    conduit_node *n2 = conduit_node_create();
    
    conduit_node *a2 = conduit_node_fetch(n2,"a");
    conduit_node *b2 = conduit_node_fetch(n2,"b");
    conduit_node *c2 = conduit_node_fetch(n2,"c");

    conduit_node_set_int(a2,a_val);
    conduit_node_set_int(b2,b_val);
    conduit_node_set_double(c2,42.0);
    
    conduit_node_update(n1,n2);

    double *c1_ptr = conduit_node_as_double_ptr(c1);

    EXPECT_EQ(c1_ptr[0],42.0);

    conduit_node_rename_child(n2,"c","d");
    
    conduit_node *d2 = conduit_node_fetch(n2,"d");

    double *d2_ptr = conduit_node_as_double_ptr(d2);

    EXPECT_EQ(d2_ptr[0],42.0);

    conduit_node_update_compatible(n1,n2);
    EXPECT_FALSE(conduit_node_has_path(n1,"d"));

    conduit_node_update_external(n1,n2);

    conduit_node *d1 = conduit_node_fetch(n1,"d");
    double *d1_ptr = conduit_node_as_double_ptr(d1);

    d2_ptr[0] = 52.0;

    // these should be the same address
    EXPECT_EQ(d1_ptr,d2_ptr);
    EXPECT_EQ(d1_ptr[0],52.0);

    conduit_node_destroy(n1);
    conduit_node_destroy(n2);
    conduit_node_destroy(ninfo);

}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_compatible)
{
    int     a_val  = 10;
    double  b_val  = 30.0;

    conduit_node *n1 = conduit_node_create();
    conduit_node *n2 = conduit_node_create();

    conduit_node *a1 = conduit_node_fetch(n1,"a");
    conduit_node *a2 = conduit_node_fetch(n2,"a");
    
    conduit_node_set_int(a1,a_val);
    conduit_node_set_double(a2,b_val);

    EXPECT_FALSE(conduit_node_compatible(n1,n2));

    conduit_node_rename_child(n2,"a","b");
    // it will be compat after we rename a to b
    EXPECT_TRUE(conduit_node_compatible(n1,n2));

    conduit_node_set_path_int(n2,"a",a_val);
    // still true!
    EXPECT_TRUE(conduit_node_compatible(n1,n2));

    conduit_node_print(n1);
    conduit_node_print(n2);

    conduit_node_destroy(n1);
    conduit_node_destroy(n2);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_compact)
{
    conduit_int32    a_val  = 10;
    conduit_int32    b_val  = 20;
    conduit_float64  c_val  = 30.0;

    conduit_node *n = conduit_node_create();

    conduit_node *a = conduit_node_fetch(n,"a");
    conduit_node *b = conduit_node_fetch(n,"b");
    conduit_node *c = conduit_node_fetch(n,"c");
    
    conduit_node_set_int32(a,a_val);
    conduit_node_set_int32(b,b_val);
    conduit_node_set_float64(c,c_val);

    EXPECT_EQ(conduit_node_total_bytes_allocated(n),16);

    EXPECT_FALSE(conduit_node_is_contiguous(n));

    conduit_node *n_c = conduit_node_create();

    conduit_node_compact_to(n, n_c);

    EXPECT_TRUE(conduit_node_is_contiguous(n_c));

    EXPECT_EQ(conduit_node_total_bytes_compact(n_c),16);
    EXPECT_EQ(conduit_node_total_strided_bytes(n_c),16);
    EXPECT_EQ(conduit_node_total_bytes_allocated(n_c),16);

    conduit_node_destroy(n);
    conduit_node_destroy(n_c);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_name_and_path)
{
    conduit_node *n = conduit_node_create();

    conduit_node *a = conduit_node_fetch(n,"a");
    conduit_node *b = conduit_node_fetch(n,"b");
    conduit_node *c = conduit_node_fetch(n,"c");

    conduit_node *e = conduit_node_fetch(n,"d/e");

    char *a_name = conduit_node_name(a);
    char *a_path = conduit_node_path(a);

    char *e_name = conduit_node_name(e);
    char *e_path = conduit_node_path(e);

    EXPECT_EQ(std::string(a_name),"a");
    EXPECT_EQ(std::string(a_path),"a");

    EXPECT_EQ(std::string(e_name),"e");
    EXPECT_EQ(std::string(e_path),"d/e");

    free(a_name);
    free(a_path);

    free(e_name);
    free(e_path);

    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_remove)
{
    int     a_val  = 10;
    int     b_val  = 20;
    double  c_val  = 30.0;
    
    conduit_node *n = conduit_node_create();
    
    conduit_node *a = conduit_node_fetch(n,"a");
    conduit_node *b = conduit_node_fetch(n,"b");
    conduit_node *c = conduit_node_fetch(n,"c");

    conduit_node_set_int(a,a_val);
    conduit_node_set_int(b,b_val);
    conduit_node_set_double(c,c_val);
    
    // remove child by index
    EXPECT_TRUE(conduit_node_has_path(n,"a"));
    conduit_node_remove_child(n,0);
    EXPECT_FALSE(conduit_node_has_path(n,"a"));
    EXPECT_TRUE(conduit_node_has_path(n,"b"));
    EXPECT_TRUE(conduit_node_has_path(n,"c"));

    // remove child by path
    conduit_node_remove_path(n,"b");
    EXPECT_FALSE(conduit_node_has_path(n,"a"));
    EXPECT_FALSE(conduit_node_has_path(n,"b"));
    EXPECT_TRUE(conduit_node_has_path(n,"c"));

    // rename
    conduit_node_rename_child(n,"c","a");
    EXPECT_FALSE(conduit_node_has_path(n,"c"));
    EXPECT_TRUE(conduit_node_has_path(n,"a"));

    conduit_node_destroy(n);
}

