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
/// file: t_c_blueprint_mcarray.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"
#include "conduit_blueprint.h"

#include <stdio.h>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(c_conduit_blueprint_mcarray, create_and_verify)
{
    conduit_node *n      = conduit_node_create();
    conduit_node *nxform = conduit_node_create();
    conduit_node *nempty = conduit_node_create();
    conduit_node *info   = conduit_node_create();

    conduit_blueprint_mcarray_examples_xyz("interleaved",10,n);
    EXPECT_TRUE(conduit_blueprint_mcarray_verify(n,info));
    EXPECT_TRUE(conduit_blueprint_mcarray_is_interleaved(n));

    EXPECT_TRUE(conduit_blueprint_mcarray_to_contiguous(n,nxform));
    EXPECT_FALSE(conduit_blueprint_mcarray_is_interleaved(nxform));
    EXPECT_TRUE(conduit_node_is_contiguous(nxform));

    conduit_blueprint_mcarray_examples_xyz("separate",10,n);
    EXPECT_TRUE(conduit_blueprint_mcarray_verify(n,info));

    conduit_blueprint_mcarray_examples_xyz("contiguous",10,n);
    EXPECT_TRUE(conduit_blueprint_mcarray_verify(n,info));
    EXPECT_TRUE(conduit_node_is_contiguous(n));
    EXPECT_FALSE(conduit_blueprint_mcarray_is_interleaved(n));

    EXPECT_TRUE(conduit_blueprint_mcarray_to_interleaved(n,nxform));
    conduit_node_print_detailed(nxform);
    EXPECT_TRUE(conduit_blueprint_mcarray_is_interleaved(nxform));

    conduit_blueprint_mcarray_examples_xyz("interleaved_mixed",10,n);
    EXPECT_TRUE(conduit_blueprint_mcarray_verify(n,info));
    
    EXPECT_FALSE(conduit_blueprint_mcarray_verify_sub_protocol("sub",n,info));
    EXPECT_FALSE(conduit_blueprint_mcarray_verify(nempty,info));

    conduit_node_destroy(n);
    conduit_node_destroy(nxform);
    conduit_node_destroy(nempty);
    conduit_node_destroy(info);
}


