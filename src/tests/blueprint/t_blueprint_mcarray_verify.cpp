//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
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
/// file: t_blueprint_mcarray_verify.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mcarray_valid_separate)
{
    Node n, info;

    n["x"].set(DataType::float64(10));
    EXPECT_TRUE(blueprint::mcarray::verify(n,info));

    n["y"].set(DataType::float64(10));
    EXPECT_TRUE(blueprint::mcarray::verify(n,info));

    n["z"].set(DataType::float64(10));
    EXPECT_TRUE(blueprint::mcarray::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mcarray_valid_contiguous)
{
    Node n, info;

    Schema s;
    s["x"].set(DataType::float64(10));
    s["y"].set(DataType::float64(10,10*sizeof(conduit::float64)));
    s["z"].set(DataType::float64(10,20*sizeof(conduit::float64)));
    n.set(s);

    EXPECT_TRUE(blueprint::mcarray::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mcarray_valid_interleaved)
{
    Node n, info;

    Schema s;
    s["x"].set(DataType::float64(10,0*sizeof(conduit::float64),3*sizeof(conduit::float64)));
    s["y"].set(DataType::float64(10,1*sizeof(conduit::float64),3*sizeof(conduit::float64)));
    s["z"].set(DataType::float64(10,2*sizeof(conduit::float64),3*sizeof(conduit::float64)));
    n.set(s);

    EXPECT_TRUE(blueprint::mcarray::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mcarray_invalid_node_type)
{
    Node n, info;

    n.set(0.0f);
    EXPECT_FALSE(blueprint::mcarray::verify(n,info));

    n.set("test");
    EXPECT_FALSE(blueprint::mcarray::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mcarray_invalid_array_types)
{
    Node n, info;

    n.reset();
    n["x"].set(DataType::char8_str(10));
    n["y"].set(DataType::char8_str(10));
    EXPECT_FALSE(blueprint::mcarray::verify(n,info));

    n.reset();
    n["x"].set(DataType::float64(10));
    n["y"].set(DataType::float64(10));
    n["z"].set(DataType::char8_str(10));
    EXPECT_FALSE(blueprint::mcarray::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mcarray_invalid_array_contents)
{
    Node n, info;

    n.reset();
    n["x"].set(DataType::float64(10));
    n["y"].set(DataType::float64(9));
    EXPECT_FALSE(blueprint::mcarray::verify(n,info));

    n.reset();
    n["x"].set(DataType::float64(10));
    n["y"].set(DataType::float64(10));
    n["z"].set(DataType::float64(11));
    EXPECT_FALSE(blueprint::mcarray::verify(n,info));

    n.reset();
    n["x"].set(DataType::float64(10));
    n["y"].set(DataType::float64(10));
    n["m"].set(0.0f);
    EXPECT_FALSE(blueprint::mcarray::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mcarray_verify_with_protocol)
{
    Node n, info;

    EXPECT_FALSE(blueprint::mcarray::verify("protocol",n,info));
    EXPECT_FALSE(blueprint::mcarray::verify("mcarray",n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mlarray_valid_simple)
{
    Node n, info;

    n["a/x"].set(DataType::float64(2));
    n["a/y"].set(DataType::float64(2));
    n["b/x"].set(DataType::float64(2));
    n["b/y"].set(DataType::float64(2));
    EXPECT_TRUE(blueprint::mlarray::verify(n,info));

    n["a/x"].set(DataType::int64(2));
    n["b/x"].set(DataType::int64(2));
    EXPECT_TRUE(blueprint::mlarray::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mlarray_valid_complex)
{
    Node n, info;

    n["a/x/1"].set(DataType::float64(2));
    EXPECT_TRUE(blueprint::mlarray::verify(n,info));
    n["a/x/2"].set(DataType::float64(2));
    EXPECT_TRUE(blueprint::mlarray::verify(n,info));

    n["b/x/1"].set(DataType::float64(2));
    EXPECT_FALSE(blueprint::mlarray::verify(n,info));
    n["b/x/2"].set(DataType::float64(2));
    EXPECT_TRUE(blueprint::mlarray::verify(n,info));

    n["a/y"].set(DataType::float64(2));
    EXPECT_FALSE(blueprint::mlarray::verify(n,info));
    n["b/y"].set(DataType::float64(2));
    EXPECT_FALSE(blueprint::mlarray::verify(n,info));

    n["a/y"].reset();
    n["b/y"].reset();
    n["a/y/1"].set(DataType::float64(2));
    n["a/y/2"].set(DataType::float64(2));
    n["b/y/1"].set(DataType::float64(2));
    n["b/y/2"].set(DataType::float64(2));
    EXPECT_TRUE(blueprint::mlarray::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mlarray_invalid_type)
{
    Node n, info;

    EXPECT_FALSE(blueprint::mlarray::verify(n,info));

    n.set(DataType::char8_str(2));
    EXPECT_FALSE(blueprint::mlarray::verify(n,info));

    n.reset();
    n["a/x"].set(DataType::float64(2));
    n["a/y"].set(DataType::float64(2));
    n["b/x"].set(DataType::float64(2));
    n["b/y"].set(DataType::char8_str(2));
    EXPECT_FALSE(blueprint::mlarray::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mlarray_invalid_structure)
{
    Node n, info;

    n["a/x"].set(DataType::float64(2));
    n["a/y"].set(DataType::float64(2));
    n["b/x"].set(DataType::float64(2));
    EXPECT_FALSE(blueprint::mlarray::verify(n,info));

    n["b/z"].set(DataType::float64(2));
    EXPECT_FALSE(blueprint::mlarray::verify(n,info));

    n["b/y"].set(DataType::float64(2));
    EXPECT_FALSE(blueprint::mlarray::verify(n,info));

    n["a/z"].set(DataType::float64(3));
    EXPECT_FALSE(blueprint::mlarray::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mcarray_verify, mlarray_verify_with_protocol)
{
    Node n, info;

    EXPECT_FALSE(blueprint::mlarray::verify("protocol",n,info));
    EXPECT_FALSE(blueprint::mlarray::verify("mlarray",n,info));
}
