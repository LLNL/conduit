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
/// file: conduit_blueprint_mesh_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "blueprint.hpp"
#include "conduit_io.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mca_examples, verify_mca)
{
    ///
    /// cases we expect to fail
    ///
    
    
    Node n;
    n["here/there"]     = 10;
    n["here/everywhere"] = 10;
    
    EXPECT_FALSE(blueprint::mca::verify_mca(n));

    n.reset();
    n["x"].set(DataType::float64(10));
    n["y"].set(DataType::float64(5));
    
    EXPECT_FALSE(blueprint::mca::verify_mca(n));

    ///
    /// cases we expect to work
    ///
    
    n.reset();
    n["x"].set(DataType::float64(10));
    EXPECT_TRUE(blueprint::mca::verify_mca(n));

    n["y"].set(DataType::float64(10));
    EXPECT_TRUE(blueprint::mca::verify_mca(n));
    
    blueprint::mca::examples::xyz("separate",
                                  5,
                                  n);
    EXPECT_TRUE(blueprint::mca::verify_mca(n));

    blueprint::mca::examples::xyz("interleaved",
                                  5,
                                  n);
    EXPECT_TRUE(blueprint::mca::verify_mca(n));

    blueprint::mca::examples::xyz("contiguous",
                                  5,
                                  n);
    EXPECT_TRUE(blueprint::mca::verify_mca(n));

}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mca_examples, verify_mca_generic)
{
    ///
    /// cases we expect to fail
    ///
    
    
    Node n, info;
    n["here/there"]     = 10;
    n["here/everywhere"] = 10;
    
    EXPECT_FALSE(blueprint::verify("mca",n,info));

    n.reset();
    n["x"].set(DataType::float64(10));
    n["y"].set(DataType::float64(5));
    
    EXPECT_FALSE(blueprint::verify("mca",n,info));

    ///
    /// cases we expect to work
    ///
    
    n.reset();
    n["x"].set(DataType::float64(10));
    EXPECT_TRUE(blueprint::verify("mca",n,info));

    n["y"].set(DataType::float64(10));
    EXPECT_TRUE(blueprint::verify("mca",n,info));
    
    blueprint::mca::examples::xyz("separate",
                                  5,
                                  n);
    EXPECT_TRUE(blueprint::verify("mca",n,info));

    blueprint::mca::examples::xyz("interleaved",
                                  5,
                                  n);
    EXPECT_TRUE(blueprint::verify("mca",n,info));

    blueprint::mca::examples::xyz("contiguous",
                                  5,
                                  n);
    EXPECT_TRUE(blueprint::verify("mca",n,info));
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mca_examples, mca_test_to_contig)
{
    
    Node n;

    index_t nvals = 5; // Number of "tuples"
    
    blueprint::mca::examples::xyz("separate",
                                  nvals,
                                  n);
    
    n.print();
    
    n.info().print();
    
    Node n_info;
    n.info(n_info);
    EXPECT_EQ(n_info["mem_spaces"].number_of_children(),3);
    
    Node n_out;
    blueprint::mca::to_contiguous(n,n_out);
    n_out.print();
    n_out.info().print();
    
    n_out.info(n_info);
    
    EXPECT_EQ(n_info["mem_spaces"].number_of_children(),1);
    
    EXPECT_TRUE(blueprint::mca::is_contiguous(n_out));
    EXPECT_FALSE(blueprint::mca::is_interleaved(n_out));    
    Node n_test;
    n_test.set_external((float64*)n_out.data_ptr(),15);
    n_test.print();
    
    float64 *n_test_ptr  = n_test.value();
    
    for(index_t i=0;i<5;i++)
    {
        EXPECT_NEAR(n_test_ptr[i],1.0,1e-5);
    }

    for(index_t i=5;i<10;i++)
    {
        EXPECT_NEAR(n_test_ptr[i],2.0,1e-5);
    }

    for(index_t i=10;i<15;i++)
    {
        EXPECT_NEAR(n_test_ptr[i],3.0,1e-5);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mca_examples, mca_test_to_interleaved)
{
    
    Node n;
    
    index_t nvals = 5; // Number of "tuples"
    blueprint::mca::examples::xyz("separate",
                                  nvals,
                                  n);
    
    n.print();
    
    n.info().print();
    
    Node n_info;
    n.info(n_info);
    EXPECT_EQ(n_info["mem_spaces"].number_of_children(),3);
    
    Node n_out;
    blueprint::mca::to_interleaved(n,n_out);
    n_out.print();
    n_out.info().print();
    
    n_out.info(n_info);
    
    EXPECT_EQ(n_info["mem_spaces"].number_of_children(),1);
    
    EXPECT_FALSE(blueprint::mca::is_contiguous(n_out));
    EXPECT_TRUE(blueprint::mca::is_interleaved(n_out));    
    
    Node n_test;
    n_test.set_external((float64*)n_out.data_ptr(),15);
    n_test.print();
    
    float64 *n_test_ptr  = n_test.value();
    
    
    for(index_t i=0;i<5;i++)
    {
        EXPECT_NEAR(n_test_ptr[i*3+0],1.0,1e-5);
        EXPECT_NEAR(n_test_ptr[i*3+1],2.0,1e-5);
        EXPECT_NEAR(n_test_ptr[i*3+2],3.0,1e-5);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mca_examples, mca_aos_to_contigious)
{
    struct uvw
    {
        float64 u;
        float64 v;
        float64 w;
    };
    
    
    uvw vel[5];
    
    for(index_t i=0;i<5;i++)
    {
        vel[i].u  = 1.0;
        vel[i].v  = 2.0;
        vel[i].w  = 3.0;
    }
    
    index_t stride = sizeof(uvw);
    CONDUIT_INFO("aos stride: " << stride);
    Node n;
    
    n["u"].set_external(&vel[0].u,5,0,stride);
    n["v"].set_external(&vel[0].v,5,0,stride);
    n["w"].set_external(&vel[0].w,5,0,stride);
    
    n.print();
    
    Node n_out;
    blueprint::mca::to_contiguous(n,n_out);
    n_out.print();
    
    Node n_test;
    n_test.set_external((float64*)n_out.data_ptr(),15);
    n_test.print();
    
    float64 *n_test_ptr  = n_test.value();
    
    for(index_t i=0;i<5;i++)
    {
        EXPECT_NEAR(n_test_ptr[i],1.0,1e-5);
    }

    for(index_t i=5;i<10;i++)
    {
        EXPECT_NEAR(n_test_ptr[i],2.0,1e-5);
    }

    for(index_t i=10;i<15;i++)
    {
        EXPECT_NEAR(n_test_ptr[i],3.0,1e-5);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mca_examples, mca_soa_to_interleaved)
{

    struct uvw
    {
        float64 us[5];
        float64 vs[5];
        float64 ws[5];
    };
    
    uvw vel;
    
    for(index_t i=0;i<5;i++)
    {
        vel.us[i] = 1.0;
        vel.vs[i] = 2.0;
        vel.ws[i] = 3.0;
    }

    Node n;
    
    n["u"].set_external(vel.us,5);
    n["v"].set_external(vel.vs,5);
    n["w"].set_external(vel.ws,5);
    
    n.print();
    
    Node n_out;
    blueprint::mca::to_interleaved(n,n_out);
    n_out.print();
    
    Node n_test;
    n_test.set_external((float64*)n_out.data_ptr(),15);
    n_test.print();
    
    float64 *n_test_ptr  = n_test.value();
    
    for(index_t i=0;i<5;i++)
    {
        EXPECT_NEAR(n_test_ptr[i*3+0],1.0,1e-5);
        EXPECT_NEAR(n_test_ptr[i*3+1],2.0,1e-5);
        EXPECT_NEAR(n_test_ptr[i*3+2],3.0,1e-5);
    }
}



//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mca_examples, mca_xyz_contiguous_mixed_types)
{
    Node n;
    blueprint::mca::examples::xyz("interleaved mixed types",
                                  10,
                                  n);
    EXPECT_TRUE(blueprint::mca::is_interleaved(n));    
    EXPECT_FALSE(blueprint::mca::is_contiguous(n));    
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mca_examples, mca_xyz)
{
    Node io_protos;
    io::about(io_protos);

    // we are using one node to hold group of example mcas purely out of 
    // convenience
    Node dsets;
    index_t npts = 100;
    
    blueprint::mca::examples::xyz("interleaved",
                                  npts,
                                  dsets["interleaved"]);

    blueprint::mca::examples::xyz("separate",
                                  npts,
                                  dsets["separate"]);

    blueprint::mca::examples::xyz("contiguous",
                                  npts,
                                  dsets["contiguous"]);
    NodeIterator itr = dsets.children();
    
    while(itr.has_next())
    {
        Node info;
        Node &mca = itr.next();
        std::string name = itr.path();
        // TODO: tests!
    }
}
