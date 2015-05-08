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
/// file: conduit_node_save_load.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"


using namespace conduit;
using namespace std;

//-----------------------------------------------------------------------------
class ExampleData
{
public:
   ExampleData()
    {}
    
    void alloc(index_t c_size)
    {
        x_vals.resize(c_size,0.0);        
        y_vals.resize(c_size,0.0);
        z_vals.resize(c_size,0.0);
        n["x"].set_external(x_vals);
        n["y"].set_external(y_vals);
        n["z"].set_external(z_vals);
    }

   Node n;
   std::vector<float64> x_vals;
   std::vector<float64> y_vals;
   std::vector<float64> z_vals;
};

//-----------------------------------------------------------------------------
TEST(conduit_node_save_load, bin_simple_file)
{

    int32   a1_val  = 10;
    int32   b1_val  = 20;
    
    int32   a2_val  = -10;
    int32   b2_val  = -20;
    
    char *data = new char[16];
    memcpy(&data[0],&a1_val,4);
    memcpy(&data[4],&b1_val,4);
    memcpy(&data[8],&a2_val,4);
    memcpy(&data[12],&b2_val,4);
    
    Schema schema("{\"dtype\":{\"a\":\"int32\",\"b\":\"int32\"},\"length\":2}");

    Node nsrc(schema,data,true);
    nsrc.save("tout_conduit_io_bin_simple_2_file");

    Node n;
    n.load("tout_conduit_io_bin_simple_2_file");
        
    n.schema().print();
    n.print_detailed();
    
    std::cout <<  n[0]["a"].as_int32() << std::endl;
    std::cout <<  n[1]["a"].as_int32() << std::endl;

    std::cout <<  n[0]["b"].as_int32() << std::endl;
    std::cout <<  n[1]["b"].as_int32() << std::endl;

    EXPECT_EQ(n[0]["a"].as_int32(), a1_val);
    EXPECT_EQ(n[1]["a"].as_int32(), a2_val);

    EXPECT_EQ(n[0]["b"].as_int32(), b1_val);
    EXPECT_EQ(n[1]["b"].as_int32(), b2_val);

    delete [] data;
}


//-----------------------------------------------------------------------------
TEST(conduit_node_save_load, mmap_simple_file)
{
    int32   a1_val  = 10;
    int32   b1_val  = 20;
    
    int32   a2_val  = -10;
    int32   b2_val  = -20;
    
    char *data = new char[16];
    memcpy(&data[0],&a1_val,4);
    memcpy(&data[4],&b1_val,4);
    memcpy(&data[8],&a2_val,4);
    memcpy(&data[12],&b2_val,4);
    
    Schema schema("{\"dtype\":{\"a\":\"int32\",\"b\":\"int32\"},\"length\":2}");

    Node nsrc(schema,data,true);
    
    nsrc.save("tout_conduit_mmap_x2");
    
   
    Node nmmap;
    nmmap.mmap("tout_conduit_mmap_x2");
    
    nmmap.schema().print();
    nmmap.print_detailed();
    
    std::cout <<  nmmap[0]["a"].as_int32() << std::endl;
    std::cout <<  nmmap[1]["a"].as_int32() << std::endl;

    std::cout <<  nmmap[0]["b"].as_int32() << std::endl;
    std::cout <<  nmmap[1]["b"].as_int32() << std::endl;

    EXPECT_EQ(nmmap[0]["a"].as_int32(), a1_val);
    EXPECT_EQ(nmmap[1]["a"].as_int32(), a2_val);

    EXPECT_EQ(nmmap[0]["b"].as_int32(), b1_val);
    EXPECT_EQ(nmmap[1]["b"].as_int32(), b2_val);

    cout << "mmap write" <<endl;
    // change mmap
    nmmap[0]["a"] = 100;
    nmmap[0]["b"] = 200;
    
   
    // standard read
    
    Node ntest;
    ntest.load(schema,"tout_conduit_mmap_x2.conduit_bin");
    EXPECT_EQ(ntest[0]["a"].as_int32(), 100);
    EXPECT_EQ(ntest[0]["b"].as_int32(), 200);

    delete [] data;
}


//-----------------------------------------------------------------------------
TEST(conduit_node_save_load, simple_restore)
{
    Node n_src;
    Node n_dest;
    
    
    std::vector<float64> v_src;
    std::vector<float64> v_dest;
    v_src.resize(10,0.0);
    v_dest.resize(10,0.0);
    
    n_src["v"].set_external(v_src);
    n_dest["v"].set_external(v_dest);
    
    for(index_t i=0;i<10;i++)
    {
        v_src[i] = 1.2 * (i+1);
    }

    n_src.print();
    n_src.info().print();
    n_src.save("tout_conduit_simple_restore");

    Node n_load;
    n_load.load("tout_conduit_simple_restore");

    n_load.print();
    

    n_dest.update(n_load);

    EXPECT_EQ(n_dest["v"].as_float64_array()[0],v_src[0]);
    EXPECT_EQ(v_dest[0],v_src[0]);

    n_dest.info().print();

}

//-----------------------------------------------------------------------------
TEST(conduit_node_save_load, simple_class_restore)
{
    ExampleData d;
    d.alloc(10);
    d.n.print();

    for(index_t i=0;i<10;i++)
    {
        d.x_vals[i] = 1.2 * i;
        d.y_vals[i] = 2.3 * i;
        d.z_vals[i] = 3.4 * i;
    }

    d.n.print();
    d.n.save("tout_conduit_restore_mmap");

    ExampleData d2;
    d2.alloc(10);
    Node nmmap;
    nmmap.mmap("tout_conduit_restore_mmap");

    d2.n.info().print();

    d2.n.update(nmmap);
    d2.n.print();

    EXPECT_EQ(d2.n["x"].as_float64_array()[1],d2.x_vals[1]);
    EXPECT_EQ(d.x_vals[1],d2.x_vals[1]);

    d2.n.info().print();

}

//-----------------------------------------------------------------------------
TEST(conduit_node_save_load, io_explicit_zero_length_vector_restore)
{
    std::vector<float> one;
    float two = 2;
    float three = 3;
    float four = 4;

    Node n1;
    n1["one"].set_external(one);
    n1["two"].set_external(&two);
    n1["three"].set_external(&three);
    n1["four"].set_external(&four);

    std::cout << "n1 before saving" << std::endl;

    n1.print_detailed();
   
    n1.save("tout_zero_len_vector_save");

    Node n2;
    n2.load("tout_zero_len_vector_save");

    std::cout << "n2 load result" << std::endl;

    n2.print_detailed();
            
    EXPECT_EQ(n1.schema()["one"].dtype().number_of_elements(),
              n2.schema()["one"].dtype().number_of_elements());
    EXPECT_EQ(n2.schema()["one"].dtype().number_of_elements(),0);
    
    n1.update(n2);

    std::cout << "n1 after updating from n2" << std::endl;

    n1.print_detailed();
    EXPECT_EQ(n1.schema()["one"].dtype().number_of_elements(),0);

}

