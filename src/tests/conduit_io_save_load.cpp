/*****************************************************************************
* Copyright (c) 2014, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory. 
* 
* All rights reserved.
* 
* This source code cannot be distributed without further review from 
* Lawrence Livermore National Laboratory.
*****************************************************************************/

///
/// file: conduit_io_binary.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;
using namespace std;


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


TEST(conduit_io_save_load, conduit_bin_simple_2_file)
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

    Node nsrc(schema,data);
    nsrc.save("test_conduit_io_bin_simple_2_file");

    Node n;
    n.load("test_conduit_io_bin_simple_2_file");
        
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
}




TEST(conduit_io_save_load, conduit_mmap_simple_2_file)
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

    Node nsrc(schema,data);
    
    nsrc.save("test_conduit_mmap_x2");
    
   
    Node nmmap;
    nmmap.mmap("test_conduit_mmap_x2");
    
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
    
    Node ntest(schema,"test_conduit_mmap_x2.conduit_bin");
    EXPECT_EQ(ntest[0]["a"].as_int32(), 100);
    EXPECT_EQ(ntest[0]["b"].as_int32(), 200);
}


// TODO: Resolve the memory access pattern in this case
// to run, remove "DISABLED_" prefix
TEST(conduit_io_save_load, DISABLED_conduit_simple_restore)
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
    d.n.save("test_conduit_restore_mmap");

    ExampleData d2;
    d2.alloc(10);
    Node nmmap;
    nmmap.mmap("test_conduit_restore_mmap");

    d2.n.info().print();

    d2.n.update(nmmap);
    d2.n.print();
    // this will fail
    EXPECT_EQ(d2.n["x"].as_float64_array()[1],d2.x_vals[1]);
    EXPECT_EQ(d.x_vals[1],d2.x_vals[1]);

    // note that the mem setup isn't quite right, 
    // we must have fell in to a realloc situation
    d2.n.info().print();

}
