// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_to_string.cpp
///
//-----------------------------------------------------------------------------


#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_to_string, simple_1)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    char *data = new char[16];
    memcpy(&data[0],&a_val,4);
    memcpy(&data[4],&b_val,4);
    memcpy(&data[8],&c_val,8);

    Schema schema("{\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}");
    Node n(schema,data,true);
    n.schema().print();
    n.print_detailed();
    EXPECT_EQ(std::string("{\"a\": 10,\"b\": 20,\"c\": 30.0}"),n.to_json("json",0,0,"",""));
    


    Schema schema2("{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}");
    Node n2(schema2,data,true);
    n2.schema().print();
    n.print_detailed();
    EXPECT_EQ(std::string("{\"g\": {\"a\": 10,\"b\": 20,\"c\": 30.0}}"),n2.to_json("json",0,0,"",""));

    delete [] data;
}



//-----------------------------------------------------------------------------
TEST(conduit_to_string, conduit_to_summary_string_obj)
{
    Node n;
    n["a"] = {0,1,2,3,4,5,6,7,8,9};
    n["b"] = {1,1,2,3,4,5,6,7,8,9};
    n["c"] = {2,1,2,3,4,5,6,7,8,9};
    n["d"] = {3,1,2,3,4,5,6,7,8,9};
    n["e"] = {4,1,2,3,4,5,6,7,8,9};
    n["f"] = {5,1,2,3,4,5,6,7,8,9};
    n["g/aa"] = {0,1,2,3,4,5,6,7,8,9};
    n["g/bb"] = {1,1,2,3,4,5,6,7,8,9};
    n["g/cc"] = {2,1,2,3,4,5,6,7,8,9};
    n["g/dd"] = {3,1,2,3,4,5,6,7,8,9};

    std::cout << "yaml rep"  << std::endl;
    std::cout << n.to_yaml() << std::endl;

    Node opts;
    std::string tres = "";
    std::string texpect = "";

    std::cout << "default to_summary_string"  << std::endl;
    tres = n.to_summary_string();
    std::cout << tres << std::endl;
// for g: yes there is an extra space here ...
    texpect = R"ST(
a: [0, 1, 2, ..., 8, 9]
b: [1, 1, 2, ..., 8, 9]
c: [2, 1, 2, ..., 8, 9]
d: [3, 1, 2, ..., 8, 9]
e: [4, 1, 2, ..., 8, 9]
f: [5, 1, 2, ..., 8, 9]
g: 
  aa: [0, 1, 2, ..., 8, 9]
  bb: [1, 1, 2, ..., 8, 9]
  cc: [2, 1, 2, ..., 8, 9]
  dd: [3, 1, 2, ..., 8, 9]
)ST";
    EXPECT_EQ(tres,texpect);

    opts.reset();
    opts["num_children_threshold"] = 5;
    opts["num_elements_threshold"] = 4;
    opts.print();
    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;

// for g: yes there is an extra space here ...
    texpect = R"ST(
a: [0, 1, ..., 8, 9]
b: [1, 1, ..., 8, 9]
c: [2, 1, ..., 8, 9]
... ( skipped 2 children )
f: [5, 1, ..., 8, 9]
g: 
  aa: [0, 1, ..., 8, 9]
  bb: [1, 1, ..., 8, 9]
  cc: [2, 1, ..., 8, 9]
  dd: [3, 1, ..., 8, 9]
)ST";
    EXPECT_EQ(tres,texpect);

    opts.reset();
    opts["num_children_threshold"] = 3;
    opts["num_elements_threshold"] = 4;
    opts.print();
    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;

    texpect = R"ST(
a: [0, 1, ..., 8, 9]
b: [1, 1, ..., 8, 9]
... ( skipped 4 children )
g: 
  aa: [0, 1, ..., 8, 9]
  bb: [1, 1, ..., 8, 9]
  ... ( skipped 1 child )
  dd: [3, 1, ..., 8, 9]
)ST";
    EXPECT_EQ(tres,texpect);

    // neg or zero for thresholds should trigger default yaml case
    opts.reset();
    opts["num_elements_threshold"] = -1;
    opts["num_children_threshold"] = -1;
    opts.print();
    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;
    EXPECT_EQ(tres,n.to_yaml());

    // high thresholds should trigger default yaml case
    opts.reset();
    opts["num_elements_threshold"] = 100;
    opts["num_children_threshold"] = 100;
    opts.print();
    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;
    EXPECT_EQ(tres,n.to_yaml());

}

//-----------------------------------------------------------------------------
TEST(conduit_to_string, conduit_to_summary_string_list)
{
    Node n, opts;
    std::string tres, texpect;
    // list cases:
    // 7 children at root
    n.append().set({0,1,2,3,4,5,6,7,8,9});
    n.append().set({1,1,2,3,4,5,6,7,8,9});
    n.append().set({2,1,2,3,4,5,6,7,8,9});
    n.append().set({3,1,2,3,4,5,6,7,8,9});
    n.append().set({4,1,2,3,4,5,6,7,8,9});
    n.append().set({5,1,2,3,4,5,6,7,8,9});
    // 4 children in sub
    Node &n_sub = n.append();
    n_sub.append().set({0,1,2,3,4,5,6,7,8,9});
    n_sub.append().set({1,1,2,3,4,5,6,7,8,9});
    n_sub.append().set({2,1,2,3,4,5,6,7,8,9});
    n_sub.append().set({3,1,2,3,4,5,6,7,8,9});

    std::cout << "yaml rep"  << std::endl;
    std::cout << n.to_yaml() << std::endl;

    std::cout << "default to_summary_string"  << std::endl;
    tres = n.to_summary_string();
    std::cout << tres << std::endl;

    texpect = R"ST(
- [0, 1, 2, ..., 8, 9]
- [1, 1, 2, ..., 8, 9]
- [2, 1, 2, ..., 8, 9]
- [3, 1, 2, ..., 8, 9]
- [4, 1, 2, ..., 8, 9]
- [5, 1, 2, ..., 8, 9]
- 
  - [0, 1, 2, ..., 8, 9]
  - [1, 1, 2, ..., 8, 9]
  - [2, 1, 2, ..., 8, 9]
  - [3, 1, 2, ..., 8, 9]
)ST";
    EXPECT_EQ(tres,texpect);

    opts.reset();
    opts["num_children_threshold"] = 5;
    opts["num_elements_threshold"] = 4;
    opts.print();
    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;

    texpect = R"ST(
- [0, 1, ..., 8, 9]
- [1, 1, ..., 8, 9]
- [2, 1, ..., 8, 9]
... ( skipped 2 children )
- [5, 1, ..., 8, 9]
- 
  - [0, 1, ..., 8, 9]
  - [1, 1, ..., 8, 9]
  - [2, 1, ..., 8, 9]
  - [3, 1, ..., 8, 9]
)ST";
    EXPECT_EQ(tres,texpect);

    opts.reset();
    opts["num_children_threshold"] = 3;
    opts["num_elements_threshold"] = 4;
    opts.print();
    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;

    texpect = R"ST(
- [0, 1, ..., 8, 9]
- [1, 1, ..., 8, 9]
... ( skipped 4 children )
- 
  - [0, 1, ..., 8, 9]
  - [1, 1, ..., 8, 9]
  ... ( skipped 1 child )
  - [3, 1, ..., 8, 9]
)ST";
    EXPECT_EQ(tres,texpect);

    // neg or zero for thresholds should trigger default yaml case
    opts.reset();
    opts["num_elements_threshold"] = -1;
    opts["num_children_threshold"] = -1;
    opts.print();
    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;
    EXPECT_EQ(tres,n.to_yaml());

    // high thresholds should trigger default yaml case
    opts.reset();
    opts["num_elements_threshold"] = 100;
    opts["num_children_threshold"] = 100;
    opts.print();
    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;
    EXPECT_EQ(tres,n.to_yaml());
}

//-----------------------------------------------------------------------------
TEST(conduit_to_string, conduit_to_summary_string_misc)
{
    Node n;
    Node opts;
    std::string tres, texpect;
    // misc cases
    // empty
    opts.reset();
    n.reset();
    opts.print();
    n.print();
    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;
    EXPECT_EQ(tres,n.to_yaml());

    // scalar
    opts.reset();
    n.reset();
    n = 10;
    opts.print();
    n.print();
    tres = n.to_summary_string();
    std::cout << tres << std::endl;
    EXPECT_EQ(tres,n.to_yaml());

    // small array
    opts.reset();
    n.reset();
    n = {1,2,3};
    opts.print();
    n.print();
    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;
    EXPECT_EQ(tres,n.to_yaml());
    
    opts.reset();
    n.reset();
    //
    opts["num_elements_threshold"] = 2;
    n = {1,2,3};
    opts.print();
    n.print();
    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;
    EXPECT_EQ(tres,"[1, ..., 3]");

    // small num of children
    opts.reset();
    n.reset();
    opts["num_children_threshold"] = 2;
    n["a"] = 10;
    n["b"] = 20;
    n["c"] = 20;

    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;

    texpect = R"ST(
a: 10
... ( skipped 1 child )
c: 20
)ST";
    EXPECT_EQ(tres,texpect);
    
    
    // list and obj
    opts.reset();
    n.reset();
    opts["num_children_threshold"] = 2;
    n["a"] = 10;
    n["b"] = 20;
    n["c"].append().set(10);
    n["c"].append().set(20);
    n["c"].append().set(30);
    n["c"].append().set(40);

    tres = n.to_summary_string(opts);
    std::cout << tres << std::endl;

    texpect = R"ST(
a: 10
... ( skipped 1 child )
c: 
  - 10
  ... ( skipped 2 children )
  - 40
)ST";
    EXPECT_EQ(tres,texpect);

}