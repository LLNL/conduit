///
/// file: conduit_json.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;


TEST(conduit_to_json_1, conduit_json)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);

    std::cout << n.to_json();
}

TEST(conduit_to_json_2, conduit_json)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;
    uint32   arr[5];
    for(index_t i=0;i<5;i++)
    {
        arr[i] = i*i;
    }

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["arr"].set(DataType::Arrays::uint32(5),arr);


    std::string pure_json = n.to_json(true);
    std::cout << n.to_json();
    std::cout << pure_json << std::endl;
    
    Generator g(pure_json,"json");
    Node n2(g);
    
    EXPECT_EQ(n["a"].as_uint32(),n2["a"].as_uint32());
    EXPECT_EQ(n["b"].as_uint32(),n2["b"].as_uint32());
    
}

TEST(conduit_to_json_3, conduit_json)
{
    std::string pure_json ="{a:[0,1,2,3,4],b:[0.0,1.1,2.2,3.3]}";
    Generator g(pure_json,"json");
    Node n(g);
    std::cout << n.to_json();
}


TEST(conduit_to_json_inline_value, conduit_json)
{
    uint32   val=0;

    std::string schema ="{dtype: uint32, value:42}";
    // TODO: check for "unit32" , bad spelling!
    Generator g(schema);
    Node n(g);
    std::cout << n.as_uint32() << std::endl;
    EXPECT_EQ(42,n.as_uint32());
    
    Generator g2(schema,&val);
    Node n2(g2);
    std::cout << n2.as_uint32() << std::endl;
    EXPECT_EQ(42,val);
    
}

TEST(conduit_to_json_inline_array, conduit_json)
{
    uint32   arr[5];

    std::string schema ="{dtype:uint32, length:5, value:[0,1,2,3,4]}";
    Node n(schema,arr);
    std::cout << n.to_json() << std::endl;
    
    uint32 *ptr = &arr[0];

    for(int i=0;i<5;i++)
    {
        //std::cout << arr[i] << " vs " << ptr[i] << std::endl;
        EXPECT_EQ(arr[i],ptr[i]);
    }
    
    std::string schema2 ="{dtype:uint32, value:[10,20,30]}";
    Node n2(schema2,arr);
    ptr =n2.as_uint32_ptr();
    std::cout << n2.to_json() << std::endl;
    
    EXPECT_EQ(n2.dtype().number_of_elements(),3);
    for(int i=0;i<n2.dtype().number_of_elements();i++)
    {
        EXPECT_EQ(ptr[i],10*(i+1));
    }
    
    std::string schema3 ="{dtype:uint32, value:[100,200,300,400,500]}";
    Node n3(schema3,arr);
    ptr =n3.as_uint32_ptr();
    std::cout << n3.to_json() << std::endl;
    
    EXPECT_EQ(n3.dtype().number_of_elements(),5);
    for(int i=0;i<n3.dtype().number_of_elements();i++)
    {
        EXPECT_EQ(ptr[i],100*(i+1));
    }
    
    std::string schema4 ="{dtype:uint32, value:[1000,2000]}";
    Node n4(schema4,arr);
    ptr =n4.as_uint32_ptr();
    std::cout << n4.to_json() << std::endl;
    
    EXPECT_EQ(n4.dtype().number_of_elements(),2);
    for(int i=0;i<n4.dtype().number_of_elements();i++)
    {
        EXPECT_EQ(ptr[i],1000*(i+1));
    }

    // checking to make sure we are using the same memory space
    for(int i=2;i<5;i++)
    {
        EXPECT_EQ(ptr[i],100*(i+1));
    }
    
    
}

