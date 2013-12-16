///
/// file: conduit_array.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
// #include "rapidjson/document.h"
using namespace conduit;

TEST(conduit_array_stride_int8, conduit_array)
{
    std::vector<int8> data(20,0);

    for(int i=0;i<20;i+=2)
    {
        data[i] = i/2;
    }


    DataType arr_t(DataType::INT8_T,
                   10,
                   0,
                   sizeof(int8)*2, // stride
                   sizeof(int8),
                   Endianness::DEFAULT_T);
    Node n;
    n["value"].set(&data[0],arr_t);

    int8_array arr = n["value"].as_int8_array();

    for(int i=0;i<10;i++)
    {
        // note: the cast is for proper printing to std::out
        std::cout << ((index_t)arr[i] )<< " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(arr[5],5);
    EXPECT_EQ(arr[9],9);   
    arr[1] = 100;
    EXPECT_EQ(data[2],100);
}    
    

