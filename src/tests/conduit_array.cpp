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

    for(int i=1;i<20;i+=2)
    {
        data[i] = -i/2;
    }
    std::cout << "Full Data" << std::endl;

    for(int i=0;i<20;i++)
    {
        std::cout << (int64) data[i] << " ";
    }
    std::cout << std::endl;
 
    DataType arr_t(DataType::INT8_T,
                   10,
                   0,
                   sizeof(int8)*2, // stride
                   sizeof(int8),
                   Endianness::DEFAULT_T);
    Node n;
    n["value"].set(arr_t,&data[0]);


    int8_array arr = n["value"].as_int8_array();

    for(int i=0;i<10;i++)
    {
        // note: the cast is for proper printing to std::out
        std::cout << "value[" << i << "] = " << ((int64)arr[i] ) << std::endl;
    }
    std::cout << std::endl;

    EXPECT_EQ(arr[5],5);
    EXPECT_EQ(arr[9],9);

    arr[1] = 100;
    EXPECT_EQ(data[2],100);
    
        std::cout << "Full Data" << std::endl;

    for(int i=0;i<20;i++)
    {
        std::cout << (int64) data[i] << " ";
    }
    std::cout << std::endl;

// TODO: These cases are failing
    Node n2(DataType::Arrays::int8(10,sizeof(int8),sizeof(int8)*2),
            &data[0]);
    int8_array arr_2 = n2.as_int8_array();
    
    for(int i=0;i<10;i++)
    {
        // note: the cast is for proper printing to std::out
        std::cout << "value[" << i << "] = " <<  ((int64)arr_2[i] ) << std::endl;
    }
    std::cout << std::endl;
    
    EXPECT_EQ(arr_2[0],0);
    EXPECT_EQ(arr_2[9],-9);   

}    
    

