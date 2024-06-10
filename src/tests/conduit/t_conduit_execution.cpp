// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_execution.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_execution.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

// //-----------------------------------------------------------------------------
// TEST(conduit_execution, test_forall)
// {
//     const index_t size = 10;

//     index_t host_vals[size];
//     index_t *dev_vals_ptr = static_cast<index_t*>(alloc(sizeof(index_t) * size));

//     conduit::execution::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
//     {
//         dev_vals_ptr[i] = i;
//     });
    
//     MagicMemory::copy(&host_vals[0], dev_vals_ptr, sizeof(index_t) * size);

//     for(index_t i=0;i<size;i++)
//     {
//       EXPECT_EQ(host_vals[i],i);
//     }

//     free(dev_vals_ptr);

// }

//-----------------------------------------------------------------------------
// TEST(conduit_execution, cpp_magic_tests)
TEST(conduit_execution, justin_fun)
{
    std::cout << "forall cases!" << std::endl;

    conduit::execution::ExecPolicy SerialPolicy(conduit::execution::policies::Serial);
    // conduit::execution::ExecPolicy DevicePolicy(conduit::execution::policies::Device);

    int size = 4;

    conduit::execution::new_forall(SerialPolicy, 0, size, [=] (int i)
    {
       std::cout << i << std::endl;
    });

    // conduit::execution::new_forall(DevicePolicy, 0, size, [=] (int i)
    // {
    //    std::cout << i << std::endl;
    // });

    // std::cout << "functor cases!" << std::endl;

    // MyFunctor func;
    // func.size = size;
    // conduit::execution::dispatch(SerialPolicy,func);
    // std::cout << func.res << std::endl;

    // conduit::execution::dispatch(DevicePolicy,func);
    // std::cout << func.res << std::endl;

    // MySpecialFunctor sfunc;
    // sfunc.size = 4;
    // conduit::execution::dispatch(SerialPolicy,sfunc);
    // std::cout << func.res << std::endl;
    
    // std::cout << "C++ 20" << std::endl;

    // int res =0;
    // /// c++ 20 allows us to double lambda instead of a functor
    // conduit::execution::dispatch(SerialPolicy, [&] <typename ComboPolicyTag>(ComboPolicyTag &exec)
    // {
    //      using thetag = typename ComboPolicyTag::tag_2;
    //      MySpecialClass<thetag> s(10);
    //      conduit::execution::new_forall<thetag>(0, size, [=] (int i)
    //      {
    //          s.exec(i);
    //      });
    //      res = 10;
    // });

}


