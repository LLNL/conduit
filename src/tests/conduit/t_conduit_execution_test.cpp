// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_execution_test.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_execution, test_forall)
{
    const index_t size = 10;

    index_t host_vals[size];
    index_t *dev_vals_ptr = static_cast<index_t*>(alloc(sizeof(index_t) * size));

    conduit::execution::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        dev_vals_ptr[i] = i;
    });
    
    MagicMemory::copy(&host_vals[0], dev_vals_ptr, sizeof(index_t) * size);

    for(index_t i=0;i<size;i++)
    {
      EXPECT_EQ(host_vals[i],i);
    }

    free(dev_vals_ptr);

}

//-----------------------------------------------------------------------------
TEST(conduit_execution, cpp_magic_tests)
{
    std::cout << "forall cases!" << std::endl;

    ExecPolicy ea(0);
    ExecPolicy eb(1);

    int size = 4;

    forall(ea, 0, size, [=] (int i)
    {
       std::cout << i << std::endl;
    });

    forall(eb, 0, size, [=] (int i)
    {
       std::cout << i << std::endl;
    });

    std::cout << "functor cases!" << std::endl;

    MyFunctor func;
    func.size = size;
    dispatch(ea,func);
    std::cout << func.res << std::endl;

    dispatch(eb,func);
    std::cout << func.res << std::endl;

    MySpecialFunctor sfunc;
    sfunc.size = 4;
    dispatch(ea,sfunc);
    std::cout << func.res << std::endl;
    
    std::cout << "C++ 20" << std::endl;

    int res =0;
    /// c++ 20 allows us to double lambda instead of a functor
    dispatch(ea, [&] <typename ComboPolicyTag>(ComboPolicyTag &exec)
    {
         using thetag = typename ComboPolicyTag::tag_2;
         MySpecialClass<thetag> s(10);
         forall<thetag>(0, size, [=] (int i)
         {
             s.exec(i);
         });
         res = 10;
    });

}


