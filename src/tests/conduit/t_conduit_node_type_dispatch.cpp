// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_node_type_dispatch.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;

index_t ARRAY_SIZE = 1000000;
index_t NUM_ITERS  = 1;


/// simple timer class
//-----------------------------------------------------------------------------
class Timer 
{

  typedef std::chrono::high_resolution_clock high_resolution_clock;
  typedef std::chrono::nanoseconds nanoseconds;
  typedef std::chrono::duration<float> fsec;

public:
    explicit Timer()
    {
      reset();
    }

    void reset()
    {
      m_start = high_resolution_clock::now();
    }

    float elapsed() const
    {
       auto ftime  = std::chrono::duration_cast<fsec>(high_resolution_clock::now() - m_start);
       return ftime.count();
    }

private:
    high_resolution_clock::time_point m_start;
};


//-----------------------------------------------------------------------------
// -- type specific template imp 
template<typename T>
float64
template_dispatch_detail(Node &n)
{
    float64 res = 0;
    index_t nele = n.dtype().number_of_elements();
    T *t_ptr = n.value();
    for(index_t i=0; i < nele;i++)
    {
        res += t_ptr[i];
    }

    return res;
}

//-----------------------------------------------------------------------------
// -- use template dispatch
float64 
template_dispatch(Node &n)
{
    if(n.dtype().is_float64())
    {
        return template_dispatch_detail<float64>(n);
    }
    else if(n.dtype().is_float32())
    {
        return template_dispatch_detail<float32>(n);
    }

    return 0.0;
}

//-----------------------------------------------------------------------------
// -- convert each element to desired type in the loop
float64 
on_the_fly_convert(Node &n)
{
    float64 res = 0;
    index_t nele = n.dtype().number_of_elements();

    Node temp;
    for(index_t i=0; i < nele;i++)
    {
        temp.set_external(n.dtype(),n.element_ptr(i));
        res += temp.to_float64();
    }

    return res;
}

//-----------------------------------------------------------------------------
// -- convert the array to desired type, then loop
float64 
array_convert(Node &n)
{
    float64 res = 0;
    index_t nele = n.dtype().number_of_elements();

    Node temp;
    n.to_float64_array(temp);
    float64 *f64_ptr = temp.value();

    for(index_t i=0; i < nele;i++)
    {
        res += f64_ptr[i];
    }

    return res;
}


//-----------------------------------------------------------------------------
TEST(conduit_node, test_dispatch)
{
    Node n;
    
    n.set(DataType::float32(ARRAY_SIZE));
    float32 *f32_ptr = n.value();
    for(index_t i=0;i<ARRAY_SIZE;i++)
    {
        f32_ptr[i] = 1.0;
    }

    std::cout << "Number of elements: " <<  ARRAY_SIZE << std::endl;
    std::cout << "Iterations: " << NUM_ITERS << std::endl;

    float64 res1,res2,res3;

    Timer t1;
    for(index_t i=0;i<NUM_ITERS;i++)
    {
        res1 = on_the_fly_convert(n);
    }
    t1.elapsed();

    Timer t2;
    for(index_t i=0;i<NUM_ITERS;i++)
    {
        res2 = array_convert(n);
    }
    t2.elapsed();

    Timer t3;
    for(index_t i=0;i<NUM_ITERS;i++)
    {
        res3 = template_dispatch(n);
    }
    t3.elapsed();

    EXPECT_EQ(res1,float64(ARRAY_SIZE));
    EXPECT_EQ(res2,float64(ARRAY_SIZE));
    EXPECT_EQ(res3,float64(ARRAY_SIZE));

    // std::cout << "on_the_fly_convert (last res): " << res1 << std::endl;
    // std::cout << "array_convert (last res):      " << res2 << std::endl;
    // std::cout << "template_dispatch (last res):  " << res3 << std::endl;

    std::cout << "on_the_fly_convert: " << t1.elapsed() << std::endl;
    std::cout << "array_convert:      " << t2.elapsed() << std::endl;
    std::cout << "template_dispatch:  " << t3.elapsed() << std::endl;


}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    // allow override of the data size via the command line
    if(argc > 1)
    {
        ARRAY_SIZE = atoi(argv[1]);
    }

    if(argc > 2)
    {
        NUM_ITERS = atoi(argv[2]);
    }


    result = RUN_ALL_TESTS();
    return result;
}




