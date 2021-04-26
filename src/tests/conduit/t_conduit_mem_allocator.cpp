// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"


struct TestAllocator
{
    static size_t m_total_bytes_alloced;
    static size_t m_alloc_count;
    static size_t m_free_count;
    static size_t m_memcpy_count;
    static size_t m_memset_count;

    static void * banana_alloc(size_t items, size_t item_size)
    {
        std::cout<<"Bananas allocate\n";
        m_total_bytes_alloced += items * item_size;
        m_alloc_count++;
        return calloc(items, item_size);
    }

    static void free_bananas(void *data_ptr)
    {
        std::cout<<"free bananas\n";
        m_free_count++;
        free(data_ptr);
    }

    static void banana_memset(void * ptr, int value, size_t num )
    {
        m_memset_count++;
        std::cout<<"set bananas\n";
        memset(ptr,value,num);
    }

    static void banana_copy(void * destination, const void * source, size_t num)
    {
        m_memcpy_count++;
        std::cout<<"copy bananas\n";
        memcpy(destination,source,num);
    }

    static void all_about_bananas()
    {
        std::cout << "[total_bytes_alloced: " << m_total_bytes_alloced <<
                     " alloc_count: "  << m_alloc_count <<
                     " free_count: "   << m_free_count <<
                     " memcpy_count: " << m_memcpy_count <<
                     " memset_count: " << m_memset_count << std::endl;
    }
};


size_t TestAllocator::m_total_bytes_alloced = 0;
size_t TestAllocator::m_alloc_count  = 0;
size_t TestAllocator::m_free_count   = 0;
size_t TestAllocator::m_memcpy_count = 0;
size_t TestAllocator::m_memset_count = 0;

//-----------------------------------------------------------------------------
TEST(conduit_memory_allocator, test_custom_allocator)
{
    conduit::Node node_info;
    conduit::Node other_info;

    conduit::utils::set_memcpy_handler(TestAllocator::banana_copy);
    conduit::utils::set_memset_handler(TestAllocator::banana_memset);

    int allocator_id
     = conduit::utils::register_allocator(TestAllocator::banana_alloc,
                                          TestAllocator::free_bananas);

    EXPECT_EQ(TestAllocator::m_memset_count,0);

    conduit::Node node;
    node.set_allocator(allocator_id);
    // this should allocate 10 * 8 bytes
    node["array"].set(conduit::DataType::float64(10));

    TestAllocator::all_about_bananas();
    EXPECT_EQ(TestAllocator::m_total_bytes_alloced,80);
    EXPECT_EQ(TestAllocator::m_alloc_count,1);
    EXPECT_EQ(TestAllocator::m_free_count,0);

    // this should allocate 4 bytes
    node["int"].set_int32(1);

    TestAllocator::all_about_bananas();
    EXPECT_EQ(TestAllocator::m_total_bytes_alloced,84);
    EXPECT_EQ(TestAllocator::m_alloc_count,2);
    EXPECT_EQ(TestAllocator::m_free_count,0);

    // ----------------------------------------------
    conduit::Node other;
    other["path"].set_float32(1.0);

    // this should allocate 4 bytes
    node.update(other);

    EXPECT_EQ(TestAllocator::m_total_bytes_alloced,88);
    EXPECT_EQ(TestAllocator::m_alloc_count,3);
    EXPECT_EQ(TestAllocator::m_free_count,0);

    node.info(node_info);
    other.info(other_info);

    // this should free everything, free count == 3
    node.reset();
    TestAllocator::all_about_bananas();
    EXPECT_EQ(TestAllocator::m_alloc_count,3);
    EXPECT_EQ(TestAllocator::m_free_count,3);

    std::cout << "Main Node Info" << std::endl;
    node_info.print();
    // all pointers should be use alloc_id = 1
    conduit::NodeConstIterator ptr_itr = node_info["mem_spaces"].children();
    while(ptr_itr.has_next())
    {
        const conduit::Node &p_n = ptr_itr.next();
        EXPECT_EQ(p_n["allocator_id"].to_int64(),1);
    }
    
    std::cout << "Other Node Info" << std::endl;
    other_info.print();

    // all pointers should be use alloc_id = 0
    ptr_itr = other_info["mem_spaces"].children();
    while(ptr_itr.has_next())
    {
        const conduit::Node &p_n = ptr_itr.next();
        EXPECT_EQ(p_n["allocator_id"].to_int64(),0);
    }
}
