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
    std::cout<<"set bananas\n";
    memset(ptr,value,num);
  }

  static void banana_copy(void * destination, const void * source, size_t num)
  {
    std::cout<<"copy bananas\n";
    memcpy(destination,source,num);
  }
};

size_t TestAllocator::m_total_bytes_alloced = 0;
size_t TestAllocator::m_alloc_count = 0;
size_t TestAllocator::m_free_count = 0;

//-----------------------------------------------------------------------------
TEST(conduit_memory_allocator, test_custom_allocator)
{
  int allocator_id
    = conduit::utils::register_mem_handler(TestAllocator::banana_alloc,
                                           TestAllocator::free_bananas,
                                           TestAllocator::banana_copy,
                                           TestAllocator::banana_memset);

  conduit::Node node;
  node.set_allocator(allocator_id);
  node["array"].set(conduit::DataType::float64(10));
  node["int"].set(1);

  conduit::Node other;
  other["path"].set(1);
  node.set(other);
}
