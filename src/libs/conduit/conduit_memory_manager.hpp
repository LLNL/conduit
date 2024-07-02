// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_memory_manager.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_MEMORY_MANAGER_HPP
#define CONDUIT_MEMORY_MANAGER_HPP

#include <cstddef>

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit_config.h"
#include <conduit.hpp>

//-----------------------------------------------------------------------------
// -- begin conduit --
//-----------------------------------------------------------------------------
namespace conduit
{

///
/// Interfaces for host and device memory allocation / deallocation.
///


//-----------------------------------------------------------------------------
/// Interface to set allocator ids (singleton)
//-----------------------------------------------------------------------------
class AllocationManager
{
public:
    /// Return host allocator id
    ///  If Umpire is enabled and no allocator has been set,
    ///  an Umpire "HOST_POOL" allocator is created, set, and returned.
    static int  host_allocator_id();

    /// Return device allocator id
    ///  If Umpire is enabled and no allocator has been set,
    ///  an Umpire "GPU_POOL" allocator is created, set, and returned.
    /// If Umpire is disabled, an error is thrown
    static int  device_allocator_id();

    /// set umpire host allocator from outside ascent via id
    /// Throws an error if Umpire is disabled
    static bool set_host_allocator_id(int id);

    /// set umpire device allocator from outside ascent via id
    /// Throws an error if Umpire is disabled
    static bool set_device_allocator_id(int id);

    // registered conduit magic memory allocator id for host memory
    static int  conduit_host_allocator_id();
    // registered conduit magic memory allocator id for device memory
    static int  conduit_device_allocator_id();

    // registers the fancy conduit memory handlers for
    // magic memset and memcpy
    static void set_conduit_mem_handlers();

private:
    static int  m_host_allocator_id;
    static int  m_device_allocator_id;
    
    static int  m_conduit_host_allocator_id;
    static int  m_conduit_device_allocator_id;

    static bool m_external_host_allocator;
    static bool m_external_device_allocator;

};

//-----------------------------------------------------------------------------
/// Host Memory allocation / deallocation interface (singleton)
///  Uses AllocationManager::host_allocator_id() when Umpire is enabled,
///  Uses malloc/free when Umpire is disabled. 
//-----------------------------------------------------------------------------
struct HostMemory
{
    static void *allocate(size_t bytes);
    static void *allocate(size_t items, size_t item_size);
    static void  deallocate(void *data_ptr);

private:
    static size_t m_total_bytes_alloced;
    static size_t m_alloc_count;
    static size_t m_free_count;

};
//-----------------------------------------------------------------------------
/// Device Memory allocation / deallocation interface (singleton)
///  Uses AllocationManager::device_allocator_id() when Umpire is enabled.
///  allocate() and deallocate() throw errors when Umpire is disabled.
//-----------------------------------------------------------------------------
struct DeviceMemory
{
    static void *allocate(size_t bytes);
    static void *allocate(size_t items, size_t item_size);
    static void  deallocate(void *data_ptr);
    static bool is_device_ptr(const void *ptr);
    static void is_device_ptr(const void *ptr, bool &is_gpu, bool &is_unified);

private:
    static size_t m_total_bytes_alloced;
    static size_t m_alloc_count;
    static size_t m_free_count;

};

//-----------------------------------------------------------------------------
struct MagicMemory
{
    static void set(void *ptr, int value, size_t num);
    static void copy(void *destination, const void *source, size_t num);
};

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
