// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_memory_manager.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit_config.h"
#include "conduit_memory_manager.hpp"
#include "conduit.hpp"

#include <cstring> // memcpy

//-----------------------------------------------------------------------------
// -- begin conduit --
//-----------------------------------------------------------------------------
namespace conduit
{

///
/// Interfaces for host and device memory allocation / deallocation.
///


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Host Memory
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
size_t HostMemory::m_total_bytes_alloced = 0;
size_t HostMemory::m_alloc_count = 0;
size_t HostMemory::m_free_count = 0;

//-----------------------------------------------------------------------------
void *
HostMemory::allocate(size_t bytes)
{
    m_total_bytes_alloced += bytes;
    m_alloc_count ++;
// #if defined(ASCENT_UMPIRE_ENABLED)
//     auto &rm = umpire::ResourceManager::getInstance ();
//     const int allocator_id = AllocationManager::host_allocator_id();
//     umpire::Allocator host_allocator = rm.getAllocator (allocator_id);
//     return host_allocator.allocate(bytes);
// #else
    return malloc(bytes);
// #endif
}

//-----------------------------------------------------------------------------
void
HostMemory::deallocate(void *data_ptr)
{
    m_free_count ++;
// #if defined(ASCENT_UMPIRE_ENABLED)
//     auto &rm = umpire::ResourceManager::getInstance ();
//     const int allocator_id = AllocationManager::host_allocator_id();
//     umpire::Allocator host_allocator = rm.getAllocator (allocator_id);
//     host_allocator.deallocate(data_ptr);
// #else
    return free(data_ptr);
// #endif
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Device Memory
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
size_t DeviceMemory::m_total_bytes_alloced = 0;
size_t DeviceMemory::m_alloc_count = 0;
size_t DeviceMemory::m_free_count = 0;

//-----------------------------------------------------------------------------
void *
DeviceMemory::allocate(size_t bytes)
{
#if !defined(CONDUIT_UMPIRE_ENABLED) // TODO is this going to exist
    CONDUIT_ERROR("Conduit was built without Umpire support. "
                   "Cannot use DeviceMemory::alloc().");
#endif

#if defined(CONDUIT_USE_RAJA) && defined(CONDUIT_UMPIRE_ENABLED)
    m_total_bytes_alloced += bytes;
    m_alloc_count ++;
    auto &rm = umpire::ResourceManager::getInstance ();
    const int allocator_id = AllocationManager::device_allocator_id();
    umpire::Allocator device_allocator = rm.getAllocator (allocator_id);
    return device_allocator.allocate(bytes);
#else
    (void) bytes; // unused
    CONDUIT_ERROR("Calling device allocator when no device is present.");
    return nullptr;
#endif
}

//-----------------------------------------------------------------------------
void
DeviceMemory::deallocate(void *data_ptr)
{
#if !defined(CONDUIT_UMPIRE_ENABLED) // TODO is this going to exist
    CONDUIT_ERROR("Conduit was built without Umpire support. "
                  "Cannot use DeviceMemory::free().");
#endif

#if defined(CONDUIT_USE_RAJA) && defined(CONDUIT_UMPIRE_ENABLED)
    m_free_count++;
    auto &rm = umpire::ResourceManager::getInstance ();
    const int allocator_id = AllocationManager::device_allocator_id();
    umpire::Allocator device_allocator = rm.getAllocator (allocator_id);
    device_allocator.deallocate (data_ptr);
#else
    (void) data_ptr;
    CONDUIT_ERROR("Calling device allocator when no device is present.");
#endif
}

//-----------------------------------------------------------------------------
// Adapted from:
// https://gitlab.kitware.com/third-party/nvpipe/blob/master/encode.c
bool
DeviceMemory::is_device_ptr(const void *ptr)
{
#if defined(CONDUIT_USE_CUDA)
    cudaPointerAttributes atts;
    const cudaError_t perr = cudaPointerGetAttributes(&atts, ptr);
    // clear last error so other error checking does
    // not pick it up
    cudaError_t error = cudaGetLastError();
    return perr == cudaSuccess &&
                (atts.type == cudaMemoryTypeDevice ||
                 atts.type == cudaMemoryTypeManaged);

#elif defined(CONDUIT_USE_HIP)
    hipPointerAttribute_t atts;
    const hipError_t perr = hipPointerGetAttributes(&atts, ptr);
    // clear last error so other error checking does
    // not pick it up
    hipError_t error = hipGetLastError();
    return perr == hipSuccess &&
                (atts.TYPE_ATTR == hipMemoryTypeDevice ||
                 atts.TYPE_ATTR == hipMemoryTypeUnified);
#else
    (void) ptr;
    return false;
#endif
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Magic Memory
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
MagicMemory::set(void * ptr, int value, size_t num )
{
#if defined(CONDUIT_USE_RAJA)
    bool is_device = DeviceMemory::is_device_ptr(ptr);
    if (is_device)
    {
#if defined(CONDUIT_USE_CUDA)
        cudaMemset(ptr,value,num);
#elif defined(CONDUIT_USE_HIP)
        hipMemset(ptr,value,num);
#endif
    }
    else
    {
        memset(ptr,value,num);
    }
#else
    memset(ptr,value,num);
#endif
}

//-----------------------------------------------------------------------------
void
MagicMemory::copy(void * destination, const void * source, size_t num)
{
#if defined(CONDUIT_USE_RAJA)
    bool src_is_gpu = DeviceMemory::is_device_ptr(source);
    bool dst_is_gpu = DeviceMemory::is_device_ptr(destination);
    if (src_is_gpu && dst_is_gpu)
    {
#if defined(CONDUIT_USE_CUDA)
        cudaMemcpy(destination, source, num, cudaMemcpyDeviceToDevice);
#elif defined(CONDUIT_USE_HIP)
        hipMemcpy(destination, source, num, hipMemcpyDeviceToDevice);
#endif
    }
    else if (src_is_gpu && !dst_is_gpu)
    {
#if defined(CONDUIT_USE_CUDA)
        cudaMemcpy(destination, source, num, cudaMemcpyDeviceToHost);
#elif defined(CONDUIT_USE_HIP)
        hipMemcpy(destination, source, num, hipMemcpyDeviceToHost);
#endif
    }
    else if (!src_is_gpu && dst_is_gpu)
    {
#if defined(CONDUIT_USE_CUDA)
        cudaMemcpy(destination, source, num, cudaMemcpyHostToDevice);
#elif defined(CONDUIT_USE_HIP)
        hipMemcpy(destination, source, num, hipMemcpyHostToDevice);
#endif
    }
    else
    {
        // we are the default memcpy in conduit so this is the normal
        // path
        memcpy(destination,source,num);
    }
#else
    memcpy(destination,source,num);
#endif
}

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
