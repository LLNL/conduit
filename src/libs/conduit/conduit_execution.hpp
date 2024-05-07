// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_execution.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_EXECUTION_HPP
#define CONDUIT_EXECUTION_HPP

#include "conduit_config.h"

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit_execution_serial.hpp"
#if defined(CONDUIT_USE_OPENMP)
#include "conduit_execution_omp.hpp"
#endif

//-----------------------------------------------------------------------------
// -- begin conduit --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::execution --
//-----------------------------------------------------------------------------
namespace execution
{

#if defined(CONDUIT_USE_RAJA)
//---------------------------------------------------------------------------//
// RAJA_ON policies for when raja is on
//---------------------------------------------------------------------------//
struct SerialExec
{
    using for_policy = RAJA::seq_exec;
#if defined(ASCENT_CUDA_ENABLED)
    // the cuda/hip policy for reductions can be used
    // by other backends, and this should suppress
    // erroneous host device warnings
    using reduce_policy = RAJA::cuda_reduce;
#elif  defined(ASCENT_HIP_ENABLED)
    using reduce_policy = RAJA::hip_reduce;
#else
    using reduce_policy = RAJA::seq_reduce;
#endif
    using atomic_policy = RAJA::seq_atomic;
    static std::string memory_space;
    
    using sort_policy = seq::sort_policy; // ?
};

//---------------------------------------------------------------------------
#if defined(CONDUIT_USE_CUDA) // TODO who is this
struct CudaExec
{
    using for_policy    = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
    using reduce_policy = RAJA::cuda_reduce;
    using atomic_policy = RAJA::cuda_atomic;
    using sort_policy = seq::sort_policy; // ?
    static std::string memory_space;
};
#endif

#if defined(CONDUIT_USE_HIP)
//---------------------------------------------------------------------------
struct HipExec
{
    using for_policy    = RAJA::hip_exec<HIP_BLOCK_SIZE>;
    using reduce_policy = RAJA::hip_reduce;
    using atomic_policy = RAJA::hip_atomic;
    using sort_policy = omp::sort_policy; // ?
    static std::string memory_space;
};
#endif

#if defined(CONDUIT_USE_OPENMP)
//---------------------------------------------------------------------------
struct OpenMPExec
{
    using for_policy = RAJA::omp_parallel_for_exec;
#if defined(CONDUIT_USE_CUDA)
    // the cuda policy for reductions can be used
    // by other backends, and this should suppress
    // erroneous host device warnings
    using reduce_policy = RAJA::cuda_reduce;
#elif defined(CONDUIT_USE_HIP)
    using reduce_policy = RAJA::hip_reduce;
#else
    using reduce_policy = RAJA::omp_reduce;
#endif
    using atomic_policy = RAJA::omp_atomic;
    static std::string memory_space;

    using sort_policy = omp::sort_policy; // ?
};
#endif


#else
//---------------------------------------------------------------------------//
// RAJA_OFF policies for when raja is OFF
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------
struct SerialExec
{
    using for_policy = seq::for_policy;
    using reduce_policy = ExecPolicyTagA;
    using atomic_policy = ExecPolicyTagA;
    using sort_policy = seq::sort_policy;
};

//---------------------------------------------------------------------------
#if defined(CONDUIT_USE_CUDA) // TODO who is this
struct CudaExec
{
    using for_policy = seq::for_policy;
    using reduce_policy = ExecPolicyTagA;
    using atomic_policy = ExecPolicyTagA;
    using sort_policy = seq::sort_policy;
};
#endif

#if defined(CONDUIT_USE_HIP)
//---------------------------------------------------------------------------
struct HipExec
{
    using for_policy = omp::for_policy;
    using reduce_policy = ExecPolicyTagA;
    using atomic_policy = ExecPolicyTagA;
    using sort_policy = omp::sort_policy;
};
#endif

#if defined(CONDUIT_USE_OPENMP)
//---------------------------------------------------------------------------
struct OpenMPExec
{
    using for_policy = omp::for_policy;
    using reduce_policy = ExecPolicyTagA;
    using atomic_policy = ExecPolicyTagA;
    using sort_policy = omp::sort_policy;
};
#endif


#endif




// cuda, hip, openMP, we might alias one of these to be device (cuda or hip)

//---------------------------------------------------------------------------
template <typename ExecutionPolicy, typename Func>
inline void
for_all(size_t begin, size_t end, Func &&func)
{
    using policy = typename ExecutionPolicy::for_policy;
    policy exec;
    exec(begin, end, func);
}

template <typename ExecutionPolicy, typename Iterator>
inline void
sort(Iterator begin, Iterator end)
{
    using policy = typename ExecutionPolicy::sort_policy;
    policy exec;
    exec(begin, end);
}

template <typename ExecutionPolicy, typename Iterator, typename Predicate>
inline void
sort(Iterator begin, Iterator end, Predicate &&predicate)
{
    using policy = typename ExecutionPolicy::sort_policy;
    policy exec;
    exec(begin, end, predicate);
}

}
//-----------------------------------------------------------------------------
// -- end conduit::execution --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
