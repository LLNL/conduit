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

enum policy_id { Serial, Device, Cuda, Hip, OpenMP };

//---------------------------------------------------------------------------//
// Runtime Policy Object
//---------------------------------------------------------------------------//
class ExecPolicy
{
public:
    ExecPolicy(policy_id _id): id(_id) {}
    policy_id id;
};

#if defined(CONDUIT_USE_RAJA)
//---------------------------------------------------------------------------//
// RAJA_ON policies for when raja is on
//---------------------------------------------------------------------------//
struct SerialExec
{
    using for_policy = RAJA::seq_exec;
#if defined(CONDUIT_USE_CUDA)
    // the cuda/hip policy for reductions can be used
    // by other backends, and this should suppress
    // erroneous host device warnings
    using reduce_policy = RAJA::cuda_reduce;
#elif  defined(CONDUIT_USE_HIP)
    using reduce_policy = RAJA::hip_reduce;
#else
    using reduce_policy = RAJA::seq_reduce;
#endif
    using atomic_policy = RAJA::seq_atomic;
    static std::string memory_space;
};

// cuda, hip, openMP, we might alias one of these to be device (cuda or hip)

//---------------------------------------------------------------------------
#if defined(CONDUIT_USE_CUDA) // TODO who is this
struct CudaExec
{
    using for_policy    = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
    using reduce_policy = RAJA::cuda_reduce;
    using atomic_policy = RAJA::cuda_atomic;
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
};
#endif


#else
//---------------------------------------------------------------------------//
// RAJA_OFF policies for when raja is OFF
//---------------------------------------------------------------------------//
struct EmptyPolicy
{};
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
struct SerialExec
{
    using for_policy = EmptyPolicy;
    using reduce_policy = EmptyPolicy;
    using atomic_policy = EmptyPolicy;
    static std::string memory_space;
};

#endif

//---------------------------------------------------------------------------//
// invoke functor with concrete template tag
//---------------------------------------------------------------------------//
template <typename ExecPolicyTag, typename Function>
inline void invoke(ExecPolicyTag &exec, Function&& func) noexcept
{
    func(exec);
}

//---------------------------------------------------------------------------//
// runtime to concrete template tag dispatch of a functor
//---------------------------------------------------------------------------//
template <typename Function>
void dispatch(ExecPolicy policy, Function&& func)
{
    switch(policy.policy_id)
    {
        case Device: // TODO I have made device prefer cuda over hip if both are available
#if defined(CONDUIT_USE_RAJA) && defined(CONDUIT_USE_CUDA)
            CudaExec ce;
            return invoke(ce, func);
#elif defined(CONDUIT_USE_RAJA) && defined(CONDUIT_USE_HIP)
            HipExec he;
            return invoke(he, func);
#else
            CONDUIT_ERROR("bad choice");
#endif
        case Cuda:
#if defined(CONDUIT_USE_RAJA) && defined(CONDUIT_USE_CUDA)
            CudaExec ce;
            return invoke(ce, func);
#else
            CONDUIT_ERROR("bad choice");
#endif
        case Hip:
#if defined(CONDUIT_USE_RAJA) && defined(CONDUIT_USE_HIP)
            HipExec he;
            return invoke(he, func);
#else
            CONDUIT_ERROR("bad choice");
#endif
        case OpenMP:
#if defined(CONDUIT_USE_RAJA) && defined(CONDUIT_USE_OPENMP)
            OpenMPExec ompe;
            return invoke(ompe, func);
#else
            CONDUIT_ERROR("bad choice");
#endif
        case Serial:
        default:
            SerialExec se;
            return invoke(se, func);
    }
}

//---------------------------------------------------------------------------//
// mock up of a raja like forall implementation 
//---------------------------------------------------------------------------//
template <typename ExecPolicy,typename Kernel>
inline void new_forall_exec(const int& begin,
                        const int& end,
                        Kernel&& kernel) noexcept
{

    std::cout << typeid(ExecPolicy).name() << "  START" << std::endl;
    for(int i=begin;i<end;i++)
    {
        kernel(i);
    }
    std::cout << typeid(ExecPolicy).name() << "  END" << std::endl;
}


//---------------------------------------------------------------------------//
// invoke forall with concrete template tag
//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename Kernel>
inline void new_forall(const int& begin,
                   const int& end,
                   Kernel&& kernel) noexcept
{
    new_forall_exec<ExecPolicy>(begin,end, std::forward<Kernel>(kernel));
}

//---------------------------------------------------------------------------//
// runtime to concrete template tag dispatch of a forall
//---------------------------------------------------------------------------//
template <typename Kernel>
inline void new_forall(ExecPolicy &policy,
                   const int& begin,
                   const int& end,
                   Kernel&& kernel) noexcept
{
    switch(policy.policy_id)
    {
        case Device: // TODO I have made device prefer cuda over hip if both are available
#if defined(CONDUIT_USE_RAJA) && defined(CONDUIT_USE_CUDA)
            CudaExec ce;
            new_forall<ce>(begin,end,std::forward<Kernel>(kernel));
            break;
#elif defined(CONDUIT_USE_RAJA) && defined(CONDUIT_USE_HIP)
            HipExec he;
            new_forall<he>(begin,end,std::forward<Kernel>(kernel));
            break;
#else
            CONDUIT_ERROR("bad choice");
#endif
        case Cuda:
#if defined(CONDUIT_USE_RAJA) && defined(CONDUIT_USE_CUDA)
            CudaExec ce;
            new_forall<ce>(begin,end,std::forward<Kernel>(kernel));
            break;
#else
            CONDUIT_ERROR("bad choice");
#endif
        case Hip:
#if defined(CONDUIT_USE_RAJA) && defined(CONDUIT_USE_HIP)
            HipExec he;
            new_forall<he>(begin,end,std::forward<Kernel>(kernel));
            break;
#else
            CONDUIT_ERROR("bad choice");
#endif
        case OpenMP:
#if defined(CONDUIT_USE_RAJA) && defined(CONDUIT_USE_OPENMP)
            HipExec ompe;
            new_forall<ompe>(begin,end,std::forward<Kernel>(kernel));
            break;
#else
            CONDUIT_ERROR("bad choice");
#endif
        case Serial:
        default:
            SerialExec se;
            new_forall<se>(begin,end,std::forward<Kernel>(kernel));
            break;
    }
}

// OLD BRAD STUFF below

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
