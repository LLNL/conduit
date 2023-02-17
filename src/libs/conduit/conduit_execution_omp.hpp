// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_execution_omp.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_EXECUTION_OMP_HPP
#define CONDUIT_EXECUTION_OMP_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"

#if defined(_OPENMP)
#include <omp.h>
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

//-----------------------------------------------------------------------------
// -- begin conduit::execution::omp --
//-----------------------------------------------------------------------------
namespace omp
{

//---------------------------------------------------------------------------
struct for_policy
{
    template <typename Func>
    inline void operator()(index_t begin, index_t end, Func &&func)
    {
#if defined(_OPENMP)
        #pragma message("omp::for_policy -> OMP")
        #pragma omp parallel for
        for(index_t i = begin; i < end; i++)
            func(i);
#else
        #pragma message("omp::for_policy -> serial")
        for(index_t i = begin; i < end; i++)
            func(i);
#endif
    }
};

//---------------------------------------------------------------------------
struct sort_policy
{
    template <typename Iterator>
    inline void operator()(Iterator begin, Iterator end)
    {
        #pragma message("omp::sort_policy -> serial")
        // TODO: implement an OpenMP sort like in RAJA.
        std::sort(begin, end);
        // This is only allowed in C++14 or later.
        //this->operator()(begin, end, [](auto &lhs, auto &rhs) { return lhs < rhs; });
    }

    template <typename Iterator, typename Predicate>
    inline void operator()(Iterator begin, Iterator end, Predicate &&predicate)
    {
        // TODO: implement an OpenMP sort like in RAJA.
        #pragma message("omp::sort_policy -> serial")
        std::sort(begin, end, predicate);
    }
};

}
//-----------------------------------------------------------------------------
// -- end conduit::execution::omp --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::execution --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
