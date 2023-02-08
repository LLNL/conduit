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

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"

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

//---------------------------------------------------------------------------
class Serial
{
public:
    template <typename Func>
    inline void for_each(index_t begin, index_t end, Func &&func)
    {
        for(index_t i = begin; i < end; i++)
            func(i);
    }

    template <typename Iterator>
    inline void sort(Iterator begin, Iterator end)
    {
        std::sort(begin, end);
    }

    template <typename Iterator, typename Predicate>
    inline void sort(Iterator begin, Iterator end, Predicate &&predicate)
    {
        std::sort(begin, end, predicate);
    }
};

//---------------------------------------------------------------------------
template <index_t Threshold = 0>
class Parallel
{
    constexpr static index_t THRESHOLD = Threshold;
public:
    template <typename Func>
    inline void for_each(index_t begin, index_t end, Func &&func)
    {
#if defined(_OPENMP)
        #pragma message "Parallel::for_each OMP"

        if(THRESHOLD > 0 && (end - begin) < THRESHOLD)
        {
            // For small numbers of elements, serial can be faster.
            for(index_t i = begin; i < end; i++)
            {
                func(i);
            }
        }
        else
        {
            #pragma omp parallel for
            for(index_t i = begin; i < end; i++)
            {
                func(i);
            }
        }

#elif defined(CONDUIT_CXX17)
        // C++ does not like this. It wants iterators.
        #pragma message "Parallel::for_each -> std::for_each par"
        std::for_each(std::execution::par, begin, end, func);
#else
        #pragma message "Parallel::for_each serial"
        // Back up to serial.
        for(index_t i = begin; i < end; i++)
            func(i);
#endif
    }

    template <typename Iterator>
    inline void sort(Iterator begin, Iterator end)
    {
#ifdef CONDUIT_CXX17
        #pragma message "Parallel::sort -> std::sort par"
        std::sort(std::execution::par, begin, end);
#else
        #pragma message "Parallel::sort serial"
        std::sort(begin, end);
#endif
    }

    template <typename Iterator, typename Predicate>
    inline void sort(Iterator begin, Iterator end, Predicate &&predicate)
    {
#ifdef CONDUIT_CXX17
        #pragma message "Parallel::sort -> std::sort par"
        std::sort(std::execution::par, begin, end, predicate);
#else
        #pragma message "Parallel::sort serial"
        std::sort(begin, end, predicate);
#endif
    }
};

//---------------------------------------------------------------------------
template <typename ExecutionPolicy, typename Func>
inline void
for_each(size_t start, size_t end, Func &&func)
{
    ExecutionPolicy().for_each(start, end, func);
}

//---------------------------------------------------------------------------
template <typename ExecutionPolicy, typename Iterator>
inline void
sort(Iterator start, Iterator end)
{
    ExecutionPolicy().sort(start, end);
}

//---------------------------------------------------------------------------
template <typename ExecutionPolicy, typename Iterator, typename Predicate>
inline void
sort(Iterator start, Iterator end, Predicate &&predicate)
{
    ExecutionPolicy().sort(start, end, predicate);
}

//-----------------------------------------------------------------------------
// -- end conduit::execution --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
