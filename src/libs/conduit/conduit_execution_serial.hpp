// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_execution_serial.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_EXECUTION_SERIAL_HPP
#define CONDUIT_EXECUTION_SERIAL_HPP

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

//-----------------------------------------------------------------------------
// -- begin conduit::execution::seq --
//-----------------------------------------------------------------------------
namespace seq
{

//---------------------------------------------------------------------------
struct for_policy
{
    template <typename Func>
    inline void operator()(index_t begin, index_t end, Func &&func)
    {
        for(index_t i = begin; i < end; i++)
            func(i);
    }
};

//---------------------------------------------------------------------------
struct sort_policy
{
    template <typename Iterator>
    inline void operator()(Iterator begin, Iterator end)
    {
        std::sort(begin, end);
    }

    template <typename Iterator, typename Predicate>
    inline void operator()(Iterator begin, Iterator end, Predicate &&predicate)
    {
        std::sort(begin, end, predicate);
    }
};

}
//-----------------------------------------------------------------------------
// -- end conduit::execution::seq --
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
