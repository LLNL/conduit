// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: cmake_cxx11_check_if_long_long_is_std_int64.cpp
///
//-----------------------------------------------------------------------------

#include <cstdint>
#include <type_traits>

static_assert(std::is_same<long long, std::int64_t>::value,
              "error: long long != std::int64_t");

int main()
{
    return 0;
}
