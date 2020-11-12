// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_docs_tutorial_helpers.hpp
///
//-----------------------------------------------------------------------------
#include <iostream>

#define BEGIN_EXAMPLE(tag)                                          \
{                                                                   \
    std::cout << "BEGIN_EXAMPLE(\"" << tag << "\")" << std::endl;   \
}                                                                   \

#define END_EXAMPLE(tag)                                            \
{                                                                   \
    std::cout << "END_EXAMPLE(\"" << tag << "\")" << std::endl;     \
}                                                                   \

