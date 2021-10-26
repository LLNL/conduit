// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: blueprint_test_helpers.hpp
///
//-----------------------------------------------------------------------------

#ifndef BLUEPRINT_TEST_HELPERS_HPP
#define BLUEPRINT_TEST_HELPERS_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include <conduit.hpp>
#include <conduit_node.hpp>

//-----------------------------------------------------------------------------
// -- begin table --
//-----------------------------------------------------------------------------
namespace table
{

//-----------------------------------------------------------------------------
void compare_to_baseline(const conduit::Node &test,
    const conduit::Node &baseline);

}
//-----------------------------------------------------------------------------
// -- end table --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin parition --
//-----------------------------------------------------------------------------
namespace partition
{

//-----------------------------------------------------------------------------
/**
Make a field that selects domains like this.
+----+----+
| 3  |  5 |
|  +-|-+  |
|  +4|4|  |
+--+-+-+--|
|  +1|1|  |
|  +-|-+  |
| 0  |  2 |
+----+----+

*/
void
add_field_selection_field(int cx, int cy, int cz,
    int iquad, int jquad, conduit::index_t main_dom, conduit::index_t fill_dom,
    conduit::Node &output);


//-----------------------------------------------------------------------------
void
make_field_selection_example(conduit::Node &output, int mask);

}
//-----------------------------------------------------------------------------
// -- end parition --
//-----------------------------------------------------------------------------

#endif
