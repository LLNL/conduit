// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: blueprint_test_helpers.cpp
///
//-----------------------------------------------------------------------------
#include "blueprint_test_helpers.hpp"

#include <conduit_blueprint.hpp>

#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin table --
//-----------------------------------------------------------------------------
namespace table
{

//-----------------------------------------------------------------------------
static void
compare_to_baseline_leaf(const Node &test, const Node &baseline)
{
    if(test.dtype().is_empty() || test.dtype().is_list() || test.dtype().is_object()
        || baseline.dtype().is_empty() || baseline.dtype().is_list()
        || baseline.dtype().is_object())
    {
        CONDUIT_ERROR("compare_to_baseline_leaf only operates on leaf nodes.");
    }
    // Sometimes when we read from a file the data types don't match.
    // Convert test to the same type as baseline then compare.
    Node temp, info;
    if(test.dtype().id() != baseline.dtype().id())
    {
        test.to_data_type(baseline.dtype().id(), temp);
    }
    else
    {
        temp.set_external(test);
    }
    EXPECT_FALSE(baseline.diff(temp, info)) << info.to_json();
}

//-----------------------------------------------------------------------------
static void
compare_to_baseline_values(const Node &test, const Node &baseline)
{
    ASSERT_EQ(baseline.number_of_children(), test.number_of_children());
    for(index_t j = 0; j < baseline.number_of_children(); j++)
    {
        const Node &baseline_value = baseline[j];
        const Node &test_value = test[j];
        EXPECT_EQ(baseline_value.name(), test_value.name());
        if(baseline_value.dtype().is_list() || baseline_value.dtype().is_object())
        {
            // mcarray
            ASSERT_EQ(baseline_value.number_of_children(), test_value.number_of_children());
            EXPECT_EQ(baseline_value.dtype().is_list(), test_value.dtype().is_list());
            EXPECT_EQ(baseline_value.dtype().is_object(), test_value.dtype().is_object());
            for(index_t k = 0; k < baseline_value.number_of_children(); k++)
            {
                const Node &baseline_comp = baseline_value[k];
                const Node &test_comp = test_value[k];
                EXPECT_EQ(baseline_comp.name(), test_comp.name());
                compare_to_baseline_leaf(test_comp, baseline_comp);
            }
        }
        else
        {
            // data array
            compare_to_baseline_leaf(test_value, baseline_value);
        }
    }
}

//-----------------------------------------------------------------------------
void compare_to_baseline(const conduit::Node &test,
    const conduit::Node &baseline)
{
    Node info;
    ASSERT_TRUE(blueprint::table::verify(baseline, info));
    ASSERT_TRUE(blueprint::table::verify(test, info));
    if(baseline.has_child("values"))
    {
        const Node &baseline_values = baseline["values"];
        const Node &test_values = test["values"];
        compare_to_baseline_values(test_values, baseline_values);
    }
    else
    {
        ASSERT_EQ(baseline.number_of_children(), test.number_of_children());
        for(index_t i = 0; i < baseline.number_of_children(); i++)
        {
            EXPECT_EQ(baseline[i].name(), test[i].name());
            const Node &baseline_values = baseline[i]["values"];
            const Node &test_values = test[i]["values"];
            compare_to_baseline_values(test_values, baseline_values);
        }
    }
}

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
void
add_field_selection_field(int cx, int cy, int cz,
    int iquad, int jquad, conduit::index_t main_dom, conduit::index_t fill_dom,
    conduit::Node &output)
{
    std::vector<conduit::int64> values(cx*cy*cz, main_dom);
    int sq = 2*jquad + iquad;
    int idx = 0;
    for(int k = 0; k < cz; k++)
    for(int j = 0; j < cy; j++)
    for(int i = 0; i < cx; i++)
    {
        int ci = (i < cx/2) ? 0 : 1;
        int cj = (j < cy/2) ? 0 : 1;
        int csq = 2*cj + ci;
        if(csq == sq)
            values[idx] = fill_dom;
        idx++;
    }
    output["fields/selection_field/type"] = "scalar";
    output["fields/selection_field/association"] = "element";
    output["fields/selection_field/topology"] = "mesh";
    output["fields/selection_field/values"].set(values);
}


//-----------------------------------------------------------------------------
void
make_field_selection_example(conduit::Node &output, int mask)
{
    int nx = 11, ny = 11, nz = 3;
    int m = 1, dc = 0;
    for(int i = 0; i < 4; i++)
    {
        if(m & mask)
            dc++;
        m <<= 1;
    }

    if(mask & 1)
    {
        conduit::Node &dom0 = (dc > 1) ? output.append() : output;
        conduit::blueprint::mesh::examples::braid("uniform", nx, ny, nz, dom0);
        dom0["state/cycle"] = 1;
        dom0["state/domain_id"] = 0;
        dom0["coordsets/coords/origin/x"] = 0.;
        dom0["coordsets/coords/origin/y"] = 0.;
        dom0["coordsets/coords/origin/z"] = 0.;
        add_field_selection_field(nx-1, ny-1, nz-1, 1,1, 0, 11, dom0);
    }

    if(mask & 2)
    {
        conduit::Node &dom1 = (dc > 1) ? output.append() : output;
        conduit::blueprint::mesh::examples::braid("uniform", nx, ny, nz, dom1);
        auto dx = dom1["coordsets/coords/spacing/dx"].to_float();
        dom1["state/cycle"] = 1;
        dom1["state/domain_id"] = 1;
        dom1["coordsets/coords/origin/x"] = dx * static_cast<double>(nx-1);
        dom1["coordsets/coords/origin/y"] = 0.;
        dom1["coordsets/coords/origin/z"] = 0.;
        add_field_selection_field(nx-1, ny-1, nz-1, 0,1, 22, 11, dom1);
    }

    if(mask & 4)
    {
        conduit::Node &dom2 = (dc > 1) ? output.append() : output;
        conduit::blueprint::mesh::examples::braid("uniform", nx, ny, nz, dom2);
        auto dy = dom2["coordsets/coords/spacing/dy"].to_float();
        dom2["state/cycle"] = 1;
        dom2["state/domain_id"] = 2;
        dom2["coordsets/coords/origin/x"] = 0.;
        dom2["coordsets/coords/origin/y"] = dy * static_cast<double>(ny-1);
        dom2["coordsets/coords/origin/z"] = 0.;
        add_field_selection_field(nx-1, ny-1, nz-1, 1,0, 33, 44, dom2);
    }

    if(mask & 8)
    {
        conduit::Node &dom3 = (dc > 1) ? output.append() : output;
        conduit::blueprint::mesh::examples::braid("uniform", nx, ny, nz, dom3);
        auto dx = dom3["coordsets/coords/spacing/dx"].to_float();
        auto dy = dom3["coordsets/coords/spacing/dy"].to_float();
        dom3["state/cycle"] = 1;
        dom3["state/domain_id"] = 3;
        dom3["coordsets/coords/origin/x"] = dx * static_cast<double>(nx-1);
        dom3["coordsets/coords/origin/y"] = dy * static_cast<double>(ny-1);
        dom3["coordsets/coords/origin/z"] = 0.;
        add_field_selection_field(nx-1, ny-1, nz-1, 0,0, 55, 44, dom3);
    }
}

}
//-----------------------------------------------------------------------------
// -- end parition --
//-----------------------------------------------------------------------------
