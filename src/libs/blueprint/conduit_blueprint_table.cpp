// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_table.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
// std lib includes will go here

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_table.hpp"
#include "conduit_log.hpp"
#include "conduit_blueprint_mcarray.hpp"

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin internal helper functions --
//-----------------------------------------------------------------------------
static const std::string PROTOCOL = "table";

//-----------------------------------------------------------------------------
static bool
verify_table_values(const Node &n, Node &info)
{
    bool res = true;

    // Each entry in n represents a column in the table.
    // All columns must have the same length.
    auto columns = n.children();
    index_t num_rows = 0;
    index_t num_cols = 0;
    bool first_column = true;
    while(columns.has_next())
    {
        const Node &n_col = columns.next();
        Node &i_col = info[n_col.name()];
        index_t this_num_elems = 0;
        bool this_res = true;

        // Each entry must be a data array or a valid mlarray
        if(n_col.dtype().is_list() || n_col.dtype().is_object())
        {
            this_res = blueprint::mcarray::verify(n_col, i_col);
            this_num_elems = (res) ? n_col[0].dtype().number_of_elements() : 0;
        }
        else if(n_col.dtype().is_empty())
        {
            this_res = false;
        }
        else
        {
            this_num_elems = n_col.dtype().number_of_elements();
        }

        // Record the error if we have one
        if(!this_res)
        {
            utils::log::error(info, PROTOCOL, "child " + utils::log::quote(n_col.name()) + " is not a data_array or valid mcarray.");
        }
        else
        {
            // Check that number of elements matches for each column
            // If this is the first column, set num_rows first.
            num_rows = (first_column) ? this_num_elems : num_rows;
            i_col["elements"] = this_num_elems;
            if(num_rows != this_num_elems)
            {
                utils::log::error(info, PROTOCOL, "child " + utils::log::quote(n_col.name()) + " does not contain the correct number of elements.");
                this_res = false;
            }
        }

        utils::log::validation(i_col, this_res);
        res &= this_res;
        first_column = false;
        num_cols++;
    }

    // Add some extra info for a valid table.
    if(res)
    {
        info["columns"] = num_cols;
        info["rows"] = num_rows;
    }
    utils::log::validation(info, res);
    return res;
}

//-----------------------------------------------------------------------------
static bool
verify_single_table(const Node &n, Node &info)
{
    bool res = true;

    // "values" child must exist
    if(!n.has_child("values"))
    {
        res = false;
        utils::log::error(info, PROTOCOL, "missing child" + utils::log::quote("values", 1));
    }

    if(res)
    {
        // "values" child must be a list or an object
        const Node &n_values = n["values"];
        Node &i_values = info["values"];
        if(!n_values.dtype().is_object() && !n_values.dtype().is_list())
        {
            res = false;
            utils::log::error(i_values, PROTOCOL, utils::log::quote("values", 0) + " must be an object or a list.");
        }

        if(res)
        {
            res = verify_table_values(n_values, i_values);
        }
    }

    utils::log::validation(info, res);
    return res;
}

//-----------------------------------------------------------------------------
static bool
verify_many_tables(const conduit::Node &n, conduit::Node &info)
{
    bool res = true;

    // Check if each child is a table
    auto children = n.children();
    index_t num_tables = 0;
    while(children.has_next())
    {
        const Node &child = children.next();
        Node &info_child = info[child.name()];
        res &= verify_single_table(child, info_child);
        num_tables++;
    }

    // Check if there were actually any children
    res &= num_tables > 0;

    if(res)
    {
        info["tables"] = num_tables;
    }
    utils::log::validation(info, res);
    return res;
}

//-----------------------------------------------------------------------------
// -- end internal helper functions --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::table --
//-----------------------------------------------------------------------------
namespace table
{

//-----------------------------------------------------------------------------
bool verify(const conduit::Node &n,
            conduit::Node &info)
{
    bool res = true;
    info.reset();

    if(n.has_child("values"))
    {
        res = verify_single_table(n, info);
    }
    else
    {
        res = verify_many_tables(n, info);
    }
    return res;
}

//-----------------------------------------------------------------------------
bool verify(const std::string &,
            const conduit::Node &,
            conduit::Node &info)
{
    // Table doesn't currently provide any nested protocols

    info.reset();
    utils::log::validation(info,false);
    return false;
}

}
//-----------------------------------------------------------------------------
// -- end conduit::table --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
