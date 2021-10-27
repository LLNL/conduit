// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_io_csv.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_relay_io_csv.hpp"

#include <fstream>
#include <iostream>

#include "conduit_log.hpp"
#include "conduit_blueprint_table.hpp"

using conduit::utils::log::quote;

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay --
//-----------------------------------------------------------------------------
namespace relay
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay::io --
//-----------------------------------------------------------------------------
namespace io
{

// Static functions, internal types

//-----------------------------------------------------------------------------
struct OptionsCsv {
    int column_width = 0;
};

//-----------------------------------------------------------------------------
/**
@brief Initialize an OptionsCsv with default values then update with values
    from options node.
*/
static OptionsCsv
parse_options(const Node &options)
{
    OptionsCsv retval;
    retval.column_width = 0;
    if(options.has_child("fixed_width"))
    {
        const Node &fixed_width = options["fixed_width"];
        if(fixed_width.dtype().is_integer())
        {
            retval.column_width = fixed_width.to_int();
            if(retval.column_width < 0) retval.column_width = 0;
        }
        else
        {
            CONDUIT_INFO("Option " << quote("fixed_width") << "passed to"
                << " write_csv, but it is not an integer. Ignored.");
        }
    }
    return retval;
}

//-----------------------------------------------------------------------------
static index_t
get_nrows(const Node &table)
{
    const Node &values = table["values"];
    index_t retval = 0;
    if(values.number_of_children() > 0)
    {
        const Node &ref = values[0];
        const index_t nc = ref.number_of_children();
        retval = nc > 0
            ? ref[0].dtype().number_of_elements()
            : ref.dtype().number_of_elements();
    }
    return retval;
}

//-----------------------------------------------------------------------------
static void
write_header(const Node &values, std::ofstream &fout)
{
    const index_t ncols = values.number_of_children();
    for(index_t col = 0; col < ncols; col++)
    {
        const Node &value = values[col];
        const std::string base_name = value.name();
        const index_t nc = value.number_of_children();
        if(nc > 0)
        {
            // Each column is "base_name/comp_name"
            for(index_t c = 0; c < nc; c++)
            {
                fout << base_name << "/" << value[c].name();
                if(c != (nc - 1)) fout << ", ";
            }
        }
        else
        {
            fout << base_name;
        }

        if(col != (ncols - 1)) fout << ", ";
    }
    fout << std::endl;
}

//-----------------------------------------------------------------------------
static void
write_element(const Node &e, std::ostream &fout)
{
    if(e.dtype().is_unsigned_integer())
    {
        fout << e.to_uint64();
    }
    else if(e.dtype().is_integer())
    {
        fout << e.to_int64();
    }
    else if(e.dtype().is_float32())
    {
        fout << e.as_float32();
    }
    else if(e.dtype().is_float64())
    {
        fout << e.as_float64();
    }
    else if(e.dtype().is_string())
    {
        fout << e.as_string();
    }
}

//-----------------------------------------------------------------------------
static void
write_row_based(const Node &table, const std::string &path,
    const OptionsCsv &)
{
    const Node &values = table["values"];

    std::ofstream fout(path);
    write_header(values, fout);

    const index_t nrows = get_nrows(table);
    const index_t ncols = values.number_of_children();
    Node temp;
    for(index_t row = 0; row < nrows; row++)
    {
        for(index_t col = 0; col < ncols; col++)
        {
            const Node &value = values[col];
            const index_t nc = value.number_of_children();
            if(nc > 0)
            {
                for(index_t c = 0; c < nc; c++)
                {
                    const Node &comp = value[c];
                    temp.set_external(DataType(comp.dtype().id(), 1),
                        const_cast<void*>(comp.element_ptr(row)));
                    write_element(temp, fout);
                    if(c != (nc - 1)) fout << ", ";
                }
            }
            else
            {
                temp.set_external(DataType(value.dtype().id(), 1),
                    const_cast<void*>(value.element_ptr(row)));
                write_element(temp, fout);
            }

            if(col != (ncols - 1)) fout << ", ";
        }
        fout << "\n";
    }
    fout.flush();
}

//-----------------------------------------------------------------------------
static void
write_single_table(const Node &table, const std::string &path,
    const OptionsCsv &options)
{
    if(options.column_width == 0)
    {
        write_row_based(table, path, options);
    }
}

//-----------------------------------------------------------------------------
void
write_csv(const Node &table, const std::string &path, const Node &options)
{
    Node info;
    const bool ok = blueprint::table::verify(table, info);
    if(!ok)
    {
        CONDUIT_ERROR("The node provided to write_csv must be a valid "
            << "blueprint table!");
    }

    const OptionsCsv opts = parse_options(options);

    if(table.has_child("values"))
    {
        write_single_table(table, path, opts);
    }
}

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::io --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
