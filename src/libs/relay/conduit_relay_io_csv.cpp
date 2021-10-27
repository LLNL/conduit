// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_io_csv.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_relay_io_csv.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "conduit_log.hpp"
#include "conduit_blueprint_table.hpp"

using conduit::utils::log::quote;

const char child_sep = '/';
const char *whitespace = " \t\n\r\f\v";

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

    // Open the file
    std::ofstream fout(path);
    if(!fout.is_open())
    {
        CONDUIT_ERROR("Unable to open file " << quote(path) << ".");
        return;
    }

    // First line, column names
    write_header(values, fout);

    // Write each row: col0, col1, col2, col3 ...
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
}

//-----------------------------------------------------------------------------
static void
write_single_table(const Node &table, const std::string &path,
    const OptionsCsv &options)
{
    // TODO: Write column based writer and benchmark performance
    write_row_based(table, path, options);
}

//-----------------------------------------------------------------------------
static Node &
add_column(const std::string &name, Node &values)
{
    // Add the column to the table
    std::size_t child_pos = name.rfind(child_sep);
    if(child_pos != std::string::npos)
    {
        // Q: What if the string contains many '/'s
        // mcarray
        std::string base_name = name.substr(0, child_pos);
        Node &base_node = values[base_name];

        child_pos++;
        std::string child_name = name.substr(child_pos, name.length() - child_pos);
        return base_node.add_child(child_name);
    }
    else
    {
        return values.add_child(name);
    }
}

//-----------------------------------------------------------------------------
static void
trim(std::string &str, const char *t = whitespace)
{
    // ltrim, rtrim
    str.erase(0, str.find_first_not_of(t));
    str.erase(str.find_last_not_of(t) + 1);
}

//-----------------------------------------------------------------------------
void
read_csv(const std::string &path, const Node &, Node &table)
{
    table.reset();
    std::ifstream fin(path);
    if(!fin.is_open())
    {
        CONDUIT_ERROR("Unable to open file " << quote(path) << ".");
        return;
    }

    // Some basic sanity checks on the file
    // Q: Need to support comment character?

    // Make sure the file has data
    std::string first_line;
    if(!std::getline(fin, first_line))
    {
        CONDUIT_ERROR("The file " << quote(path) << "appears to be empty.");
        return;
    }

    // Get number of rows in file
    const auto data_start = fin.tellg();
    std::string line;
    index_t nrows = 0;
    while(std::getline(fin, line))
    {
        nrows++;
    }

    // Make sure the file is a CSV file.
    // Q: What if it was one column
    const char del = ',';
    std::size_t pos = first_line.find(del);
    if(pos == std::string::npos)
    {
        CONDUIT_ERROR("The file " << quote(path) << "does not appear to contain CSV data.");
        return;
    }

    // TODO: Handle CSV files without column names
    // Allocate the output table
    const DataType dtype(DataType::FLOAT32_ID, nrows);
    std::vector<std::string> column_names;
    Node &values = table["values"];
    {
        std::size_t start = 0;
        std::size_t end = first_line.find(del);
        while(end != std::string::npos)
        {
            const auto len = end - start;
            column_names.push_back(first_line.substr(start, len));

            trim(column_names.back(), whitespace);
            Node &col = add_column(column_names.back(), values);
            col.set_dtype(dtype);

            // Update positions
            start = end + 1;
            end = first_line.find(del, start);
        }
        // Get the last name
        column_names.push_back(first_line.substr(start));
        trim(column_names.back(), whitespace);
        Node &col = add_column(column_names.back(), values);
        col.set_dtype(dtype);
    }

    line.erase();
    fin.clear();
    fin.seekg(data_start, fin.beg);
    index_t row = 0;
    while(std::getline(fin, line))
    {
        std::size_t icol = 0;
        std::size_t start = 0;
        std::size_t end = line.find(del);
        while(end != std::string::npos)
        {
            const auto len = end - start;
            if(icol > column_names.size())
            {
                CONDUIT_ERROR("Error while reading file, row " << row << " contains too many column entries!");
                return;
            }

            Node *n = values.fetch_ptr(column_names[icol]);
            if(!n)
            {
                CONDUIT_ERROR("The allocated output table does not contain the path " << quote(column_names[icol]) << ".");
                return;
            }

            std::string data = line.substr(start, len);
            float *d = static_cast<float*>(n->element_ptr(row));
            try
            {
                *d = std::stof(data);
            }
            catch(const std::exception &e)
            {
                CONDUIT_ERROR(e.what());
                return;
            }

            // Update positions
            icol++;
            start = end + 1;
            end = line.find(del, start);
        }
        // Get the last name
        if(icol > column_names.size())
        {
            CONDUIT_ERROR("Error while reading file, row " << row << " contains too many column entries!");
            return;
        }

        Node *n = values.fetch_ptr(column_names[icol]);
        if(!n)
        {
            CONDUIT_ERROR("The allocated output table does not contain the path " << quote(column_names[icol]) << ".");
            return;
        }

        std::string data = line.substr(start);
        float *d = static_cast<float*>(n->element_ptr(row));
        try
        {
            *d = std::stof(data);
        }
        catch(const std::exception &e)
        {
            CONDUIT_ERROR(e.what());
            return;
        }
        row++;
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
    // else
    // {
    //     write_multiple_tables(table, path, opts);
    // }
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
