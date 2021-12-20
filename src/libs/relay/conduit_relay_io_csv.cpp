// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_io_csv.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_relay_io_csv.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <type_traits>
#include <limits>

#include "conduit_log.hpp"
#include "conduit_blueprint_table.hpp"

using conduit::utils::log::quote;

// Some constants used by writer/reader

// Child separator used by writer to name columns
const char child_sep = '/';

// Prefix used for file names when the given table collection is a list
const std::string table_list_prefix = "table_list_";

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
write_row_based(const Node &table, const std::string &path)
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
write_single_table(const Node &table, const std::string &path)
{
    // IDEA: Write column based writer and benchmark performance
    write_row_based(table, path);
}

//-----------------------------------------------------------------------------
static void
write_multiple_tables(const Node &all_tables, const std::string &base_path)
{
    const index_t ntables = all_tables.number_of_children();
    if(ntables < 1)
    {
        return;
    }

    utils::create_directory(base_path);
    if(all_tables.dtype().is_list())
    {
        for(index_t i = 0; i < ntables; i++)
        {
            const Node &table = all_tables[i];
            const std::string full_path = base_path + utils::file_path_separator()
                + table_list_prefix + std::to_string(i) + ".csv";
            write_single_table(table, full_path);
        }
    }
    else // if(table.dtype().is_object())
    {
        for(index_t i = 0; i < ntables; i++)
        {
            const Node &table = all_tables[i];
            const std::string full_path = base_path + utils::file_path_separator()
                + table.name() + ".csv";
            write_single_table(table, full_path);
        }
    }
}

//-----------------------------------------------------------------------------
static Node &
add_column(const std::string &name, Node &values)
{
    if(name.empty())
    {
        return values.append();
    }

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
add_columns(Node &values, std::vector<std::string> &col_names,
    index_t ncols, const DataType &dtype)
{
    if(col_names.empty())
    {
        for(index_t i = 0; i < ncols; i++)
        {
            Node &col = add_column("", values);
            col.set_dtype(dtype);
            col_names.push_back(col.name());
        }
    }
    else
    {
        for(const std::string &name : col_names)
        {
            Node &col = add_column(name, values);
            col.set_dtype(dtype);
        }
    }
}

//-----------------------------------------------------------------------------
/**
@brief Reads the first line of the file. Populates "col_names" with the names
    of each column. If there are no column names "col_names" will be empty.
@return The number of columns
*/
static index_t
read_column_names(const std::string &line, std::vector<std::string> &col_names,
    const char sep = ',')
{
    col_names.clear();
    std::size_t start = 0;
    std::size_t end = line.find(sep);
    while(end != std::string::npos)
    {
        const auto len = end - start;
        col_names.push_back(line.substr(start, len));
        utils::trim_string(col_names.back());
        // Update positions
        start = end + 1;
        end = line.find(sep, start);
    }
    // Get the last name
    col_names.push_back(line.substr(start));
    utils::trim_string(col_names.back());

    // Check these are column names and not data
    bool all_numbers = true;
    for(const std::string &name : col_names)
    {
        std::size_t idx = 0;
        try
        {
            // NOTE: C++ version of this function can throw
            std::stod(name, &idx);
        }
        catch (...)
        {
            idx = 0;
        }
        if(idx != name.size())
        {
            all_numbers = false;
            break;
        }
    }

    // If all numbers then this row should not be treated as column names
    if(all_numbers)
    {
        col_names.clear();
    }
    return static_cast<index_t>(col_names.size());
}

//-----------------------------------------------------------------------------
template<typename FloatType>
void
read_csv_data(std::ifstream &fin, const std::ifstream::pos_type &offset,
    const std::vector<std::string> &col_names, Node &values,
    const char sep = ',')
{
    static_assert(std::is_floating_point<FloatType>().value,
        "Function can only read floating point types from CSV file.");
    std::string line;
    fin.clear();
    fin.seekg(offset, fin.beg);
    index_t row = 0;
    while(std::getline(fin, line))
    {
        std::size_t icol = 0;
        std::size_t start = 0;
        std::size_t end = line.find(sep);
        while(end != std::string::npos)
        {
            const auto len = end - start;
            if(icol > col_names.size())
            {
                CONDUIT_ERROR("Error while reading file, row " << row << " contains too many column entries!");
                return;
            }

            Node *n = values.fetch_ptr(col_names[icol]);
            if(!n)
            {
                CONDUIT_ERROR("The allocated output table does not contain the path " << quote(col_names[icol]) << ".");
                return;
            }

            std::string data = line.substr(start, len);
            try
            {
                if(std::is_same<float, FloatType>().value)
                {
                    float *d = static_cast<float*>(n->element_ptr(row));
                    *d = std::stof(data);
                }
                else if(std::is_same<double, FloatType>().value)
                {
                    double *d = static_cast<double*>(n->element_ptr(row));
                    *d = std::stod(data);
                }
#ifdef CONDUIT_HAS_LONG_DOUBLE
                else if(std::is_same<long double, FloatType>().value)
                {
                    long double *d = static_cast<long double*>(n->element_ptr(row));
                    *d = std::stold(data);
                }
#endif
            }
            catch (...)
            {
                CONDUIT_ERROR("Unable to parse row " << row << " in column " << icol << "."
                    << " The string " << quote(data) << "is not a number.");
            }

            // Update positions
            icol++;
            start = end + 1;
            end = line.find(sep, start);
        }
        // Get the last name
        if(icol > col_names.size())
        {
            CONDUIT_ERROR("Error while reading file, row " << row << " contains too many column entries!");
            return;
        }

        Node *n = values.fetch_ptr(col_names[icol]);
        if(!n)
        {
            CONDUIT_ERROR("The allocated output table does not contain the path " << quote(col_names[icol]) << ".");
            return;
        }

        std::string data = line.substr(start);
        try
        {
            if(std::is_same<float, FloatType>().value)
            {
                float *d = static_cast<float*>(n->element_ptr(row));
                *d = std::stof(data);
            }
            else if(std::is_same<double, FloatType>().value)
            {
                double *d = static_cast<double*>(n->element_ptr(row));
                *d = std::stod(data);
            }
#ifdef CONDUIT_HAS_LONG_DOUBLE
            else if(std::is_same<long double, FloatType>().value)
            {
                long double *d = static_cast<long double*>(n->element_ptr(row));
                *d = std::stold(data);
            }
#endif
        }
        catch (...)
        {
            CONDUIT_ERROR("Unable to parse row " << row << " in column " << icol << "."
                << " The string " << quote(data) << "is not a number.");
        }
        row++;
    }
}

//-----------------------------------------------------------------------------
static void
read_single_table(const std::string &path, const bool use_float64, Node &table)
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

    std::vector<std::string> column_names;
    const index_t ncols = read_column_names(first_line, column_names);
    // If there was no header for the column names, seek back to start
    if(column_names.empty())
    {
        fin.seekg(0, fin.beg);
    }

    // Get number of rows in file
    const auto data_start = fin.tellg();
    std::string line;
    index_t nrows = 0;
    while(std::getline(fin, line))
    {
        nrows++;
    }

    // Allocate the output table
    const DataType dtype((use_float64) ? DataType::FLOAT64_ID : DataType::FLOAT32_ID,
        nrows);
    Node &values = table["values"];
    add_columns(values, column_names, ncols, dtype);

    if(dtype.is_float())
    {
        read_csv_data<float>(fin, data_start, column_names, values);
    }
    else if(dtype.is_double())
    {
        read_csv_data<double>(fin, data_start, column_names, values);
    }
#ifdef CONDUIT_HAS_LONG_DOUBLE
    else if(dtype.is_long_double())
    {
        read_csv_data<long double>(fin, data_start, column_names, values);
    }
#endif
    else
    {
        CONDUIT_ERROR("Unsupported data type for read_csv_data, needs to be a floating point type.");
    }
}

//-----------------------------------------------------------------------------
static void
read_many_tables(const std::string &path, const bool use_float64, Node &table)
{
    // Path must've been a directory
    std::vector<std::string> dir_contents;
    utils::list_directory_contents(path, dir_contents);

    // We will only attempt to read the csv files
    std::vector<std::string> csv_files;
    for(const auto &filename : dir_contents)
    {
        if(utils::is_file(filename)
            && filename.substr(filename.length() - 4) == ".csv")
        {
            csv_files.push_back(filename);
        }
    }

    bool is_list = true;
    for(const auto &filename : csv_files)
    {
        const auto should_start = filename.rfind(utils::file_path_separator()) + 1;
        const auto loc = filename.find(table_list_prefix);
        if(loc != should_start)
        {
            is_list = false;
            break;
        }
    }

    if(is_list)
    {
        // Extract the index number out of the name
        using idx_pair = std::pair<int, const std::string*>;
        std::vector<idx_pair> list_idxs;
        list_idxs.reserve(csv_files.size());
        for(const std::string &name : csv_files)
        {
            const auto offset = name.rfind(table_list_prefix) + table_list_prefix.size();
            int idx = std::numeric_limits<int>::max();
            // std::cout << name.substr(offset) << std::endl;
            try {
                idx = std::stoi(name.substr(offset));
            }
            catch (...)
            {
                idx = std::numeric_limits<int>::max();
            }
            list_idxs.push_back({idx, &name});
        }

        // Sort buy the index number
        std::sort(list_idxs.begin(), list_idxs.end(),
            [&](const idx_pair &p0, const idx_pair &p1)
        {
            return p0.first < p1.first;
        });

        for(const auto &pair : list_idxs)
        {
            // std::cout << pair.first << " " << *pair.second << std::endl;
            read_single_table(*pair.second, use_float64, table.append());
        }
    }
    else
    {
        for(const auto &filename : csv_files)
        {
            const auto no_ext = filename.size() - 4;
            const auto no_sep = filename.rfind(utils::file_path_separator()) + 1;
            const auto len = no_ext - no_sep;
            read_single_table(filename, use_float64, table[filename.substr(no_sep, len)]);
        }
    }
}

//-----------------------------------------------------------------------------
void
read_csv(const std::string &path, const Node &opts, Node &table)
{
    const bool many_tables = utils::is_directory(path);

    bool use_float64 = false;
    if(opts.has_child("use_float64"))
    {
        const Node &n_use_float64 = opts["use_float64"];
        if(n_use_float64.dtype().is_number())
        {
            use_float64 = opts["use_float64"].to_int() != 0;
        }
        else
        {
            CONDUIT_ERROR("options[" << quote("use_float64") <<
                "] must be a number. It will be treated as a boolean (.to_int() != 0).");
        }
    }

    if(!many_tables)
    {
        read_single_table(path, use_float64, table);
    }
    else
    {
        read_many_tables(path, use_float64, table);
    }
}

//-----------------------------------------------------------------------------
void
write_csv(const Node &table, const std::string &path, const Node &)
{
    Node info;
    const bool ok = blueprint::table::verify(table, info);
    if(!ok)
    {
        CONDUIT_ERROR("The node provided to write_csv must be a valid "
            << "blueprint table!");
    }

    // IDEA: Support a "fixed_width" option and write the file
    //  one column at a time instead of one row at a time.
    // Benchmark performance vs row-based writer.

    if(table.has_child("values"))
    {
        write_single_table(table, path);
    }
    else
    {
        write_multiple_tables(table, path);
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
