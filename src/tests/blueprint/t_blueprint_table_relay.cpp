// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_table_relay.cpp
///
//-----------------------------------------------------------------------------
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include <conduit.hpp>
#include <conduit_blueprint_mesh.hpp>
#include <conduit_blueprint_mesh_examples.hpp>
#include <conduit_blueprint_table_examples.hpp>
#include <conduit_relay_io.hpp>
#include <conduit_relay_io_csv.hpp>

#include "blueprint_test_helpers.hpp"

#include "gtest/gtest.h"

using namespace conduit;

// Returns 0 if okay, error code otherwise
int cleanup_dir(const std::string &dirname)
{
    if(utils::is_directory(dirname))
    {
        std::vector<std::string> fnames;
        utils::list_directory_contents(dirname, fnames);
        for(const auto &name : fnames)
        {
            if (!utils::is_file(name)) return 1;
            if (!utils::remove_file(name)) return 2;
        }
        if (!utils::remove_directory(dirname)) return 3;
    }
    return 0;
}

TEST(t_blueprint_table_relay, read_write_csv)
{
    const std::string filename = "t_blueprint_table_relay_read_write_csv.csv";

    // Remove the file if it exists
    if(utils::is_file(filename))
    {
        ASSERT_TRUE(utils::remove_file(filename));
    }

    // Load example
    Node table;
    blueprint::table::examples::basic(5, 4, 3, table);

    // Write then read
    Node opts;
    relay::io::write_csv(table, filename, opts);

    Node read_table;
    relay::io::read_csv(filename, opts, read_table);

    // Ensure the original table and the one that was read from the file match
    table::compare_to_baseline(read_table, table);
}

TEST(t_blueprint_table_relay, read_write_relay)
{
    const std::string filename = "t_blueprint_table_relay_read_write_relay.csv";

    if(utils::is_file(filename))
    {
        ASSERT_TRUE(utils::remove_file(filename));
    }

    Node table;
    blueprint::table::examples::basic(4, 3, 2, table);

    Node opts;
    relay::io::save(table, filename);

    Node read_table;
    relay::io::load(filename, read_table);

    table::compare_to_baseline(read_table, table);
}

TEST(t_blueprint_table_relay, read_write_multi_table)
{
    const std::string filename = "t_blueprint_table_relay_read_write_multi_table.csv";

    ASSERT_EQ(0, cleanup_dir(filename));

    Node mesh;
    blueprint::mesh::examples::basic("uniform", 5, 4, 3, mesh);

    // Need to create a vertex field, don't want to compare float coord locations.
    const int npts = 5 * 4 * 3;
    mesh["fields/vertex_field/topology"].set(mesh["topologies"][0].name());
    mesh["fields/vertex_field/association"].set("vertex");
    mesh["fields/vertex_field/values"].set_dtype(DataType::c_int(npts));
    int *vf = static_cast<int*>(mesh["fields/vertex_field/values"].element_ptr(0));
    for(int i = 0; i < npts; i++)
    {
        *vf++ = npts - 1 - i;
    }

    Node table, opts;
    opts["add_vertex_locations"] = 0;
    opts["add_element_centers"] = 0;
    blueprint::mesh::flatten(mesh, opts, table);

    relay::io::save(table, filename);

    ASSERT_TRUE(utils::is_directory(filename));
    {
        std::vector<std::string> files;
        utils::list_directory_contents(filename, files);
        // Remove the directory
        for(auto i = 0u; i < files.size(); i++)
        {
            std::cout << files[i] << std::endl;
            files[i] = files[i].substr(files[i].rfind(utils::file_path_separator()) + 1);
        }
        EXPECT_EQ(2, files.size());
        auto itr = std::find(files.begin(), files.end(), "element_data.csv");
        EXPECT_NE(files.end(), itr);
        itr = std::find(files.begin(), files.end(), "vertex_data.csv");
        EXPECT_NE(files.end(), itr);
    }

    Node read_table;
    relay::io::load(filename, opts, read_table);

    // std::cout << read_table.to_json() << std::endl;
    table::compare_to_baseline(read_table, table, false);
}

TEST(t_blueprint_table_relay, read_write_multi_table_list)
{
    const std::string filename = "t_blueprint_table_relay_read_write_multi_table_list.csv";
    ASSERT_EQ(0, cleanup_dir(filename));

    Node table;
    blueprint::table::examples::basic(5, 4, 3, table.append());
    {
        // Add another column
        const index_t nrows = table[0]["values/points/x"].dtype().number_of_elements();
        Node &n = table[0]["values/new_column"];
        n.set_dtype(DataType::index_t(nrows));
        index_t *data = static_cast<index_t*>(n.element_ptr(0));
        for(index_t i = 0; i < nrows; i++)
        {
            data[i] = nrows - 1 - i;
        }
    }

    blueprint::table::examples::basic(6, 3, 2, table.append());

    relay::io::save(table, filename);

    Node read_table;
    relay::io::load(filename, read_table);

    table::compare_to_baseline(read_table, table);
}

TEST(t_blueprint_table_relay, read_write_table_with_list_values)
{
    const std::string filename =
        "t_blueprint_table_relay_read_write_table_with_list_values.csv";
    ASSERT_EQ(0, cleanup_dir(filename));

    Node basic_table;
    blueprint::table::examples::basic(5, 4, 3, basic_table);

    // Remove the names
    Node table;
    Node &table_values = table["values"];
    const index_t ncolumns = basic_table["values"].number_of_children();
    for(index_t i = 0; i < ncolumns; i++)
    {
        Node &n = table_values.append();
        n.set_external(basic_table["values"][i]);
    }

    relay::io::save(table, filename);

    Node read_table;
    relay::io::load(filename, read_table);

    table::compare_to_baseline(read_table, table);
}
