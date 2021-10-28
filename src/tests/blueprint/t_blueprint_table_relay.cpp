// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_table_relay.cpp
///
//-----------------------------------------------------------------------------

#include <cstdlib>

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
    blueprint::mesh::examples::spiral(4, mesh);

    Node table, opts;
    blueprint::mesh::flatten(mesh, opts, table);

    relay::io::save(table, filename);

    ASSERT_TRUE(utils::is_directory(filename));

    Node read_table;
    relay::io::load(filename, read_table);

    table::compare_to_baseline(read_table, table, false);
}
