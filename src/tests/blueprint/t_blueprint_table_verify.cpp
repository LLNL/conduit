// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_table_verify.cpp
///
//-----------------------------------------------------------------------------

#include <array>

#include <conduit.hpp>
#include <conduit_blueprint.hpp>

#include <gtest/gtest.h>

using namespace conduit;

static constexpr index_t flat_num_rows = 10;
static constexpr index_t flat_num_cols = 2;
std::array<int, flat_num_rows> flat_col0{0,0,0,0,0,0,0,0,0,0};
std::array<int, flat_num_rows> flat_col1{1,1,1,1,1,1,1,1,1,1};

static constexpr index_t mc_num_rows = 3;
static constexpr index_t mc_num_cols = 2;
std::array<int, 9> mc_col0{0,1,2,0,1,2,0,1,2};
std::array<int, 9> mc_col1{0,0,0,1,1,1,2,2,2};

static constexpr index_t mixed_num_rows = 10;
static constexpr index_t mixed_num_cols = 6;
std::array<int, mixed_num_rows> mixed_col0{0,0,0,0,0,0,0,0,0,0};
std::array<int, mixed_num_rows> mixed_col1{1,1,1,1,1,1,1,1,1,1};

//-----------------------------------------------------------------------------
static void
make_flat_table(Node &table)
{
    table["values/col0"].set_external(DataType::c_int(flat_num_rows), flat_col0.data());
    table["values/col1"].set_external(DataType::c_int(flat_num_rows), flat_col1.data());
}

//-----------------------------------------------------------------------------
static void
make_mc_table(Node &table)
{
    table["values/col0/x"].set_external(DataType::c_int(3, sizeof(int)*0, sizeof(int)*3), mc_col0.data());
    table["values/col0/y"].set_external(DataType::c_int(3, sizeof(int)*1, sizeof(int)*3), mc_col0.data());
    table["values/col0/z"].set_external(DataType::c_int(3, sizeof(int)*2, sizeof(int)*3), mc_col0.data());

    table["values/col1/u"].set_external(DataType::c_int(3, sizeof(int)*0, sizeof(int)*3), mc_col1.data());
    table["values/col1/v"].set_external(DataType::c_int(3, sizeof(int)*1, sizeof(int)*3), mc_col1.data());
    table["values/col1/w"].set_external(DataType::c_int(3, sizeof(int)*2, sizeof(int)*3), mc_col1.data());
}

//-----------------------------------------------------------------------------
static void
make_mixed_table(Node &table)
{
    table["values/scalar0"].set_external(DataType::c_int(mixed_num_rows), mixed_col0.data());
    blueprint::mcarray::examples::xyz("interleaved", mixed_num_rows, table["values/vector0"]);
    blueprint::mcarray::examples::xyz("separate", mixed_num_rows, table["values/vector1"]);
    blueprint::mcarray::examples::xyz("contiguous", mixed_num_rows, table["values/vector2"]);
    blueprint::mcarray::examples::xyz("interleaved_mixed", mixed_num_rows, table["values/vector3"]);
    table["values/scalar1"].set_external(DataType::c_double(mixed_num_rows), mixed_col1.data());
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, flat_data)
{
    Node table;
    make_flat_table(table);

    Node info;
    bool res = blueprint::table::verify(table, info);
    ASSERT_TRUE(res) << info.to_json();

    EXPECT_EQ(flat_num_rows, info["values/rows"].to_index_t());
    EXPECT_EQ(flat_num_cols, info["values/columns"].to_index_t());
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, mc_data)
{
    std::array<int, 9> col0{0,1,2,0,1,2,0,1,2};
    std::array<int, 9> col1{0,0,0,1,1,1,2,2,2};

    Node table;
    make_mc_table(table);

    Node info;
    bool res = blueprint::table::verify(table, info);
    ASSERT_TRUE(res) << info.to_json();

    EXPECT_EQ(mc_num_rows, info["values/rows"].to_index_t());
    EXPECT_EQ(mc_num_cols, info["values/columns"].to_index_t());
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, mixed_data)
{
    Node table;
    make_mixed_table(table);

    Node info;
    bool res = blueprint::table::verify(table, info);
    ASSERT_TRUE(res) << info.to_json();

    EXPECT_EQ(mixed_num_rows, info["values/rows"].to_index_t());
    EXPECT_EQ(mixed_num_cols, info["values/columns"].to_index_t());
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, mixed_data_with_name)
{
    Node table;
    make_mixed_table(table["name"]);
    
    Node info;
    bool res = blueprint::table::verify(table, info);
    ASSERT_TRUE(res) << info.to_json();
    EXPECT_EQ(mixed_num_rows, info["name/values/rows"].to_index_t());
    EXPECT_EQ(mixed_num_cols, info["name/values/columns"].to_index_t());
    EXPECT_EQ(1, info["tables"].to_index_t());
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, many_flat_data)
{
    constexpr index_t num_tables = 2;

    Node tables;
    for(index_t i = 0; i < num_tables; i++)
    {
        Node &table = tables.append();
        make_flat_table(table);

        Node info;
        bool res = blueprint::table::verify(table, info);
        ASSERT_TRUE(res) << info.to_json();

        EXPECT_EQ(flat_num_rows, info["values/rows"].to_index_t());
        EXPECT_EQ(flat_num_cols, info["values/columns"].to_index_t());
    }

    Node info;
    bool res = blueprint::table::verify(tables, info);
    ASSERT_TRUE(res) << info.to_json();

    EXPECT_EQ(num_tables, info["tables"].to_index_t());
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, many_mc_data)
{
    constexpr index_t num_tables = 2;

    Node tables;
    for(index_t i = 0; i < num_tables; i++)
    {
        Node &table = tables.append();
        make_mc_table(table);

        Node info;
        bool res = blueprint::table::verify(table, info);
        ASSERT_TRUE(res) << info.to_json();

        EXPECT_EQ(mc_num_rows, info["values/rows"].to_index_t());
        EXPECT_EQ(mc_num_cols, info["values/columns"].to_index_t());
    }

    Node info;
    bool res = blueprint::table::verify(tables, info);
    ASSERT_TRUE(res) << info.to_json();

    EXPECT_EQ(num_tables, info["tables"].to_index_t());
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, many_mixed_data)
{
    constexpr index_t num_tables = 2;

    Node tables;
    for(index_t i = 0; i < num_tables; i++)
    {
        Node &table = tables.append();
        make_mixed_table(table);

        Node info;
        bool res = blueprint::table::verify(table, info);
        ASSERT_TRUE(res) << info.to_json();

        EXPECT_EQ(mixed_num_rows, info["values/rows"].to_index_t());
        EXPECT_EQ(mixed_num_cols, info["values/columns"].to_index_t());
    }

    Node info;
    bool res = blueprint::table::verify(tables, info);
    ASSERT_TRUE(res) << info.to_json();

    EXPECT_EQ(num_tables, info["tables"].to_index_t());
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, many_mixed_data_with_names)
{
    constexpr index_t num_tables = 2;
    constexpr index_t num_rows = 10;
    constexpr index_t num_cols = 6;
    std::array<int,    num_rows> col0{0,0,0,0,0,0,0,0,0,0};
    std::array<double, num_rows> col1{1.,1.,1.,1.,1.,1.,1.,1.,1.,1.};

    Node tables;
    for(index_t i = 0; i < num_tables; i++)
    {
        Node &table = tables["table_" + std::to_string(i)];
        make_mixed_table(table);

        Node info;
        bool res = blueprint::table::verify(table, info);
        ASSERT_TRUE(res) << info.to_json();

        EXPECT_EQ(mixed_num_rows, info["values/rows"].to_index_t());
        EXPECT_EQ(mixed_num_cols, info["values/columns"].to_index_t());
    }

    Node info;
    bool res = blueprint::table::verify(tables, info);
    ASSERT_TRUE(res) << info.to_json();

    EXPECT_EQ(num_tables, info["tables"].to_index_t());
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, empty_node)
{
    Node n, info;
    bool res = blueprint::table::verify(n, info);
    EXPECT_FALSE(res) << info.to_json();
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, invalid_node)
{
    Node not_a_table, info;
    
    // Does not have values
    not_a_table["name"] = "not_a_table";
    bool res = blueprint::table::verify(not_a_table, info);
    EXPECT_FALSE(res) << info.to_json();

    // Has invalid values
    not_a_table["values"] = "Invalid values!";
    res = blueprint::table::verify(not_a_table, info);
    EXPECT_FALSE(res) << info.to_json();

    not_a_table.reset();

    // Contains valid table named "name"
    make_mixed_table(not_a_table["name"]);
    not_a_table["name/random_field"] = "This is a table!";
    res = blueprint::table::verify(not_a_table, info);
    ASSERT_TRUE(res) << info.to_json();
    EXPECT_EQ(mixed_num_rows, info["name/values/rows"].to_index_t());
    EXPECT_EQ(mixed_num_cols, info["name/values/columns"].to_index_t());
    
    // Introduces invalid table at top level
    not_a_table["invalid"] = "Invalid table!";
    res = blueprint::table::verify(not_a_table, info);
    EXPECT_FALSE(res) << info.to_json();

    // Introduces table with invalid values
    not_a_table["invalid"].reset();
    not_a_table["invalid/values"] = "Invalid values!";
    res = blueprint::table::verify(not_a_table, info);
    EXPECT_FALSE(res) << info.to_json();
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, different_lengths)
{
    Node table;
    make_mixed_table(table);
    table["values/invalid"].set(DataType::c_int(mixed_num_rows+1));
    
    Node info;
    bool res = blueprint::table::verify(table, info);
    EXPECT_FALSE(res) << info.to_json();
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_verify, empty_column)
{
    Node table;
    make_mixed_table(table);
    table["values/scalar1"].reset();
    
    Node info;
    bool res = blueprint::table::verify(table, info);
    EXPECT_FALSE(res) << info.to_json();
}
