// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_node.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"

#include <stdio.h>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(c_conduit_node, simple)
{
    conduit_node *n = conduit_node_create();
    
    EXPECT_TRUE(conduit_node_is_root(n));
    
    conduit_node_set_int(n,10);
    
    EXPECT_EQ(conduit_node_as_int(n),10);
    int *int_ptr = conduit_node_as_int_ptr(n);
    EXPECT_EQ(int_ptr[0],10);
    
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, simple_hier)
{
    int     a_val  = 10;
    int     b_val  = 20;
    double  c_val  = 30.0;
    
    conduit_node *n = conduit_node_create();
    
    conduit_node *a = conduit_node_fetch(n,"a");
    conduit_node *b = conduit_node_fetch(n,"b");
    conduit_node *c = conduit_node_fetch(n,"c");

    EXPECT_TRUE(conduit_node_is_root(n));
    
    EXPECT_FALSE(conduit_node_is_root(a));
    EXPECT_FALSE(conduit_node_is_root(b));
    EXPECT_FALSE(conduit_node_is_root(c));
    
    conduit_node_set_int(a,a_val);
    conduit_node_set_int(b,b_val);
    conduit_node_set_double(c,c_val);

    
    EXPECT_EQ(conduit_node_as_int(a),a_val);
    EXPECT_EQ(conduit_node_as_int(b),b_val);
    EXPECT_EQ(conduit_node_as_double(c),c_val);
        
    int    *a_ptr = conduit_node_as_int_ptr(a);
    int    *b_ptr = conduit_node_as_int_ptr(b);
    double *c_ptr = conduit_node_as_double_ptr(c);
    
    EXPECT_EQ(a_ptr[0],a_val);
    EXPECT_EQ(b_ptr[0],b_val);
    EXPECT_EQ(c_ptr[0],c_val);
    
    conduit_node_print(n);
    
    /// these are no-ops
    conduit_node_destroy(a);
    conduit_node_destroy(b);
    conduit_node_destroy(c);

    conduit_node_print(n);
    
    /// this actually deletes the node
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_diff)
{
    int     a_val  = 10;
    int     b_val  = 20;
    double  c_val  = 30.0;
    
    conduit_node *ninfo = conduit_node_create();
    
    conduit_node *n1 = conduit_node_create();
    
    conduit_node *a1 = conduit_node_fetch(n1,"a");
    conduit_node *b1 = conduit_node_fetch(n1,"b");
    conduit_node *c1 = conduit_node_fetch(n1,"c");

    conduit_node_set_int(a1,a_val);
    conduit_node_set_int(b1,b_val);
    conduit_node_set_double(c1,c_val);

    conduit_node *n2 = conduit_node_create();
    
    conduit_node *a2 = conduit_node_fetch(n2,"a");
    conduit_node *b2 = conduit_node_fetch(n2,"b");
    conduit_node *c2 = conduit_node_fetch(n2,"c");

    conduit_node_set_int(a2,a_val);
    conduit_node_set_int(b2,b_val);
    conduit_node_set_double(c2,c_val);
    
    // no diff
    EXPECT_FALSE(conduit_node_diff(n1,n2,ninfo,1e-12));

    conduit_node_set_double(c2,32.0);

    // there is a diff
    EXPECT_TRUE(conduit_node_diff(n1,n2,ninfo,1e-12));
    
    conduit_node_set_double(c2,30.0);
    
    conduit_node *d2 = conduit_node_fetch(n2,"d");
    conduit_node_set_double(d2,42.0);

    // there is diff
    EXPECT_TRUE(conduit_node_diff(n1,n2,ninfo,1e-12));
    // no diff compat
    EXPECT_FALSE(conduit_node_diff_compatible(n1,n2,ninfo,1e-12));
    
    conduit_node_set_double(c2,32.0);
    
    // there is a diff compat
    EXPECT_TRUE(conduit_node_diff_compatible(n1,n2,ninfo,1e-12));
    
    conduit_node_destroy(n1);
    conduit_node_destroy(n2);
    conduit_node_destroy(ninfo);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_update)
{
    int     a_val  = 10;
    int     b_val  = 20;
    double  c_val  = 30.0;
    
    conduit_node *ninfo = conduit_node_create();
    
    conduit_node *n1 = conduit_node_create();
    
    conduit_node *a1 = conduit_node_fetch(n1,"a");
    conduit_node *b1 = conduit_node_fetch(n1,"b");
    conduit_node *c1 = conduit_node_fetch(n1,"c");

    conduit_node_set_int(a1,a_val);
    conduit_node_set_int(b1,b_val);
    conduit_node_set_double(c1,c_val);

    conduit_node *n2 = conduit_node_create();
    
    conduit_node *a2 = conduit_node_fetch(n2,"a");
    conduit_node *b2 = conduit_node_fetch(n2,"b");
    conduit_node *c2 = conduit_node_fetch(n2,"c");

    conduit_node_set_int(a2,a_val);
    conduit_node_set_int(b2,b_val);
    conduit_node_set_double(c2,42.0);
    
    conduit_node_update(n1,n2);

    double *c1_ptr = conduit_node_as_double_ptr(c1);

    EXPECT_EQ(c1_ptr[0],42.0);

    conduit_node_rename_child(n2,"c","d");
    
    conduit_node *d2 = conduit_node_fetch(n2,"d");

    double *d2_ptr = conduit_node_as_double_ptr(d2);

    EXPECT_EQ(d2_ptr[0],42.0);

    conduit_node_update_compatible(n1,n2);
    EXPECT_FALSE(conduit_node_has_path(n1,"d"));

    conduit_node_update_external(n1,n2);

    conduit_node *d1 = conduit_node_fetch(n1,"d");
    double *d1_ptr = conduit_node_as_double_ptr(d1);

    d2_ptr[0] = 52.0;

    // these should be the same address
    EXPECT_EQ(d1_ptr,d2_ptr);
    EXPECT_EQ(d1_ptr[0],52.0);

    conduit_node_destroy(n1);
    conduit_node_destroy(n2);
    conduit_node_destroy(ninfo);

}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_compatible)
{
    int     a_val  = 10;
    double  b_val  = 30.0;

    conduit_node *n1 = conduit_node_create();
    conduit_node *n2 = conduit_node_create();

    conduit_node *a1 = conduit_node_fetch(n1,"a");
    conduit_node *a2 = conduit_node_fetch(n2,"a");
    
    conduit_node_set_int(a1,a_val);
    conduit_node_set_double(a2,b_val);

    EXPECT_FALSE(conduit_node_compatible(n1,n2));

    conduit_node_rename_child(n2,"a","b");
    // it will be compat after we rename a to b
    EXPECT_TRUE(conduit_node_compatible(n1,n2));

    conduit_node_set_path_int(n2,"a",a_val);
    // still true!
    EXPECT_TRUE(conduit_node_compatible(n1,n2));

    conduit_node_print(n1);
    conduit_node_print(n2);

    conduit_node_destroy(n1);
    conduit_node_destroy(n2);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_compact)
{
    conduit_int32    a_val  = 10;
    conduit_int32    b_val  = 20;
    conduit_float64  c_val  = 30.0;

    conduit_node *n = conduit_node_create();

    conduit_node *a = conduit_node_fetch(n,"a");
    conduit_node *b = conduit_node_fetch(n,"b");
    conduit_node *c = conduit_node_fetch(n,"c");
    
    conduit_node_set_int32(a,a_val);
    conduit_node_set_int32(b,b_val);
    conduit_node_set_float64(c,c_val);

    EXPECT_EQ(conduit_node_total_bytes_allocated(n),16);

    EXPECT_FALSE(conduit_node_is_contiguous(n));

    conduit_node *n_c = conduit_node_create();

    conduit_node_compact_to(n, n_c);

    EXPECT_TRUE(conduit_node_is_contiguous(n_c));

    EXPECT_EQ(conduit_node_total_bytes_compact(n_c),16);
    EXPECT_EQ(conduit_node_total_strided_bytes(n_c),16);
    EXPECT_EQ(conduit_node_total_bytes_allocated(n_c),16);

    conduit_node_destroy(n);
    conduit_node_destroy(n_c);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_name_and_path)
{
    conduit_node *n = conduit_node_create();

    conduit_node *a = conduit_node_fetch(n,"a");
    conduit_node *b = conduit_node_fetch(n,"b");
    conduit_node *c = conduit_node_fetch(n,"c");

    conduit_node *e = conduit_node_fetch(n,"d/e");

    char *a_name = conduit_node_name(a);
    char *a_path = conduit_node_path(a);

    char *e_name = conduit_node_name(e);
    char *e_path = conduit_node_path(e);

    EXPECT_EQ(std::string(a_name),"a");
    EXPECT_EQ(std::string(a_path),"a");

    EXPECT_EQ(std::string(e_name),"e");
    EXPECT_EQ(std::string(e_path),"d/e");

    free(a_name);
    free(a_path);

    free(e_name);
    free(e_path);

    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_remove)
{
    int     a_val  = 10;
    int     b_val  = 20;
    double  c_val  = 30.0;
    
    conduit_node *n = conduit_node_create();
    
    conduit_node *a = conduit_node_fetch(n,"a");
    conduit_node *b = conduit_node_fetch(n,"b");
    conduit_node *c = conduit_node_fetch(n,"c");

    conduit_node_set_int(a,a_val);
    conduit_node_set_int(b,b_val);
    conduit_node_set_double(c,c_val);
    
    // remove child by index
    EXPECT_TRUE(conduit_node_has_path(n,"a"));
    conduit_node_remove_child(n,0);
    EXPECT_FALSE(conduit_node_has_path(n,"a"));
    EXPECT_TRUE(conduit_node_has_path(n,"b"));
    EXPECT_TRUE(conduit_node_has_path(n,"c"));

    // remove child by path
    conduit_node_remove_path(n,"b");
    EXPECT_FALSE(conduit_node_has_path(n,"a"));
    EXPECT_FALSE(conduit_node_has_path(n,"b"));
    EXPECT_TRUE(conduit_node_has_path(n,"c"));

    // rename
    conduit_node_rename_child(n,"c","a");
    EXPECT_FALSE(conduit_node_has_path(n,"c"));
    EXPECT_TRUE(conduit_node_has_path(n,"a"));

    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_parse)
{
    conduit_node *n = conduit_node_create();

    conduit_node_parse(n,"{\"a\": 42.0}","json");
    conduit_node *a = conduit_node_fetch(n,"a");
    EXPECT_EQ(conduit_node_as_double(a),42.0);

    conduit_node_parse(n,"a: 42.0","yaml");
    a = conduit_node_fetch(n,"a");
    EXPECT_EQ(conduit_node_as_double(a),42.0);

    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_save_load)
{
    conduit_node *n1 = conduit_node_create();
    conduit_node *n2 = conduit_node_create();

    conduit_node *a1 = conduit_node_fetch(n1,"a");
    conduit_node_set_double(a1,42.0);

    conduit_node_save(n1,"tout_c_node_save.json","json");
    conduit_node_load(n2,"tout_c_node_save.json","json");
    conduit_node *a2 = conduit_node_fetch(n2,"a");
    EXPECT_EQ(conduit_node_as_double(a2),42.0);

    conduit_node_save(n1,"tout_c_node_save.yaml","yaml");
    conduit_node_load(n2,"tout_c_node_save.yaml","yaml");
    a2 = conduit_node_fetch(n2,"a");
    EXPECT_EQ(conduit_node_as_double(a2),42.0);

    conduit_node_destroy(n1);
    conduit_node_destroy(n2);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_to_json)
{
    conduit_node *n = conduit_node_create();
    conduit_node_parse(n,"{\"a\": 42.0}","json");

    char* res = conduit_node_to_json(n);
    std::string res_str = res;
    EXPECT_EQ(res_str, "\n{\n  \"a\": 42.0\n}");

    free(res);
    
    // now with opts
    
    conduit_node *nopts = conduit_node_create();
    conduit_node_set_path_int(nopts,"depth",1);
    conduit_node_set_path_char8_str(nopts,"pad"," PAD ");
    conduit_node_set_path_char8_str(nopts,"eoe"," EOE\n ");

    conduit_node_print(nopts);

    res = conduit_node_to_json_with_options(n,nopts);
    res_str  =res;

    std::cout << res_str << std::endl;

    std::string texpect = R"ST( EOE
  PAD  PAD { EOE
  PAD  PAD  PAD  PAD "a": 42.0 EOE
  PAD  PAD })ST";

    EXPECT_EQ(res_str, texpect);

    free(res);
    conduit_node_destroy(n);
    conduit_node_destroy(nopts);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_to_yaml)
{
    conduit_node *n = conduit_node_create();
    conduit_node_parse(n,"a: 42.0\n","yaml");

    char* res = conduit_node_to_yaml(n);
    std::string res_str = res;
    std::cout << res_str << std::endl;
    EXPECT_EQ(res_str, "\na: 42.0\n");

    free(res);
 
    // now with opts
 
    conduit_node *nopts = conduit_node_create();
    conduit_node_set_path_int(nopts,"depth",1);
    conduit_node_set_path_char8_str(nopts,"pad"," PAD ");
    conduit_node_set_path_char8_str(nopts,"eoe"," EOE\n ");

    conduit_node_print(nopts);

    res = conduit_node_to_yaml_with_options(n,nopts);
    res_str = res;

    std::cout << res_str << std::endl;

    std::string texpect = R"ST( EOE
  PAD  PAD a: 1 EOE
 )ST";
 
    free(res);
    conduit_node_destroy(n);
    conduit_node_destroy(nopts);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_to_string)
{
    conduit_node *n = conduit_node_create();
    conduit_node_parse(n,"a: 42.0\n","yaml");

    char* res = conduit_node_to_string(n);
    std::string res_str = res;
    std::cout << res_str << std::endl;
    EXPECT_EQ(res_str, "\na: 42.0\n");

    free(res);
 
    // now with opts
 
    conduit_node *nopts = conduit_node_create();
    conduit_node_set_path_char8_str(nopts,"protocol","yaml");
    conduit_node_set_path_int(nopts,"depth",1);
    conduit_node_set_path_char8_str(nopts,"pad"," PAD ");
    conduit_node_set_path_char8_str(nopts,"eoe"," EOE\n ");

    res = conduit_node_to_string_with_options(n,nopts);
    res_str = res;

    std::cout << res_str << std::endl;

    std::string texpect = R"ST( EOE
  PAD  PAD a: 1 EOE
 )ST";
 
    free(res);
    conduit_node_destroy(n);
    conduit_node_destroy(nopts);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_to_summary_string)
{
    conduit_node *n = conduit_node_create();
    conduit_node_parse(n,"a: 42.0\n","yaml");

    char* res = conduit_node_to_summary_string(n);
    std::string res_str = res;
    std::cout << res_str << std::endl;
    EXPECT_EQ(res_str, "\na: 42.0\n");

    free(res);
 
    // now with opts
 
    conduit_node *nopts = conduit_node_create();
    conduit_node_set_path_char8_str(nopts,"protocol","yaml");
    conduit_node_set_path_int(nopts,"depth",1);
    conduit_node_set_path_char8_str(nopts,"pad"," PAD ");
    conduit_node_set_path_char8_str(nopts,"eoe"," EOE\n ");

    res = conduit_node_to_summary_string_with_options(n,nopts);
    res_str = res;

    std::cout << res_str << std::endl;

    std::string texpect = R"ST( EOE
  PAD  PAD a: 1 EOE
 )ST";
 
    free(res);
    conduit_node_destroy(n);
    conduit_node_destroy(nopts);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_child_with_embedded_slashes)
{
    conduit_node *n = conduit_node_create();

    conduit_node *n_1 = conduit_node_fetch(n,"normal/path");
    conduit_node_set_int(n_1,10);

    conduit_node *n_2 = conduit_node_add_child(n,"child_with_/_inside");
    conduit_node_set_int(n_2,42);

    EXPECT_TRUE(conduit_node_has_path(n,"normal/path"));
    EXPECT_FALSE(conduit_node_has_child(n,"normal/path"));
    EXPECT_FALSE(conduit_node_has_path(n,"child_with_/_inside"));
    EXPECT_TRUE(conduit_node_has_child(n,"child_with_/_inside"));

    EXPECT_EQ(conduit_node_number_of_children(n),2);
    
    conduit_node *n_2_test = conduit_node_child_by_name(n,"child_with_/_inside");

    EXPECT_EQ(n_2, n_2_test);
    
    EXPECT_EQ(conduit_node_as_int(n_1),10);
    EXPECT_EQ(conduit_node_as_int(n_2_test),42);

    conduit_node_remove_child_by_name(n,"child_with_/_inside");
    EXPECT_EQ(conduit_node_number_of_children(n),1);
    EXPECT_TRUE(conduit_node_has_path(n,"normal/path"));
    EXPECT_FALSE(conduit_node_has_child(n,"child_with_/_inside"));


    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_fetch_existing)
{
    conduit_node *n = conduit_node_create();

    conduit_node *n_1 = conduit_node_fetch(n,"normal/path");
    conduit_node_set_int(n_1,10);

    conduit_node *n_2 = conduit_node_fetch_existing(n,"normal/path");

    EXPECT_EQ(n_1,n_2);
    
    EXPECT_EQ(conduit_node_as_int(n_2),10);
    

    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_reset)
{
    conduit_node *n = conduit_node_create();

    conduit_node *n_1 = conduit_node_fetch(n,"normal/path");
    conduit_node_set_int(n_1,10);

    // check number of children
    EXPECT_EQ(conduit_node_number_of_children(n),1);
    const conduit_datatype *dtype = conduit_node_dtype(n);
    EXPECT_EQ(conduit_datatype_is_object(dtype),1);

    conduit_node_reset(n);

    EXPECT_EQ(conduit_node_number_of_children(n),0);

    // check that dtype is empty
    dtype = conduit_node_dtype(n);
    EXPECT_EQ(conduit_datatype_is_empty(dtype),1);

    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node, c_move_and_swap)
{
    conduit_node *n_a = conduit_node_create();
    conduit_node *n_b = conduit_node_create();

    conduit_node_set_path_int(n_a,"data",10);
    conduit_node_set_path_int(n_b,"data",20);

    conduit_node_swap(n_a,n_b);

    EXPECT_EQ(conduit_node_fetch_path_as_int(n_a,"data"),20);
    EXPECT_EQ(conduit_node_fetch_path_as_int(n_b,"data"),10);

    // now move b into a, b will be reset as a result
    conduit_node_move(n_a,n_b);

    EXPECT_EQ(conduit_node_number_of_children(n_b),0);
    // check that dtype is empty
    const conduit_datatype *dtype = conduit_node_dtype(n_b);
    EXPECT_EQ(conduit_datatype_is_empty(dtype),1);

    EXPECT_EQ(conduit_node_fetch_path_as_int(n_a,"data"),10);
    
    conduit_node_destroy(n_a);
    conduit_node_destroy(n_b);
}



