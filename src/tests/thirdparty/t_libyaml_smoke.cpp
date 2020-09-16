// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_libyaml_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "yaml.h"
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(libyaml_smoke, basic_event_use)
{
    yaml_parser_t parser;
    yaml_event_t  event;

    // Initialize parser
    EXPECT_TRUE(yaml_parser_initialize(&parser));

    const char yaml_str[] = " hello : world\n line: 2\n";

    // set input
    yaml_parser_set_input_string(&parser,
                                 (const unsigned char*)yaml_str,
                                 strlen(yaml_str));

    bool ok = true;
    bool found_mapping = false;
    bool found_scalar   = false;
    
    bool found_hello = false;
    bool found_world = false;

    while(ok)
    {
        EXPECT_TRUE(yaml_parser_parse(&parser, &event));

    
        switch(event.type)
        { 
            case YAML_MAPPING_START_EVENT:
            {
                found_mapping = true;
                break;
            }
            case YAML_SCALAR_EVENT: 
            {
                found_scalar = true;
                std::string val((const char*)event.data.scalar.value);
                std::cout << val << std::endl;
                if(val == "hello")
                {
                    found_hello = true;
                }
                else if( val == "world")
                {
                    found_world = true;
                }
                break;
            }
            default:
                break;
        }

        ok = event.type != YAML_STREAM_END_EVENT;
        yaml_event_delete(&event);
    }

    EXPECT_TRUE(found_mapping);
    EXPECT_TRUE(found_scalar);
    EXPECT_TRUE(found_hello);
    EXPECT_TRUE(found_world);


    // cleanup parser
    yaml_parser_delete(&parser);
}

//-----------------------------------------------------------------------------
TEST(libyaml_smoke, basic_document_use)
{
    yaml_parser_t parser;
    yaml_document_t document;

    const char yaml_str[] = " hello : world\n line: 2\n";

    // Initialize parser
    EXPECT_TRUE(yaml_parser_initialize(&parser));

    // set input
    yaml_parser_set_input_string(&parser,
                                 (const unsigned char*)yaml_str,
                                 strlen(yaml_str));

    // construct document
    EXPECT_TRUE(yaml_parser_load(&parser, &document));
    yaml_node_t *node = yaml_document_get_root_node(&document);

    EXPECT_TRUE(node != NULL);

    // for this example, we we know the root is a mapping node
    EXPECT_TRUE( node->type == YAML_MAPPING_NODE );

    yaml_node_pair_t *pair = node->data.mapping.pairs.start;
    EXPECT_TRUE(pair != NULL);

    yaml_node_t *key = yaml_document_get_node(&document, pair->key);
    yaml_node_t *val = yaml_document_get_node(&document, pair->value);

    EXPECT_TRUE(key != NULL);
    EXPECT_TRUE(val != NULL);
    
    std::string key_name((const char*)key->data.scalar.value);
    std::string val_str((const char*)val->data.scalar.value);

    std::cout << key_name << ": " << val_str << std::endl;

    EXPECT_EQ(key_name,"hello");
    EXPECT_EQ(val_str,"world");

    // cleanup doc
    yaml_document_delete(&document);

    // cleanup parser
    yaml_parser_delete(&parser);
}

