// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

#include "conduit_node.hpp"
#include "tv_data_display.h"

#include <string>
#include <sstream>

static const char * empty_Node_TV_string = "(empty Node)";

const std::string dtype_to_TV_string ( const conduit::DataType dtype );
const std::string dtype_to_TV_string ( const conduit::DataType dtype, const char *hint );
const std::string index_to_TV_string ( int idx );
const std::string dtype_subscript ( const conduit::DataType dtype );

const std::string dtype_to_TV_string ( const conduit::DataType dtype )
{
   // We'll see if this works.
   const char *hint = dtype.name().c_str();

   return dtype_to_TV_string(dtype, hint);
}

const std::string dtype_subscript ( const conduit::DataType dtype )
{
   std::stringstream ss;

   if (dtype.number_of_elements() > 1)
   {
      ss << "[" << dtype.number_of_elements() << "]";
   }

   return ss.str();
}

const std::string dtype_to_TV_string ( const conduit::DataType dtype, const char *hint )
{
   std::stringstream ss;

   ss << hint;
   if (dtype.number_of_elements() > 1)
   {
      ss << index_to_TV_string(dtype.number_of_elements());
   }

   std::cout << "dtype_to_TV_string: " << ss.str() << std::endl;

   return ss.str();
}

const std::string index_to_TV_string ( int idx )
{
   std::stringstream ss;

   ss << "[" << idx << "]";

   return ss.str();
}

int TV_ttf_display_type ( const conduit::Node *n )
{
   const std::string type_name = dtype_to_TV_string(n->dtype());

   switch(n->dtype().id()) {
   case conduit::DataType::EMPTY_ID:
      TV_ttf_add_row ("data", TV_ttf_type_ascii_string, empty_Node_TV_string);
      break;
   case conduit::DataType::OBJECT_ID:
   {
      const std::vector<std::string> & child_names = n->child_names();
      for (const std::string & name : child_names) {
         const conduit::Node * ptr = n->fetch_ptr(name);
         TV_ttf_add_row (name.c_str(), "conduit::Node *", &ptr);
         std::cout << "Added child " << name << " with pointer " << ptr << std::endl;
      }
      break;
   }
   case conduit::DataType::LIST_ID:
   {
      int number_of_children = n->number_of_children();
      for (int cidx = 0; cidx < number_of_children; ++cidx) {
         const conduit::Node * ptr = n->child_ptr(cidx);
         TV_ttf_add_row (index_to_TV_string(cidx).c_str(), "conduit::Node *", &ptr);
      }
      break;
   }
   case conduit::DataType::INT8_ID:
   {
      TV_ttf_add_row ("data", dtype_to_TV_string(n->dtype(), "conduit::int8").c_str(), n->data_ptr());
      break;
   }
   case conduit::DataType::INT16_ID:
   {
      TV_ttf_add_row ("data", dtype_to_TV_string(n->dtype(), "conduit::int16").c_str(), n->data_ptr());
      break;
   }
   case conduit::DataType::INT32_ID:
   {
      TV_ttf_add_row ("data", dtype_to_TV_string(n->dtype(), "conduit::int32").c_str(), n->data_ptr());
      break;
   }
   case conduit::DataType::INT64_ID:
   {
      TV_ttf_add_row ("data", dtype_to_TV_string(n->dtype(), "conduit::int64").c_str(), n->data_ptr());
      break;
   }
   case conduit::DataType::UINT8_ID:
   {
      TV_ttf_add_row ("data", dtype_to_TV_string(n->dtype(), "conduit::uint8").c_str(), n->data_ptr());
      break;
   }
   case conduit::DataType::UINT16_ID:
   {
      TV_ttf_add_row ("data", dtype_to_TV_string(n->dtype(), "conduit::uint16").c_str(), n->data_ptr());
      break;
   }
   case conduit::DataType::UINT32_ID:
   {
      TV_ttf_add_row ("data", dtype_to_TV_string(n->dtype(), "conduit::uint32").c_str(), n->data_ptr());
      break;
   }
   case conduit::DataType::UINT64_ID:
   {
      TV_ttf_add_row ("data", dtype_to_TV_string(n->dtype(), "conduit::uint64").c_str(), n->data_ptr());
      break;
   }
   case conduit::DataType::FLOAT32_ID:
   {
      TV_ttf_add_row ("data", dtype_to_TV_string(n->dtype(), "conduit::float32").c_str(), n->data_ptr());
      break;
   }
   case conduit::DataType::FLOAT64_ID:
   {
      TV_ttf_add_row ("data", dtype_to_TV_string(n->dtype(), "conduit::float64").c_str(), n->data_ptr());
      break;
   }
   case conduit::DataType::CHAR8_STR_ID:
      TV_ttf_add_row ("data", TV_ttf_type_ascii_string, n->as_char8_str());
      break;
   }

   return TV_ttf_format_ok;
}

