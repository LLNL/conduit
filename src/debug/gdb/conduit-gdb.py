# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

import gdb
import gdb.printing
import itertools


class ConduitTypes:
   """Represent Conduit TypeIDs."""
   def __init__(self):
      self.EMPTY_ID     = gdb.parse_and_eval("(int)conduit::DataType::EMPTY_ID");
      self.OBJECT_ID    = gdb.parse_and_eval("(int)conduit::DataType::OBJECT_ID");
      self.LIST_ID      = gdb.parse_and_eval("(int)conduit::DataType::LIST_ID");
      self.INT8_ID      = gdb.parse_and_eval("(int)conduit::DataType::INT8_ID");
      self.INT16_ID     = gdb.parse_and_eval("(int)conduit::DataType::INT16_ID");
      self.INT32_ID     = gdb.parse_and_eval("(int)conduit::DataType::INT32_ID");
      self.INT64_ID     = gdb.parse_and_eval("(int)conduit::DataType::INT64_ID");
      self.UINT8_ID     = gdb.parse_and_eval("(int)conduit::DataType::UINT8_ID");
      self.UINT16_ID    = gdb.parse_and_eval("(int)conduit::DataType::UINT16_ID");
      self.UINT32_ID    = gdb.parse_and_eval("(int)conduit::DataType::UINT32_ID");
      self.UINT64_ID    = gdb.parse_and_eval("(int)conduit::DataType::UINT64_ID");
      self.FLOAT32_ID   = gdb.parse_and_eval("(int)conduit::DataType::FLOAT32_ID");
      self.FLOAT64_ID   = gdb.parse_and_eval("(int)conduit::DataType::FLOAT64_ID");
      self.CHAR8_STR_ID = gdb.parse_and_eval("(int)conduit::DataType::CHAR8_STR_ID");

   def type_string(self, val):
      dtype_id = self.node_type(val)
      tp = ""
      if dtype_id == self.INT8_ID:
         tp = "char"
      elif dtype_id == self.INT16_ID:
         tp = "short"
      elif dtype_id == self.INT32_ID:
         tp = "int"
      elif dtype_id == self.INT64_ID:
         tp = "long"
      elif dtype_id == self.UINT8_ID:
         tp = "unsigned char"
      elif dtype_id == self.UINT16_ID:
         tp = "unsigned short"
      elif dtype_id == self.UINT32_ID:
         tp = "unsigned int"
      elif dtype_id == self.UINT64_ID:
         tp = "unsigned long"
      elif dtype_id == self.FLOAT32_ID:
         tp = "float"
      elif dtype_id == self.FLOAT64_ID:
         tp = "double"

      return tp

   def node_type(self, val):
      dtype_id = val['m_schema'].dereference()
      int_t = gdb.lookup_type('int')
      return dtype_id['m_dtype']['m_id'].cast(int_t)

class EmptyNodePrinter:
   """Print an empty conduit::Node object."""

   def __init__(self):
      pass

   def to_string(self):
      return "empty Node"


class StringNodePrinter:
   """Print a string conduit::Node object."""

   def __init__(self, val, types):
      self.types = types
      self.val = val

   def to_string(self):
      t = gdb.lookup_type('const char *').pointer()
      v = self.val['m_data'].cast(t)
      return v

   def display_hint(self):
      return 'string'

class ArrayNodePrinter:
   """Print an array numeric conduit::Node object."""

   def __init__(self, val, types):
      self.types = types
      self.val = val
      self.num_elts = self.val['m_schema'].dereference()['m_dtype']['m_num_ele']

   def to_string(self):
      return "{{ array length {0} }}".format(self.num_elts)

   def children (self):
      tp = self.types.type_string(self.val)
      t = gdb.lookup_type(tp).pointer()
      v = self.val['m_data'].cast(t)

      for i in range(self.num_elts):
         yield "[{0}]".format(i), v[i]

   def display_hint(self):
      return 'array'


class ScalarNodePrinter:
   """Print a scalar numeric conduit::Node object."""

   def __init__(self, val, types):
      self.types = types
      self.val = val

   def to_string(self):
      tp = self.types.type_string(self.val)
      t = gdb.lookup_type(tp).pointer()
      v = self.val['m_data'].cast(t)
      return v[0]

class TreeNodePrinter:
   """Let subclasses count children."""

   def __init__(self, types):
      self.types = types

   def count_children(self, val):
      dtype_id = self.types.node_type(val)

      tp = ''
      if dtype_id == self.types.OBJECT_ID:
         tp = 'conduit::Schema::Schema_Object_Hierarchy'
      elif dtype_id == self.types.LIST_ID:
         tp = 'conduit::Schema::Schema_List_Hierarchy'

      t = gdb.lookup_type(tp).pointer()
      hier_data = val['m_schema'].dereference()['m_hierarchy_data'].cast(t).dereference()['children']
      hier_data_first = hier_data['_M_impl']['_M_start']
      hier_data_last = hier_data['_M_impl']['_M_finish']
      return hier_data_last - hier_data_first


class ListNodePrinter(TreeNodePrinter):
   """Print a list conduit::Node object."""

   def __init__(self, val, types):
      super().__init__(types)
      self.val = val
      self.num_children = self.count_children(val)

   def to_string(self):
      return "{{ list length {0} }}".format(self.num_children)

   def display_hint(self):
      return 'array'

   def children (self):
      for i in range(self.num_children):
         yield ("[{0}]".format(i),
                self.val['m_children']['_M_impl']['_M_start'][i].dereference())
         # yield ("idx", "lv")


class ObjectNodePrinter(TreeNodePrinter):
   """Print an object conduit::Node object."""

   def __init__(self, val, types):
      super().__init__(types)
      self.val = val
      self.num_children = self.count_children(val)

   def to_string(self):
      return "{{ object children {0} }}".format(self.num_children)

   def display_hint(self):
      return 'map'

   def children (self):
      names = self.object_children_names()

      for i in range(2*self.num_children):
         yield ("{0}".format(names[i]),
                self.val['m_children']['_M_impl']['_M_start'][i].dereference())
      # return [("blah", "1"), ("blah", "a"), ("blah", "2"), ("blah", "b"), ("blah","3"), ("blah", "c")]

   def object_children_names(self):
      dtype_id = self.types.node_type(self.val)

      if dtype_id == self.types.OBJECT_ID:
         tp = 'conduit::Schema::Schema_Object_Hierarchy'
         t = gdb.lookup_type(tp).pointer()
         return self.val['m_schema'].dereference()['m_hierarchy_data'].cast(t)['object_order']['_M_impl']['_M_start']
      else:
         return None

def node_pp_function(val):
   if str(val.type) == 'conduit::Node':
      types = ConduitTypes()
      tid = types.node_type(val)
      if tid == types.EMPTY_ID:
         return EmptyNodePrinter()
      elif tid == types.CHAR8_STR_ID:
         return StringNodePrinter(val, types)
      elif tid == types.LIST_ID:
         return ListNodePrinter(val, types)
      elif tid == types.OBJECT_ID:
         return ObjectNodePrinter(val, types)
      else:
         elt_count = val['m_schema'].dereference()['m_dtype']['m_num_ele']
         if elt_count < 2:
            return ScalarNodePrinter(val, types)
         else:
            return ArrayNodePrinter(val, types)
   return None


gdb.pretty_printers.append(node_pp_function)
