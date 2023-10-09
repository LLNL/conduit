# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

import gdb
import gdb.printing
import itertools


class EmptyNodePrinter:
   """Print an empty conduit::Node object."""

   def __init__(self, val):
      self.val = val

   def to_string(self):
      return "empty Node"


class StringNodePrinter:
   """Print a string conduit::Node object."""

   def __init__(self, val):
      self.val = val

   def to_string(self):
      t = gdb.lookup_type('const char *').pointer()
      v = self.val['m_data'].cast(t)
      return v

   def display_hint():
      return 'string'

def type_string(val):
   dtype_id = val['m_schema'].dereference()
   dtype_id = dtype_id['m_dtype']['m_id']
   tp = "int"
   if dtype_id == CONDUIT_INT8_ID:
      tp = "int8"
   elif dtype_id == CONDUIT_INT16_ID:
      tp = "int16"
   elif dtype_id == CONDUIT_INT32_ID:
      tp = "int32"
   elif dtype_id == CONDUIT_INT64_ID:
      tp = "int64"
   elif dtype_id == CONDUIT_UINT8_ID:
      tp = "uint8"
   elif dtype_id == CONDUIT_UINT16_ID:
      tp = "uint16"
   elif dtype_id == CONDUIT_UINT32_ID:
      tp = "uint32"
   elif dtype_id == CONDUIT_UINT64_ID:
      tp = "uint64"
   elif dtype_id == CONDUIT_FLOAT32_ID:
      tp = "float32"
   elif dtype_id == CONDUIT_FLOAT64_ID:
      tp = "double"

   return tp

def count_children(val):
   schema = val['schema']
   dtype_id = schema['m_dtype']['m_id']

   tp = ''
   if dtype_id == DataType::OBJECT_ID:
      tp = 'Schema_Object_Hierarchy'
   elif dtype_id == DataType::LIST_ID:
      tp = 'Schema_List_Hierarchy'

   t = gdb.lookup_type(tp).pointer()
   hier_data = schema['m_hierarchy_data'].cast(t)['children']
   hier_data_first = hier_data['_Myfirst']
   hier_data_last = hier_data['_Mylast']
   return hier_data_last - hier_data_first

def object_children_names(val):
   schema = val['schema']
   dtype_id = schema['m_dtype']['m_id']

   tp = ''
   if dtype_id == DataType::OBJECT_ID:
      tp = 'Schema_Object_Hierarchy'

   t = gdb.lookup_type(tp).pointer()
   return schema['m_hierarchy_data'].cast(t)['object_order']

class ArrayNodePrinter:
   """Print an array numeric conduit::Node object."""

   def __init__(self, val):
      self.val = val

   def to_string(self):
      tp = type_string(self.val)
      t = gdb.lookup_type(tp).pointer()
      v = self.val['m_data'].cast(t)

      elt_count = self.val['m_schema'].dereference()['m_dtype']['m_num_ele']

      for i in range(elt_count):
         yield "[{0}]".format(i), v[i]

   def display_hint():
      return 'array'


class ScalarNodePrinter:
   """Print a scalar numeric conduit::Node object."""

   def __init__(self, val):
      self.val = val

   def to_string(self):
      tp = type_string(self.val)
      t = gdb.lookup_type(tp).pointer()
      v = self.val['m_data'].cast(t)
      return v[0]


class ListNodePrinter:
   """Print a list conduit::Node object."""

   def __init__(self, val):
      self.val = val
      self.num_children = count_children(val)

   def to_string(self):
      return "{{ list length {0} }}".format(self.num_children)

   def display_hint():
      return 'array'

   def children (self):
      for i in range(self.num_children):
         yield "[{0}]".format(i), self.val['m_children'][i]


class ObjectNodePrinter:
   """Print an object conduit::Node object."""

   def __init__(self, val):
      self.val = val
      self.num_children = count_children(val)

   def to_string(self):
      return "{{ object children {0} }}".format(self.num_children)

   def display_hint():
      return 'map'

   def children (self):
      names = object_children_names(self.val)
      
      for i in range(self.num_children):
         yield "{0}".names[i], self.val['m_children'][i]


def node_pp_function(val):
   if str(val.type) == 'Node':
      tid = get_Node_type_id(val)
      if tid == EMPTY_TYPE:
         return EmptyNodePrinter(val)
      elif tid == STRING_TYPE:
         return StringNodePrinter(val)
      elif tid == LIST_TYPE:
         return ListNodePrinter(val)
      elif tid == OBJECT_TYPE:
         return ObjectNodePrinter(val)
      else:
         elt_count = self.val['m_schema'].dereference()['m_dtype']['m_num_ele']
         if elt_count < 2:
            return ScalarNodePrinter(val)
         else:
            return ArrayNodePrinter(val)
   return None


gdb.pretty_printers.append(node_pp_function)
