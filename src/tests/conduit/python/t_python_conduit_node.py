# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_python_conduit_node.py
 description: Unit tests for conduit::Node python module interface.

"""

##############################################################
# make sure windows can pick up dlls for python 3.8 and newer
import os
if "PATH" in os.environ:
    for dll_path in os.environ["PATH"].split(";"):
        os.add_dll_directory(dll_path)
##############################################################

import sys
import unittest

from conduit import Node

import numpy as np

class Test_Conduit_Node(unittest.TestCase):
    def test_simple(self):
        a_val = np.uint32(10)
        b_val = np.uint32(20)
        c_val = np.float64(30.0)

        n = Node()
        n['a'] = a_val
        n['b'] = b_val
        n['c'] = c_val

        self.assertTrue(n['a'] == a_val)
        self.assertTrue(n['b'] == b_val)
        self.assertTrue(n['c'] == c_val)

    def test_nested(self):
        val = np.uint32(10)
        n = Node()
        n['a']['b'] = val
        print(n['a']['b'])
        self.assertEqual(n['a']['b'],val)

    def test_vector(self):
        vec = np.array(range(100), np.uint32)
        n = Node()
        n['a'] = vec
        self.assertEqual(n['a'][99], 99)

    def test_fetch(self):
        vec = np.array(range(100), np.uint32)
        n = Node()
        n['a'] = vec
        na = n.fetch('a')
        na_val = na.value()
        self.assertEqual(na_val[99], 99)

    def test_child(self):
        vec = np.array(range(100), np.uint32)
        n = Node()
        n['a'] = vec
        na = n.child(0)
        na_val = na.value()
        self.assertEqual(na_val[99], 99)
        n['b'] = vec
        self.assertEqual(n.number_of_children(),2)

    def test_save_load(self):
        # on windows, this breaks at 27 !?
        alen = 26
        vec = np.array(range(alen), np.uint32)
        n = Node()
        n['a'] = vec
        print(n)
        n.save("test_pyconduit_node_save_load.conduit_bin")
        nl = Node()
        nl.load("test_pyconduit_node_save_load.conduit_bin")
        print(nl)
        self.assertEqual(nl['a'][alen-1], alen-1)

        n.save("test_pyconduit_node_json_save_load.json",protocol="json")
        nl = Node()
        nl.load("test_pyconduit_node_json_save_load.json", protocol="json")
        print(nl)
        self.assertEqual(nl['a'][alen-1], alen-1)

        n.save("test_pyconduit_node_base64_json_save_load.conduit_base64_json", protocol="conduit_base64_json")
        nl = Node()
        nl.load("test_pyconduit_node_base64_json_save_load.conduit_base64_json", protocol="conduit_base64_json")
        print(nl)
        self.assertEqual(nl['a'][alen-1], alen-1)

        n.save("test_pyconduit_node_json_save_load.yaml",protocol="yaml")
        nl = Node()
        nl.load("test_pyconduit_node_json_save_load.yaml", protocol="yaml")
        print(nl)
        self.assertEqual(nl['a'][alen-1], alen-1)

    def test_parse(self):
        n = Node()
        n.parse('{"a": 42.0}',"json")
        self.assertTrue(n['a'] == np.float64(42.0))
        n.parse('a: 52.0',"yaml")
        self.assertTrue(n['a'] == np.float64(52.0))

    def test_parent(self):
        vec = np.array(range(100), np.uint32)
        n = Node()
        n['a'] = vec
        na = n.fetch('a')
        self.assertFalse(na.is_root())
        # todo: test parent()

    def test_total_bytes(self):
        vec = np.array(range(100), np.uint32)
        n = Node()
        n['a'] = vec
        self.assertEqual(n.total_strided_bytes(),4 * 100)
        self.assertEqual(n.total_bytes_compact(),4 * 100)
        # TODO: check if n.is_compact() should pass as well?
        # it doesn't currently
        self.assertTrue(n.fetch('a').is_compact())

    def test_paths(self):
        n = Node()
        n['a'] = 1
        n['b'] = 2
        n['c'] = 3
        for v in ['a','b','c']:
            self.assertTrue(n.has_path(v))
        paths = n.child_names()
        for v in ['a','b','c']:
            self.assertTrue(v in paths)

    def test_list(self):
        n = Node()
        n.append().set(1)
        self.assertTrue(n.child(0).value(),1)
        self.assertTrue(n[0],1)
        n2 = Node()
        n2_c = n2.append()
        n2_c.set(2)
        self.assertEqual(n2.child(0).value(),2)

        n3 = Node()
        n3.fetch("here").append().set("a")
        n3.fetch("here").append().set("b")
        self.assertTrue(n3.fetch("here").child(0).value(),"a")
        self.assertTrue(n3.fetch("here").child(1).value(),"b")

        n4 = Node()
        n4["here"].append().set("a")
        n5 = n4["here"]
        n5.append().set("b")
        self.assertTrue(n4["here"].child(0).value(),"a")
        self.assertTrue(n4["here"].child(1).value(),"b")
        self.assertTrue(n4["here"][0],"a")
        self.assertTrue(n4["here"][1],"b")

    def test_remove(self):
        n = Node()
        n['a'] = 1
        n['b'] = 2
        n['c'] = 3
        self.assertEqual(n.number_of_children(),3)
        n.remove(path='c')
        self.assertEqual(n.number_of_children(),2)
        paths = n.child_names()
        for v in ['a','b']:
            self.assertTrue(v in paths)
        n.remove(index=0)
        paths = n.child_names()
        for v in ['b']:
            self.assertTrue(v in paths)

    def test_info(self):
        n = Node()
        n['a'] = 1
        n['b'] = 2
        n['c'] = 3
        ni = n.info();
        #print ni
        self.assertEqual(ni["total_strided_bytes"],n.total_strided_bytes())

    def test_set_all_types(self):
        types = [ 'int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64',
                  'float32', 'float64']
        for type in types:
            data = np.array(range(10), dtype=type)
            n = Node()
            n.set(data)
            for i in range(len(data)):
                self.assertEqual(n.value()[i], data[i])

    def test_set_external(self):
        types = ['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']
        for type in types:
            ext_data = np.array(range(10), dtype=type)
            n = Node()
            n.set_external(ext_data)
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])
            ext_data[5] = 11
            n.value()[8] = 77
            n.value()[2] = 8
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])

    def test_set_external_node(self):
        n = Node()
        n.set(np.array(range(10), np.int32))
        n2 = Node()
        # test set external with node
        n2.set_external(n)
        for i in range(10):
            self.assertEqual(n.value()[i], n2.value()[i])
        n.value()[2] = 8
        n.value()[8] = 77
        # set of n should reflect in n2 with set_external
        self.assertEqual(8, n2.value()[2])
        self.assertEqual(77, n2.value()[8])

    def test_set_external_basic_slice(self):
        types = ['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']
        for type in types:
            base_data = np.array(range(20), dtype=type)
            ext_data  = base_data[1:16]
            n = Node()
            n.set_external(ext_data)
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])
            ext_data[5] = 11
            n.value()[6] = 77
            n.value()[2] = 8
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])

    def test_set_external_basic_strides(self):
        types = ['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']
        for type in types:
            base_data = np.array(range(20), dtype=type)
            ext_data  = base_data[1:16:2]
            n = Node()
            n.set_external(ext_data)
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])
            ext_data[5] = 11
            n.value()[6] = 77
            n.value()[2] = 8
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])

    def test_diff(self):
        n1 = Node()
        n2 = Node()
        info = Node()
        n1['a'] = 1
        self.assertTrue(n1.diff(n2,info))
        print(info)
        n2['a'] = 1
        self.assertFalse(n1.diff(n2,info))


        n2['b'] = 2.0
        self.assertTrue(n1.diff(n2,info))
        self.assertFalse(n1.diff_compatible(n2,info))
        n1['b'] = 1.0
        self.assertFalse(n1.diff(n2,info,10))

    def test_list_of_ints(self):
        # also covered by test_set_all_types
        # but this was the reproducer for
        #  https://github.com/LLNL/conduit/issues/281
        n = Node()
        a = np.array(list((1,2,3)))
        n['a'] = a
        self.assertEqual(n['a'][0], 1)
        self.assertEqual(n['a'][1], 2)
        self.assertEqual(n['a'][2], 3)

    def test_compact_to(self):
        n = Node()
        n['a'] = 1
        n['b'] = 2
        n['c'] = 3
        ni = n.info()
        self.assertEqual(ni["mem_spaces"].number_of_children(), 3)

        n2 = Node()
        n.compact_to(n2)
        ni = n2.info()
        print(ni)
        self.assertEqual(ni["mem_spaces"].number_of_children(), 1)

    def test_update(self):
        n = Node()
        data = np.array(range(10), dtype='float64')
        n["data"].set_external(data)

        print(n)

        n2 = Node()
        n2.update(n)
        print(n2)
        self.assertEqual(n2["data"][0],0)

        n3 = Node()
        n3.update_external(n)
        data[0] = 10
        print(n3)
        self.assertEqual(n3["data"][0],10)

        n4 = Node()
        n4["data"] = 10
        n4.update_compatible(n)
        print(n4)
        self.assertEqual(n4["data"],10)


    def test_reset(self):
        n = Node()
        data = np.array(range(10), dtype='float64')
        n["data"].set_external(data)

        print(n)

        n.reset()
        self.assertEqual(n.number_of_children(), 0)

    def test_child_rename(self):
        a_val = np.uint32(10)
        b_val = np.uint32(20)

        n = Node()
        with self.assertRaises(Exception):
            n.rename_child('a','b')

        n['a'] = a_val
        n['b'] = b_val

        with self.assertRaises(Exception):
            n.rename_child('bad','good')

        with self.assertRaises(Exception):
            n.rename_child('b','a')

        self.assertTrue(n['a'] == a_val)
        self.assertTrue(n['b'] == b_val)

        n.rename_child('b','c')

        self.assertTrue(n['a'] == a_val)
        self.assertTrue(n['c'] == b_val)

    def test_string(self):
        n = Node();
        n.set("my string!")
        print(n)
        self.assertEqual(n.value(),"my string!")
        # test numpy string
        nps = np.string_("my numpy string!")
        n.set(nps)
        print(n)
        print(repr(n))
        self.assertEqual(n.value(),"my numpy string!")
        aofstrs = np.array(["here","are","a","few","strings"])
        print(aofstrs)
        n.set(aofstrs)
        print(n)
        self.assertEqual(n[0],"here")
        self.assertEqual(n[1],"are")
        self.assertEqual(n[2],"a")
        self.assertEqual(n[3],"few")
        self.assertEqual(n[4],"strings")

    def test_numeric_tuples(self):
        n = Node()
        n["tuple_0"].set((1, 2, 3, 4))
        n["tuple_1"].set((1.0, 2.0, 3.0, 4.0))
        n["tuple_2"].set((1, 2, 3, 4.0))
        print(n)
        self.assertEqual(n['tuple_0'][0], 1)
        self.assertEqual(n['tuple_0'][1], 2)
        self.assertEqual(n['tuple_0'][2], 3)
        self.assertEqual(n['tuple_0'][3], 4)

        self.assertEqual(n['tuple_1'][0], 1.0)
        self.assertEqual(n['tuple_1'][1], 2.0)
        self.assertEqual(n['tuple_1'][2], 3.0)
        self.assertEqual(n['tuple_1'][3], 4.0)

        self.assertEqual(n['tuple_2'][0], 1.0)
        self.assertEqual(n['tuple_2'][1], 2.0)
        self.assertEqual(n['tuple_2'][2], 3.0)
        self.assertEqual(n['tuple_2'][3], 4.0)


    def test_numeric_lists(self):
        n = Node()
        n["list_0"].set((1, 2, 3, 4))
        n["list_1"].set((1.0, 2.0, 3.0, 4.0))
        n["list_2"].set((1, 2, 3, 4.0))
        print(n)
        self.assertEqual(n['list_0'][0], 1)
        self.assertEqual(n['list_0'][1], 2)
        self.assertEqual(n['list_0'][2], 3)
        self.assertEqual(n['list_0'][3], 4)

        self.assertEqual(n['list_1'][0], 1.0)
        self.assertEqual(n['list_1'][1], 2.0)
        self.assertEqual(n['list_1'][2], 3.0)
        self.assertEqual(n['list_1'][3], 4.0)

        self.assertEqual(n['list_2'][0], 1.0)
        self.assertEqual(n['list_2'][1], 2.0)
        self.assertEqual(n['list_2'][2], 3.0)
        self.assertEqual(n['list_2'][3], 4.0)

    def test_general_tuples(self):
        n = Node()
        n.set((1, "here"))
        print(n)
        self.assertEqual(n[0], 1.0)
        self.assertEqual(n[1], "here")

    def test_general_lists(self):
        n = Node()
        n.set([1, "there"])
        print(n)
        self.assertEqual(n[0], 1.0)
        self.assertEqual(n[1], "there")

    def test_key_with_slash(self):
        n = Node()
        n["normal/path"] = 10
        n.add_child("child_with_/_inside").set(42)
        print(n)
        self.assertTrue(n.has_path("normal/path"))
        self.assertFalse(n.has_child("normal/path"))
        self.assertFalse(n.has_path("child_with_/_inside"))
        self.assertTrue(n.has_child("child_with_/_inside"))
        self.assertEqual(2,n.number_of_children())
        self.assertEqual(n["normal/path"],10);
        self.assertEqual(n.child(name="child_with_/_inside").value(),42);
        n["normal"].remove_child("path")
        self.assertFalse(n.has_path("normal/path"))

    def test_fetch_existing(self):
        n = Node()
        n["my/path"] = 10

        n_sub = n.fetch_existing("my/path")

        self.assertEqual(n_sub.value(),10);

        with self.assertRaises(Exception):
            n.fetch_existing('bad/path')

    def test_to_string(self):
        a_val = np.uint32(10)
        b_val = np.uint32(20)
        c_val = np.float64(30.0)

        n = Node()
        n['a'] = a_val
        n['b'] = b_val
        n['c'] = c_val

        res_to_str_def  = n.to_string()
        res_to_str_yaml = n.to_string(protocol="yaml")
        res_to_str_json = n.to_string(protocol="json")

        res_to_yaml = n.to_yaml()
        res_to_json = n.to_json()

        self.assertEqual(res_to_str_def,  res_to_yaml);
        self.assertEqual(res_to_str_yaml, res_to_yaml);

        self.assertEqual(res_to_str_json,  res_to_json);

        n.print_detailed()

    def test_numpy_slice_as_set_input(self):
        n = Node()
        # slice with non trivial strides
        numpy_array = np.array(range(21), dtype='float64')
        v = numpy_array.reshape((3, 7))
        print("Input Array")
        print(v)
        print("Desired Slice")
        print(v[:,0])
        n['v'] = v
        n['vs'] = v[:,0]
        n['vs_expected'] = np.array(v[:,0],np.float64)
        print(n)
        sdiff = np.setdiff1d(n['vs'], v[:,0])
        print("Set Difference: ",sdiff )
        self.assertEqual(len(sdiff), 0);
        # a more complex slice
        numpy_array = np.array(range(105), dtype='float64')
        v = numpy_array.reshape((3, 7, 5))
        print("Input Array")
        print(v)
        print("Desired Slice")
        print(v[:,0,3:5])
        n['v'] = v
        n['vs'] = v[:,0,3:5]
        n['vs_expected'] = np.array(v[:,0,3:5],np.float64)
        print(n)
        sdiff = np.setdiff1d(n['vs'], v[:,0,3:5])
        print("Set Difference: ",sdiff )
        self.assertEqual(len(sdiff), 0);


    def test_numpy_slice_as_set_external_input(self):
        n = Node()
        # slice with non trivial strides
        numpy_array = np.array(range(21), dtype='float64')
        v = numpy_array.reshape((3, 7))
        print("Input Array")
        print(v)
        print("Desired Slice")
        print(v[:,0])
        n['v'] = v
        n['vs'].set_external(v[:,0])
        n['vs_expected'] = np.array(v[:,0],np.float64)
        print(n)
        sdiff = np.setdiff1d(n['vs'], v[:,0])
        print("Set Difference: ",sdiff )
        self.assertEqual(len(sdiff), 0);
        # a more complex slice, can't use set external here.
        n = Node()
        numpy_array = np.array(range(105), dtype='float64')
        v = numpy_array.reshape((3, 7, 5))
        with self.assertRaises(TypeError):
            n['vs'].set_external(v[:,0,3:5])
        # lets do a 1-d eff slice, this should work since
        # it reduces to a 1-D strided case
        n['vs'].set_external(v[:,0,0])
        n['vs_expected'] = np.array(v[:,0,0],np.float64)


    def test_describe(self):
        n = Node()
        n["a"] = [1,2,3,4,5];
        n["b"] = [1,2,3];
        n["c"] = [1,2,3,4,5,6];
        n["d"] = [1,2,3,4,5,6,7];
        n["e"] = [1,2,3,4,5,6,7,8,9,10,11,12];
        n["f"] = [1.0,2.0,3.0,4.0,5.0,6.0,7.0];
        n["g"] = [2.0,4.0];

        d = n.describe()
        print(d)

        self.assertEqual(d["a/count"],5);
        self.assertEqual(d["b/count"],3);
        self.assertEqual(d["c/count"],6);
        self.assertEqual(d["d/count"],7);
        self.assertEqual(d["e/count"],12);
        self.assertEqual(d["f/count"],7);

        self.assertEqual(d["a/min"],1)
        self.assertEqual(d["b/min"],1)
        self.assertEqual(d["c/min"],1)
        self.assertEqual(d["d/min"],1)
        self.assertEqual(d["e/min"],1)
        self.assertEqual(d["f/min"],1.0)

        self.assertEqual(d["a/max"],5)
        self.assertEqual(d["b/max"],3)
        self.assertEqual(d["c/max"],6)
        self.assertEqual(d["d/max"],7)
        self.assertEqual(d["e/max"],12)
        self.assertEqual(d["f/max"],7.0)

        self.assertEqual(d["g/mean"],3.0);

        opts = Node()
        opts["threshold"] = 10
        d = n.describe(opts)
        print(d)

    def test_summary_string(self):
        n = Node()
        n["a"] = [1,2,3,4,5];
        n["b"] = [1,2,3];
        n["c"] = [1,2,3,4,5,6];
        n["d"] = [1,2,3,4,5,6,7];
        n["e"] = [1,2,3,4,5,6,7,8,9,10,11,12];
        n["f"] = [1.0,2.0,3.0,4.0,5.0,6.0,7.0];
        n["g"] = [2.0,4.0];

        print(repr(n))

        r = n.to_summary_string()
        print(r)
        texp = """
a: [1, 2, 3, 4, 5]
b: [1, 2, 3]
c: [1, 2, 3, ..., 5, 6]
d: [1, 2, 3, ..., 6, 7]
e: [1, 2, 3, ..., 11, 12]
f: [1.0, 2.0, 3.0, ..., 6.0, 7.0]
g: [2.0, 4.0]
"""
        self.assertEqual(r,texp)
    
        opts = Node()
        opts["num_children_threshold"] = 2
        opts["num_elements_threshold"] = 3
        r = n.to_summary_string(opts)
        print(r)

        texp = """
a: [1, 2, ..., 5]
... ( skipped 5 children )
g: [2.0, 4.0]
"""
        self.assertEqual(r,texp)
        r = n.to_summary_string(opts=opts)
        print(r)

        self.assertEqual(r,texp)

        opts = Node()
        opts["num_children_threshold"] = 100
        opts["num_elements_threshold"] = -1
        r = n.to_summary_string(opts)
        print(r)

        self.assertEqual(r,n.to_yaml())


if __name__ == '__main__':
    unittest.main()


