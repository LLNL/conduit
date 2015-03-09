//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <Python.h>
#include <structmember.h>
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------//
// include numpy
// 
// TODO: Use 1.7 deprecated API, or not ?
//---------------------------------------------------------------------------//
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "conduit.h"
#include "Conduit_Python_Exports.h"

using namespace conduit;

static PyObject* PyConduitSchema_getObject(const Schema* schema);
static int       PyConduitSchema_Check(PyObject* obj);
static PyObject* PyConduitNode_getObject(Node* node);
static int       PyConduitNode_Check(PyObject* obj);
static int       PyConduitNode_SetFromPython(Node& node, PyObject* value);
static PyObject* PyConduit_createNumpyType(Node& node, int type);
static PyObject* PyConduit_convertNodeToPython(Node& node);
static PyObject* getType(const char* name);

struct Py_ConduitSchema {
    PyObject_HEAD
    Schema* schema;
    int owns_data;
};

//---------------------------------------------------------------------------//
static PyObject * 
PyConduitSchema_New(PyTypeObject* type,
                    PyObject* args,
                    PyObject* kwds)
{
    static char *kwlist[] = {"value", NULL};
    PyObject* value = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &value)) {
        return (NULL);
    }

    Py_ConduitSchema* self = (Py_ConduitSchema*)type->tp_alloc(type, 0);
    if (self) {
        self->schema = 0;
        self->owns_data = 0;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static int
PyConduitSchema_init(Py_ConduitSchema* self,
                     PyObject* args,
                     PyObject* kwds)
{
     static char *kwlist[] = {"value", NULL};
     PyObject* value = NULL;
     if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &value)) {
         return (NULL);
     }

     if (value) {

         if (PyConduitSchema_Check(value)) {
             self->schema = new Schema(*((Py_ConduitSchema*)value)->schema);
         } else if (PyString_Check(value)) {
             self->schema = new Schema(PyString_AsString(value));
         } else {
             PyErr_SetString(PyExc_TypeError, "Invalid initializer for schema");
             return (-1);
         }
     } else {
         self->schema = new Schema();
     }

     self->owns_data = 1;

     return (0);

}

//---------------------------------------------------------------------------//
static void
PyConduitSchema_dealloc(Py_ConduitSchema* self)
{
    if (self->owns_data) {
        delete self->schema;
    }
    self->ob_type->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitSchema_str(Py_ConduitSchema* self)
{
   std::string output = self->schema->to_json();
   return (Py_BuildValue("s", output.c_str()));
}

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
static PyTypeObject PyConduit_SchemaType = {
   PyObject_HEAD_INIT(NULL)
   0,
   "Schema",
   sizeof(Py_ConduitSchema),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyConduitSchema_dealloc,                   /* tp_dealloc */
   0, /* tp_print */
   0, /* tp_getattr */
   0, /* tp_setattr */
   0, /* tp_compare */
   0, /* tp_repr */
   0, /* tp_as_number */
   0, /* tp_as_sequence */
   0, /* as_mapping */
   0, /* hash */
   0, /* call */
   (reprfunc)PyConduitSchema_str,                         /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "Conduit schema objects",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   0, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyConduitSchema_init,
   0, /* alloc */
   PyConduitSchema_New,                                   /* new */
   0, /* tp_free */
   0, /* tp_is_gc */
   0, /* tp_bases */
   0, /* tp_mro */
   0, /* tp_cache */
   0, /* tp_subclasses */
   0,  /* tp_weaklist */
   0,
   0
};

//---------------------------------------------------------------------------//
struct Py_ConduitNodeIter {
    PyObject_HEAD
};

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNodeIter_next(PyObject* self)
{
    // if done return NULL, else return item and increment
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit:::Node class
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
struct Py_ConduitNode {
   PyObject_HEAD
   Node* node;
};

//---------------------------------------------------------------------------//
template <class T>
static void
PyConduit_fillVector(std::vector<T>& vec,
                     PyArrayObject* arr)
{
    T* data = (T*)PyArray_BYTES(arr);
    npy_intp size = PyArray_SIZE(arr);
    vec.resize(size);
    for (npy_intp i = 0; i < size; i++) {
        vec[i] = data[i];
    }
    
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_New(PyTypeObject* type,
                  PyObject* args,
                  PyObject* kwds)
{
    static char *kwlist[] = {"value", NULL};
    PyObject* value = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &value)) {
        return (NULL);
    }

    Py_ConduitNode* self = (Py_ConduitNode*)type->tp_alloc(type, 0);
    if (self) {
        self->node = 0;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static int
PyConduitNode_init(Py_ConduitNode* self,
                   PyObject* args,
                   PyObject* kwds)
{
     static char *kwlist[] = {"value", NULL};
     PyObject* value = NULL;
     if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &value)) {
         return (NULL);
     }
    self->node = new Node();

    if (value) {
        return (PyConduitNode_SetFromPython(*self->node, value));
    } else {
        return (0);
    }
}

//---------------------------------------------------------------------------//
static void 
PyConduitNode_dealloc(Py_ConduitNode* self)
{
   if (self->node->schema().is_root()) {
       delete self->node;
   }
   self->ob_type->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_str(Py_ConduitNode* self)
{
   std::string output = self->node->to_json();
   return (Py_BuildValue("s", output.c_str()));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_GetItem(Py_ConduitNode* self,
                      PyObject* key)
{
    if (!PyString_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Key must be a string");
        return (NULL);
    }

    PyObject* retval = NULL;
    char* ckey = PyString_AsString(key);

    if (self->node->has_path(ckey)) {
        Node& node = (*self->node)[ckey];
        retval = PyConduit_convertNodeToPython(node);
    } else {
        retval = PyConduitNode_getObject(&(*self->node)[ckey]);
    }
    return (retval);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_data(Py_ConduitNode* self)
{
    PyObject* retval = NULL;
    retval = PyConduit_convertNodeToPython(*self->node);
    return (retval);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_generate(Py_ConduitNode* self,
                       PyObject* args)
{
    /// TODO: sigs to support
    /// json_schema
    /// json_schema, protocol
    
    /// json_schema, data
    /// json_schema, protocol, data
        
     const char *json_schema;
     if (!PyArg_ParseTuple(args, "s", &json_schema))
     {
         PyErr_SetString(PyExc_TypeError, "Save file path must be a string");
         return NULL;
     }
     
     self->node->generate(std::string(json_schema));
     Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_save(Py_ConduitNode* self,
                   PyObject* args)
{
     const char *obase;     
     if (!PyArg_ParseTuple(args, "s", &obase))
     {
         PyErr_SetString(PyExc_TypeError, "Save file path must be a string");
         return NULL;
     }
     
     self->node->save(std::string(obase));
     Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_load(Py_ConduitNode* self,
                   PyObject* args)
{
    /// TODO: sigs to support via kwargs: path, or path and schema
    const char *ibase;     
    if (!PyArg_ParseTuple(args, "s", &ibase))
    {
        PyErr_SetString(PyExc_TypeError, "Load file path must be a string");
        return NULL;
    }

    self->node->load(std::string(ibase));
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_mmap(Py_ConduitNode* self,
                   PyObject* args)
{
    /// TODO: sigs to support via kwargs: path, or path and schema
    const char *ibase;     
    if (!PyArg_ParseTuple(args, "s", &ibase))
    {
        PyErr_SetString(PyExc_TypeError, "Load file path must be a string");
        return NULL;
    }

    self->node->mmap(std::string(ibase));
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_fetch(Py_ConduitNode* self,
                     PyObject* args)
{
     const char *key;
     PyObject* retval = NULL;
     if (!PyArg_ParseTuple(args, "s", &key))
     {
         PyErr_SetString(PyExc_TypeError, "Key must be a string");
         return NULL;
     }

    retval = PyConduitNode_getObject(&(*self->node).fetch(key));
    return (retval);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_child(Py_ConduitNode* self,
                    PyObject* args)
{
     Py_ssize_t idx;
     PyObject* retval = NULL;
     if (!PyArg_ParseTuple(args, "n", &idx))
     {
         PyErr_SetString(PyExc_TypeError, "Index must be an integer");
         return NULL;
     }

    retval = PyConduitNode_getObject(&(*self->node).child(idx));
    return (retval);
}

//-----------------------------------------------------------------------------
//
// -- Node information methods --
//
//-----------------------------------------------------------------------------
// schema access
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_schema(Py_ConduitNode *self)
{
    return (PyConduitSchema_getObject(&self->node->schema()));
}
//---------------------------------------------------------------------------//
/// TODO: dtype()
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// parent access
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject * 
PyConduitNode_has_parent(Py_ConduitNode *self)
{
    if(self->node->has_parent())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject* 
PyConduitNode_parent(Py_ConduitNode* self)
{
    if(~self->node->has_parent())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return PyConduitNode_getObject(self->node->parent());
    }
}

//---------------------------------------------------------------------------//
//memory space info
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_total_bytes(Py_ConduitNode *self)
{
    return PyLong_FromSsize_t(self->node->total_bytes());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_total_bytes_compact(Py_ConduitNode *self)
{
    return PyLong_FromSsize_t(self->node->total_bytes_compact());
}

//---------------------------------------------------------------------------//
static PyObject * 
PyConduitNode_is_compact(Py_ConduitNode *self)
{
    if(self->node->is_compact())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}
//---------------------------------------------------------------------------//
///TODO: info
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static int PyConduitNode_SetItem(Py_ConduitNode* self, PyObject* key,
                                       PyObject* value)
{
    if (!PyString_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Key must be a string");
        return (-1);
    }

    char* ckey = PyString_AsString(key);
    Node& node = (*self->node)[ckey];

    return (PyConduitNode_SetFromPython(node, value));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_iter(Py_ConduitNode* self)
{
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_set(Py_ConduitNode* self,
                  PyObject* args)
{
    PyObject* value;
    if (!PyArg_ParseTuple(args, "O", &value)) {
         return (NULL);
    }

    if (PyConduitNode_SetFromPython(*self->node, value)) {
         return (NULL);
    } else {
        Py_RETURN_NONE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_set_path(Py_ConduitNode* self,
                       PyObject* args)
{
    PyObject* value, *path;
    if (!PyArg_ParseTuple(args, "OO", &path, &value)) {
         return (NULL);
    }

    if (PyConduitNode_SetItem(self, path, value)) {
        return (NULL);
    } else {
        Py_RETURN_NONE;
    }
}

//----------------------------------------------------------------------------//
// Node methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyConduitNode_METHODS[] = {
    //-----------------------------------------------------------------------//
    {"set",
     (PyCFunction)PyConduitNode_set,
     METH_VARARGS,
     "Sets the node"},
    //-----------------------------------------------------------------------//
    {"fetch",
     (PyCFunction)PyConduitNode_fetch,
     METH_VARARGS, 
     "Fetches the node at a given path"},
    //-----------------------------------------------------------------------//
    {"child",
     (PyCFunction)PyConduitNode_child,
     METH_VARARGS, 
     "Retrieves the child node at a given index"},
    //-----------------------------------------------------------------------//
    {"data",
     (PyCFunction)PyConduitNode_data,
     METH_NOARGS, 
     "{data val of node}"},
    //-----------------------------------------------------------------------//
    {"generate",
     (PyCFunction)PyConduitNode_generate,
     METH_VARARGS,  // will become kwargs
     "Generate a node"},
    //-----------------------------------------------------------------------//
    {"save",
     (PyCFunction)PyConduitNode_save,
     METH_VARARGS, 
     "Saves a node to a file pair"},
    //-----------------------------------------------------------------------//
    {"load",
     (PyCFunction)PyConduitNode_load,
     METH_VARARGS,  // will become kwargs
     "Loads a node from a file pair"},
    //-----------------------------------------------------------------------//
    {"mmap",
     (PyCFunction)PyConduitNode_mmap,
     METH_VARARGS, // will become kwargs
     "Memory Maps a node from a file pair"},
    //-----------------------------------------------------------------------//
    {"set_path",
     (PyCFunction)PyConduitNode_set_path,
     METH_VARARGS,
     "Sets the node at the given path"},
    //-----------------------------------------------------------------------//
    {"schema",
     (PyCFunction)PyConduitNode_schema, 
     METH_NOARGS,
     "Returns the schema for the node"}, 
    //-----------------------------------------------------------------------//
    {"has_parent",
     (PyCFunction)PyConduitNode_has_parent, 
     METH_NOARGS,
     "Returns of the node has a parent node"}, 
    //-----------------------------------------------------------------------//
    {"parent",
     (PyCFunction)PyConduitNode_parent, 
     METH_NOARGS,
     "Returns this nodes parent, or None if no parent"}, 
    //-----------------------------------------------------------------------//
    {"total_bytes",
     (PyCFunction)PyConduitNode_total_bytes, 
     METH_NOARGS,
     "Returns the total bytes of this node's data"}, 
    //-----------------------------------------------------------------------//
    {"total_bytes_compact",
     (PyCFunction)PyConduitNode_total_bytes_compact, 
     METH_NOARGS,
     "Returns the total bytes of compact rep of node's data"}, 
    //-----------------------------------------------------------------------//
    {"is_compact",
     (PyCFunction)PyConduitNode_is_compact, 
     METH_NOARGS,
     "Returns if this node's data is in compact form"}, 
    //-----------------------------------------------------------------------//
    // end node methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
static PyMappingMethods node_as_mapping = {
   (lenfunc)0,    // len operator is not supported
   (binaryfunc)PyConduitNode_GetItem,
   (objobjargproc)PyConduitNode_SetItem,
};

//---------------------------------------------------------------------------//
static PyTypeObject PyConduit_NodeType = {
   PyObject_HEAD_INIT(NULL)
   0,
   //PyObject_VAR_HEAD
   "Node",
   sizeof(Py_ConduitNode),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyConduitNode_dealloc,                   /* tp_dealloc */
   0, /* tp_print */
   0, /* tp_getattr */
   0, /* tp_setattr */
   0, /* tp_compare */
   0, /* tp_repr */
   0, /* tp_as_number */
   0, /* tp_as_sequence */
   &node_as_mapping, /* as_mapping */
   0, /* hash */
   0, /* call */
   (reprfunc)PyConduitNode_str,                         /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "Conduit node objects",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   (getiterfunc)PyConduitNode_iter, /* iter */
   0, /* iternext */
   PyConduitNode_METHODS,
   0,
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyConduitNode_init,
   0, /* alloc */
   PyConduitNode_New,                                   /* new */
   0, /* tp_free */
   0, /* tp_is_gc */
   0, /* tp_bases */
   0, /* tp_mro */
   0, /* tp_cache */
   0, /* tp_subclasses */
   0,  /* tp_weaklist */
   0,
   0
};


//---------------------------------------------------------------------------//
// conduit:::about
//---------------------------------------------------------------------------//
static PyObject*
py_conduit_about(PyObject *self)
{
    return PyString_FromString(conduit::about().c_str());
}


//---------------------------------------------------------------------------//
// Python Module Method Defs
//---------------------------------------------------------------------------//
static PyMethodDef conduit_python_funcs[] =
{
    //-----------------------------------------------------------------------//
    {"about",
     (PyCFunction)py_conduit_about,
      METH_NOARGS,
      NULL},
    //-----------------------------------------------------------------------//
    // end conduit methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, METH_VARARGS, NULL}
};

//---------------------------------------------------------------------------//
// Main entry point
//---------------------------------------------------------------------------//
extern "C" void
CONDUIT_PYTHON_API initconduit_python(void)
{
    if (PyType_Ready(&PyConduit_SchemaType) < 0) {
        return;
    }
    PyObject* schema    = Py_InitModule3("Schema", PyConduitNode_METHODS,
                                       "Schema class for Conduit");
    PyModule_AddObject(schema, "Schema", (PyObject*)&PyConduit_SchemaType);

    if (PyType_Ready(&PyConduit_NodeType) < 0) {
        return;
    }

    PyObject* node    = Py_InitModule3("Node", PyConduitNode_METHODS,
                                       "Node class for Conduit");
    PyModule_AddObject(node, "Node", (PyObject*)&PyConduit_NodeType);

    PyObject* conduit =  Py_InitModule("conduit_python", conduit_python_funcs);
    Py_INCREF(schema);
    PyModule_AddObject(conduit, "Schema", schema);
    Py_INCREF(node);
    PyModule_AddObject(conduit, "Node", node);
    // req setup for numpy
    import_array();
}

//---------------------------------------------------------------------------//
static PyObject *
getType(const char* name)
{
    PyObject* module = PyImport_AddModule(name);
    PyObject* dict = PyModule_GetDict(module);
    PyObject* object = PyDict_GetItemString(dict, name);
    Py_INCREF(object);
    return (object);
}

//---------------------------------------------------------------------------//
static int
PyConduitSchema_Check(PyObject* obj)
{
    return (PyObject_TypeCheck(obj, &PyConduit_SchemaType));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitSchema_getObject(const Schema* schema)
{
    PyTypeObject* type = (PyTypeObject*)getType("Schema");
    Py_ConduitSchema* retval = (Py_ConduitSchema*)type->tp_alloc(type, 0);
    retval->schema = (Schema*)schema;
    retval->owns_data = 0;
    return ((PyObject*)retval);
}

//---------------------------------------------------------------------------//
static int
PyConduitNode_Check(PyObject *obj)
{
    return (PyObject_TypeCheck(obj, &PyConduit_NodeType));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduitNode_getObject(Node *node)
{
    PyTypeObject* type = (PyTypeObject*)getType("Node");
    Py_ConduitNode* retval = (Py_ConduitNode*)type->tp_alloc(type, 0);
    retval->node = node;
    return ((PyObject*)retval);
}
//---------------------------------------------------------------------------//
static int
PyConduitNode_SetFromPython(Node &node,
                            PyObject *value)
{
    if (PyConduitNode_Check(value)) {
        node = *((Py_ConduitNode*)value)->node;
    } else if (PyConduitSchema_Check(value)) {
        node = *((Py_ConduitSchema*)value)->schema;
    } else if (PyString_Check(value)) {
        node = PyString_AsString(value);
    } else if (PyInt_Check(value)) {
        node = PyInt_AsLong(value);
    } else if (PyLong_Check(value)) {
        node = PyLong_AsLong(value);
    } else if (PyFloat_Check(value)) {
        node = PyFloat_AS_DOUBLE(value);
    } else if (PyArray_Check(value)) {
        PyArray_Descr* desc = PyArray_DESCR((PyArrayObject*)value);
        PyArrayObject* arr = (PyArrayObject*)value;
        switch (desc->type_num) {
            case NPY_UINT8 : {
                std::vector<uint8> vec;
                PyConduit_fillVector(vec, arr);
                node = vec;
                break;
            }
            case NPY_UINT16 : {
                std::vector<uint16> vec;
                PyConduit_fillVector(vec, arr);
                node = vec;
                break;
            }
            case NPY_UINT32 : {
                std::vector<uint32> vec;
                PyConduit_fillVector(vec, arr);
                node = vec;
                break;
            }
            case NPY_UINT64 : {
                std::vector<uint64> vec;
                PyConduit_fillVector(vec, arr);
                node = vec;
                break;
            }
            case NPY_INT8 : {
                std::vector<int8> vec;
                PyConduit_fillVector(vec, arr);
                node = vec;
                break;
            }
            case NPY_INT16 : {
                std::vector<int16> vec;
                PyConduit_fillVector(vec, arr);
                node = vec;
                break;
            }
            case NPY_INT32 : {
                std::vector<int32> vec;
                PyConduit_fillVector(vec, arr);
                node = vec;
                break;
            }
            case NPY_INT64 : {
                std::vector<int64> vec;
                PyConduit_fillVector(vec, arr);
                node = vec;
                break;
            }
            case NPY_FLOAT32 : {
                std::vector<float32> vec;
                PyConduit_fillVector(vec, arr);
                node = vec;
                break;
            }
            case NPY_FLOAT64 : {
                std::vector<float64> vec;
                PyConduit_fillVector(vec, arr);
                node = vec;
                break;
            }
        }
    } else if (PyArray_CheckScalar(value)) {
        PyArray_Descr* desc = PyArray_DescrFromScalar(value);
        switch (desc->type_num) {
            case NPY_INT8 : {
                int8 val;
                PyArray_ScalarAsCtype(value, &val);
                node = val;
                break;
            }
            case NPY_INT16 : {
                int16 val;
                PyArray_ScalarAsCtype(value, &val);
                node = val;
                break;
            }
            case NPY_INT32 : {
                int32 val;
                PyArray_ScalarAsCtype(value, &val);
                node = val;
                break;
            }
            case NPY_INT64 : {
                int64 val;
                PyArray_ScalarAsCtype(value, &val);
                node = val;
                break;
            }
            case NPY_UINT8 : {
                uint8 val;
                PyArray_ScalarAsCtype(value, &val);
                node = val;
                break;
            }
            case NPY_UINT16 : {
                uint16 val;
                PyArray_ScalarAsCtype(value, &val);
                node = val;
                break;
            }
            case NPY_UINT32 : {
                uint32 val;
                PyArray_ScalarAsCtype(value, &val);
                node = val;
                break;
            }
            case NPY_UINT64 : {
                uint64 val;
                PyArray_ScalarAsCtype(value, &val);
                node = val;
                break;
            }
            case NPY_FLOAT32 : {
                float32 val;
                PyArray_ScalarAsCtype(value, &val);
                node = val;
                break;
            }
            case NPY_FLOAT64 : {
                float64 val;
                PyArray_ScalarAsCtype(value, &val);
                node = val;
                break;
            }
        }

    } else {
        PyErr_SetString(PyExc_TypeError, "Value type not supported");
        return (-1);
    }

    return (0);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_createNumpyType(Node& node,
                          int type)
{
    const DataType& dtype = node.dtype();
    PyArray_Descr* descr = PyArray_DescrFromType(type);
    PyObject* retval = NULL;
    void* data = node.as_char8_str();
    npy_intp len = dtype.number_of_elements();
    if (len == 1) {
        retval = PyArray_Scalar(data, descr, NULL);
    } else {
        retval = PyArray_SimpleNewFromData(1, &len, type, data);
    }
    return (retval);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_convertNodeToPython(Node& node)
{
    const DataType& type = node.dtype();
    int numpy_type = -1;
    PyObject* retval;

    switch (type.id()) {
        case DataType::EMPTY_T :
        case DataType::OBJECT_T : {
            retval = PyConduitNode_getObject(&node);
            break;
        }
        case DataType::CHAR8_STR_T : {
            retval = PyString_FromString(node.as_char8_str());
            break;
        }
        case DataType::INT8_T : {
            numpy_type = NPY_INT8;
            break;
        }
        case DataType::INT16_T : {
            numpy_type = NPY_INT16;
            break;
        }
        case DataType::INT32_T : {
            numpy_type = NPY_INT32;
            break;
        }
        case DataType::INT64_T : {
            numpy_type = NPY_INT64;
            break;
        }
        case DataType::UINT8_T : {
            numpy_type = NPY_UINT8;
            break;
        }
        case DataType::UINT16_T : {
            numpy_type = NPY_UINT16;
            break;
        }
        case DataType::UINT32_T : {
            numpy_type = NPY_UINT32;
            break;
        }
        case DataType::UINT64_T : {
            numpy_type = NPY_UINT64;
            break;
        }
        case DataType::FLOAT32_T : {
            numpy_type = NPY_FLOAT32;
            break;
        }
        case DataType::FLOAT64_T : {
            numpy_type = NPY_FLOAT64;
            break;
        }
        default : {
            PyErr_SetString(PyExc_TypeError, "Python Conduit does not support this type");
            return (NULL);
        }
    }

    if (type.id() != DataType::OBJECT_T &&
        type.id() != DataType::CHAR8_STR_T) {

        retval = PyConduit_createNumpyType(node, numpy_type);
    }

    return (retval);
}
