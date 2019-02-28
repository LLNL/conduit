//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
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


//-----------------------------------------------------------------------------
// -- Python includes (these must be included first) -- 
//-----------------------------------------------------------------------------
#include <Python.h>
#include <structmember.h>
#include "bytesobject.h"

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------//
// conduit includes
//---------------------------------------------------------------------------//
#include "conduit.hpp"
#include "conduit_relay.hpp"

#include "conduit_relay_python_exports.h"

// conduit python module capi header
#include "conduit_python.hpp"


using namespace conduit;
using namespace conduit::relay::io;

//-----------------------------------------------------------------------------
// PyVarObject_TAIL is used at the end of each PyVarObject def
// to make sure we have the correct number of initializers across python
// versions.
//-----------------------------------------------------------------------------
#ifdef Py_TPFLAGS_HAVE_FINALIZE
#define PyVarObject_TAIL ,0
#else
#define PyVarObject_TAIL
#endif

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Begin Functions to help with Python 2/3 Compatibility.
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


#if defined(IS_PY3K)

//-----------------------------------------------------------------------------
int
PyString_Check(PyObject *o)
{
    return PyUnicode_Check(o);
}

//-----------------------------------------------------------------------------
char *
PyString_AsString(PyObject *py_obj)
{
    char *res = NULL;
    if(PyUnicode_Check(py_obj))
    {
        PyObject * temp_bytes = PyUnicode_AsEncodedString(py_obj,
                                                          "ASCII",
                                                          "strict"); // Owned reference
        if(temp_bytes != NULL)
        {
            res = strdup(PyBytes_AS_STRING(temp_bytes));
            Py_DECREF(temp_bytes);
        }
        else
        {
            // TODO: Error
        }
    }
    else if(PyBytes_Check(py_obj))
    {
        res = strdup(PyBytes_AS_STRING(py_obj));
    }
    else
    {
        // TODO: ERROR or auto convert?
    }
    
    return res;
}

//-----------------------------------------------------------------------------
PyObject *
PyString_FromString(const char *s)
{
    return PyUnicode_FromString(s);
}

//-----------------------------------------------------------------------------
void
PyString_AsString_Cleanup(char *bytes)
{
    free(bytes);
}


//-----------------------------------------------------------------------------
int
PyInt_Check(PyObject *o)
{
    return PyLong_Check(o);
}

//-----------------------------------------------------------------------------
long
PyInt_AsLong(PyObject *o)
{
    return PyLong_AsLong(o);
}

#else // python 2.6+

//-----------------------------------------------------------------------------
#define PyString_AsString_Cleanup(c) { /* noop */ }

#endif

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// End Functions to help with Python 2/3 Compatibility.
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
struct PyRelay_IOHandle
{
    PyObject_HEAD
    IOHandle *handle;
};


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// IOHandle Object 
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_IOHandle_new(PyTypeObject *type,
                     PyObject*, // args -- unused
                     PyObject*) // kwds -- unused
{
    PyRelay_IOHandle *self = (PyRelay_IOHandle*)type->tp_alloc(type, 0);

    if (self)
    {
        self->handle = 0;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static void
PyRelay_IOHandle_dealloc(PyRelay_IOHandle *self)
{
    if(self->handle != NULL)
    {
        delete self->handle;
    }
    
    Py_TYPE(self)->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------//
static int
PyRelay_IOHandle_init(PyRelay_IOHandle *self,
                     PyObject*, // args -- unused
                     PyObject*) // kwds -- unused
{
    self->handle = new IOHandle();
    return 0;
}

//-----------------------------------------------------------------------------
static PyObject *
PyRelay_IOHandle_open(PyRelay_IOHandle *self,
                      PyObject *args,
                      PyObject *kwargs)
{

    static const char *kwlist[] = {"path",
                                   "protocol",
                                   "options",
                                   NULL};

    char     *path       = NULL;
    char     *protocol   = NULL;
    PyObject *py_options = NULL;
    
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s|sO",
                                     const_cast<char**>(kwlist),
                                     &path,
                                     &protocol,
                                     &py_options))
    {
        return NULL;
    }
    
    
    if( (py_options != NULL) && !PyConduit_Node_Check(py_options) )
    {
        PyErr_SetString(PyExc_TypeError,
                        "IOHandle.open 'options' argument must "
                        "be a conduit.Node");
        return NULL;
    }

    Node opts;
    Node *opts_ptr = &opts;

    if(py_options != NULL)
    {
        opts_ptr = PyConduit_Node_Get_Node_Ptr(py_options);
    }
    
    std::string protocol_str;

    if(protocol != NULL)
    {
        protocol_str = std::string(protocol);
    }
    
    // TODO: TRY CATCH
    self->handle->open(std::string(path),
                       protocol_str,
                       *opts_ptr);

    Py_RETURN_NONE; 
}

//-----------------------------------------------------------------------------
static PyObject *
PyRelay_IOHandle_read(PyRelay_IOHandle *self,
                       PyObject *args,
                       PyObject *kwargs)
{

    static const char *kwlist[] = {"node",
                                   "path",
                                   NULL};

    PyObject *py_node = NULL;
    char     *path    = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "O|s",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &path))
    {
        return NULL;
    }
    
    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "IOHandle.read 'node' argument must "
                        "be a conduit.Node");
        return NULL;
    }

    Node *node_ptr = PyConduit_Node_Get_Node_Ptr(py_node);
    
    // TODO: TRY CATCH
    
    if(path == NULL)
    {
        self->handle->read(*node_ptr);
    }
    else
    {
        self->handle->read(std::string(path),*node_ptr);
    }

    Py_RETURN_NONE; 
}

//-----------------------------------------------------------------------------
static PyObject *
PyRelay_IOHandle_write(PyRelay_IOHandle *self,
                       PyObject *args,
                       PyObject *kwargs)
{

    static const char *kwlist[] = {"node",
                                   "path",
                                   NULL};

    PyObject *py_node = NULL;
    char     *path    = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "O|s",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &path))
    {
        return NULL;
    }
    
    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "IOHandle.write 'node' argument must "
                        "be a conduit.Node");
        return NULL;
    }

    Node *node_ptr = PyConduit_Node_Get_Node_Ptr(py_node);
    
    // TODO: TRY CATCH
    
    if(path == NULL)
    {
        self->handle->write(*node_ptr);
    }
    else
    {
        self->handle->write(*node_ptr,std::string(path));
    }

    Py_RETURN_NONE; 
}

//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_IOHandle_list_child_names(PyRelay_IOHandle *self,
                                  PyObject *args,
                                  PyObject *kwargs)
{
    static const char *kwlist[] = {"path", NULL};
    char *path = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "|s",
                                     const_cast<char**>(kwlist),
                                     &path))
    {
        return NULL;
    }
    
    std::vector<std::string> cld_names;
    
    if(path == NULL)
    {
        self->handle->list_child_names(cld_names);
    }
    else
    {
        self->handle->list_child_names(std::string(path),
                                       cld_names);
    }

    /// TODO: I think there is a faster way in the Python CAPI
    /// since we know the size of the list.
    PyObject *retval = PyList_New(0);
    
    for (std::vector<std::string>::const_iterator itr = cld_names.begin();
         itr < cld_names.end(); ++itr)
    {
        PyList_Append(retval, PyString_FromString( (*itr).c_str()));
    }

    return retval;
}

//-----------------------------------------------------------------------------
static PyObject *
PyRelay_IOHandle_has_path(PyRelay_IOHandle *self,
                          PyObject *args,
                          PyObject *kwargs)
{

    static const char *kwlist[] = {"path", NULL};

    char *path = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s",
                                     const_cast<char**>(kwlist),
                                     &path))
    {
        return NULL;
    }


    bool res = self->handle->has_path(std::string(path));

    if(res)
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

//-----------------------------------------------------------------------------
static PyObject *
PyRelay_IOHandle_remove(PyRelay_IOHandle *self,
                        PyObject *args,
                        PyObject *kwargs)
{

    static const char *kwlist[] = {"path", NULL};

    char *path = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s",
                                     const_cast<char**>(kwlist),
                                     &path))
    {
        return NULL;
    }

    self->handle->remove(std::string(path));
    Py_RETURN_NONE; 
}

//-----------------------------------------------------------------------------
static PyObject *
PyRelay_IOHandle_close(PyRelay_IOHandle *self)
{
    self->handle->close();
    Py_RETURN_NONE; 
}


//----------------------------------------------------------------------------//
// IOHandle methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyRelay_IOHandle_METHODS[] = {
    //-----------------------------------------------------------------------//
    {"open",
     (PyCFunction)PyRelay_IOHandle_open,
     METH_VARARGS | METH_KEYWORDS,
     "Opens a Relay IO Handle"},
    {"read",
     (PyCFunction)PyRelay_IOHandle_read,
     METH_VARARGS | METH_KEYWORDS,
     "Reads from an active Relay IO Handle"},
    {"write",
     (PyCFunction)PyRelay_IOHandle_write,
     METH_VARARGS | METH_KEYWORDS,
     "Writes to an active Relay IO Handle"},
    {"list_child_names",
     (PyCFunction)PyRelay_IOHandle_list_child_names,
     METH_VARARGS | METH_KEYWORDS,
     "Returns a list of child names"},
    {"remove",
     (PyCFunction)PyRelay_IOHandle_remove,
     METH_VARARGS | METH_KEYWORDS,
     "Removes a path"},
    {"has_path",
     (PyCFunction)PyRelay_IOHandle_has_path,
     METH_VARARGS | METH_KEYWORDS,
     "Checks if a path exists"},
    {"close",
     (PyCFunction)PyRelay_IOHandle_close,
      METH_NOARGS,
     "Closes an active Relay IO Handle"},
    //-----------------------------------------------------------------------//
    // end Generator methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
static PyTypeObject PyRelay_IOHandle_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "IOHandle",
   sizeof(PyRelay_IOHandle),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyRelay_IOHandle_dealloc,   /* tp_dealloc */
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
   0, // (reprfunc)PyRelay_IOHandle_str,                         /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "Conduit Relay IO Handle objects",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyRelay_IOHandle_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyRelay_IOHandle_init, /* init */
   0, /* alloc */
   PyRelay_IOHandle_new,   /* new */
   0, /* tp_free */
   0, /* tp_is_gc */
   0, /* tp_bases */
   0, /* tp_mro */
   0, /* tp_cache */
   0, /* tp_subclasses */
   0,  /* tp_weaklist */
   0,
   0
   PyVarObject_TAIL
};


//---------------------------------------------------------------------------//
// conduit::relay::io::about
//---------------------------------------------------------------------------//
static PyObject *
PyRelay_io_about()
{
    //create and return a node with the result of about
    PyObject *py_node_res = PyConduit_Node_Python_Create();
    Node *node = PyConduit_Node_Get_Node_Ptr(py_node_res);
    conduit::relay::io::about(*node);
    return (PyObject*)py_node_res;
}

//---------------------------------------------------------------------------//
// conduit::relay::io::save
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_io_save(PyObject *, //self
                PyObject *args,
                PyObject *kwargs)
{
    PyObject   *py_node  = NULL;
    const char *path     = NULL;
    const char *protocol = NULL;
    PyObject   *py_opts  = NULL;
    
    static const char *kwlist[] = {"node",
                                   "path",
                                   "protocol",
                                   "options",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Os|sO",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &path,
                                     &protocol,
                                     &py_opts))
    {
        return (NULL);
    }
    
    if(py_node != NULL)
    {
        if(!PyConduit_Node_Check(py_node))
        {
            PyErr_SetString(PyExc_TypeError,
                            "relay::save 'node' argument must be a "
                            "conduit.Node instance");
            return NULL;
        }
    }
    
    // default opts is an empty node which is ignored
    Node opts;
    Node *opts_ptr = &opts;
    if(py_opts != NULL)
    {
        if(!PyConduit_Node_Check(py_node))
        {
            PyErr_SetString(PyExc_TypeError,
                            "relay::save 'options' argument must be a "
                            "conduit.Node instance");
            return NULL;
        }
        
        opts_ptr = PyConduit_Node_Get_Node_Ptr(py_opts);
    }
    
    
    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    // default protocol string is empty which auto detects
    std::string protocol_str("");

    if(protocol != NULL)
    {
        protocol_str = std::string(protocol);
    }
    
    try
    {
    
        relay::io::save(node,
                        std::string(path),
                        protocol_str,
                        *opts_ptr);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_IOError,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::io::save_merged
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_io_save_merged(PyObject *, //self
                       PyObject *args,
                       PyObject *kwargs)
{
    PyObject   *py_node  = NULL;
    const char *path     = NULL;
    const char *protocol = NULL;
    PyObject   *py_opts  = NULL;
    
    static const char *kwlist[] = {"node",
                                   "path",
                                   "protocol",
                                   "options",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Os|sO",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &path,
                                     &protocol,
                                     &py_opts))
    {
        return (NULL);
    }
    
    if(py_node != NULL)
    {
        if(!PyConduit_Node_Check(py_node))
        {
            PyErr_SetString(PyExc_TypeError,
                            "relay::save 'node' argument must be a "
                            "conduit.Node instance");
            return NULL;
        }
    }
    
    // default opts is an empty node which is ignored
    Node opts;
    Node *opts_ptr = &opts;
    if(py_opts != NULL)
    {
        if(!PyConduit_Node_Check(py_node))
        {
            PyErr_SetString(PyExc_TypeError,
                            "relay::save_merged 'options' argument must be a "
                            "conduit.Node instance");
            return NULL;
        }
        
        opts_ptr = PyConduit_Node_Get_Node_Ptr(py_opts);
    }
    
    
    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    // default protocol string is empty which auto detects
    std::string protocol_str("");    
    if(protocol != NULL)
    {
        protocol_str = std::string(protocol);
    }
    
    try
    {
        relay::io::save_merged(node,
                               std::string(path),
                               protocol_str,
                               *opts_ptr);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_IOError,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::io::load
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_io_load(PyObject *, //self
                PyObject *args,
                PyObject *kwargs)
{
    PyObject   *py_node  = NULL;
    const char *path     = NULL;
    const char *protocol = NULL;
    
    static const char *kwlist[] = {"node","path","protocol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Os|s",
                                     const_cast<char**>(kwlist),
                                     &py_node, &path, &protocol))
    {
        return (NULL);
    }
    
    if(py_node != NULL)
    {
        if(!PyConduit_Node_Check(py_node))
        {
            PyErr_SetString(PyExc_TypeError,
                            "relay::load 'node' argument must be a "
                            "conduit.Node instance");
            return NULL;
        }
    }

    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);
    // default protocol string is empty which auto detects
    std::string protocol_str("");
    
    if(protocol != NULL)
    {
        protocol_str = std::string(protocol);
    }
    
    
    try
    {
        relay::io::load(std::string(path),
                        protocol_str,
                        node);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_IOError,
                        e.message().c_str());
        return NULL;
    }
    
    Py_RETURN_NONE;
}




//---------------------------------------------------------------------------//
// conduit::relay::io::load_merged
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_io_load_merged(PyObject *, //self
                       PyObject *args,
                       PyObject *kwargs)
{
    PyObject   *py_node  = NULL;
    const char *path     = NULL;
    const char *protocol = NULL;
    
    static const char *kwlist[] = {"node","path","protocol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Os|s",
                                     const_cast<char**>(kwlist),
                                     &py_node, &path, &protocol))
    {
        return (NULL);
    }
    
    if(py_node != NULL)
    {
        if(!PyConduit_Node_Check(py_node))
        {
            PyErr_SetString(PyExc_TypeError,
                            "relay::load 'node' argument must be a "
                            "conduit.Node instance");
            return NULL;
        }
    }

    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);
    // default protocol string is empty which auto detects
    std::string protocol_str("");
    
    if(protocol != NULL)
    {
        protocol_str = std::string(protocol);
    }
    
    try
    {
        relay::io::load_merged(std::string(path),
                               protocol_str,
                               node);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_IOError,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}
//---------------------------------------------------------------------------//
// Python Module Method Defs
//---------------------------------------------------------------------------//
static PyMethodDef relay_io_python_funcs[] =
{
    //-----------------------------------------------------------------------//
    //-----------------------------------------------------------------------//
    {"about",
     (PyCFunction)PyRelay_io_about,
      METH_NOARGS,
      NULL},
    {"save",
     (PyCFunction)PyRelay_io_save,
      METH_VARARGS | METH_KEYWORDS,
      NULL},
    {"load",
     (PyCFunction)PyRelay_io_load,
      METH_VARARGS | METH_KEYWORDS,
      NULL},
    {"save_merged",
     (PyCFunction)PyRelay_io_save_merged,
      METH_VARARGS | METH_KEYWORDS,
      NULL},
    {"load_merged",
     (PyCFunction)PyRelay_io_load_merged,
      METH_VARARGS | METH_KEYWORDS,
      NULL},
    //-----------------------------------------------------------------------//
    // end relay io methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, METH_VARARGS, NULL}
};

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// Module Init Code
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

struct module_state {
    PyObject *error;
};

//---------------------------------------------------------------------------//
#if defined(IS_PY3K)
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Extra Module Setup Logic for Python3
//---------------------------------------------------------------------------//
#if defined(IS_PY3K)
//---------------------------------------------------------------------------//
static int
relay_io_python_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static int 
relay_io_python_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static struct PyModuleDef relay_io_python_module_def = 
{
        PyModuleDef_HEAD_INIT,
        "relay_io_python",
        NULL,
        sizeof(struct module_state),
        relay_io_python_funcs,
        NULL,
        relay_io_python_traverse,
        relay_io_python_clear,
        NULL
};



#endif

//---------------------------------------------------------------------------//
// The module init function signature is different between py2 and py3
// This macro simplifies the process of returning when an init error occurs.
//---------------------------------------------------------------------------//
#if defined(IS_PY3K)
#define PY_MODULE_INIT_RETURN_ERROR return NULL
#else
#define PY_MODULE_INIT_RETURN_ERROR return
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// Main entry point
//---------------------------------------------------------------------------//
extern "C" 
//---------------------------------------------------------------------------//
#if defined(IS_PY3K)
PyObject *CONDUIT_RELAY_PYTHON_API PyInit_conduit_relay_io_python(void)
#else
void CONDUIT_RELAY_PYTHON_API initconduit_relay_io_python(void)
#endif
//---------------------------------------------------------------------------//
{    
    //-----------------------------------------------------------------------//
    // create our main module
    //-----------------------------------------------------------------------//

#if defined(IS_PY3K)
    PyObject *relay_io_module = PyModule_Create(&relay_io_python_module_def);
#else
    PyObject *relay_io_module = Py_InitModule((char*)"conduit_relay_io_python",
                                              relay_io_python_funcs);
#endif


    if(relay_io_module == NULL)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    struct module_state *st = GETSTATE(relay_io_module);
    
    st->error = PyErr_NewException((char*)"relay_io_python.Error",
                                   NULL,
                                   NULL);
    if (st->error == NULL)
    {
        Py_DECREF(relay_io_module);
        PY_MODULE_INIT_RETURN_ERROR;
    }

    // setup for conduit python c api
    if(import_conduit() < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }
    
    //-----------------------------------------------------------------------//
    // init our custom types
    //-----------------------------------------------------------------------//

    if (PyType_Ready(&PyRelay_IOHandle_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    //-----------------------------------------------------------------------//
    // add IOHandle
    //-----------------------------------------------------------------------//

    Py_INCREF(&PyRelay_IOHandle_TYPE);
    PyModule_AddObject(relay_io_module,
                       "IOHandle",
                       (PyObject*)&PyRelay_IOHandle_TYPE);

#if defined(IS_PY3K)
    return relay_io_module;
#endif

}

