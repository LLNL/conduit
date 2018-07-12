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
                            "Conduit::Node");
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
                            "Conduit::Node");
            return NULL;
        }
        
        opts_ptr = PyConduit_Node_Get_Node_Ptr(py_opts);
    }
    
    
    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);
    std::string protocol_str("");
    
    // default protocol string is empty which auto detects
    if(protocol != NULL)
    {
        protocol_str = std::string(protocol);
    }
    
    relay::io::save(node,
                    std::string(path),
                    protocol_str,
                    *opts_ptr);


    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::io::save
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
                            "Conduit::Node");
            return NULL;
        }
    }
    
    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);
    
    if(protocol != NULL)
    {
        relay::io::load(std::string(path),std::string(protocol),node);
    }
    else
    {
        relay::io::load(std::string(path),node);
    }

    Py_RETURN_NONE;
}


//---------------------------------------------------------------------------//
// Python Module Method Defs
//---------------------------------------------------------------------------//
static PyMethodDef relay_io_python_funcs[] =
{
    //-----------------------------------------------------------------------//
    {"save",
     (PyCFunction)PyRelay_io_save,
      METH_VARARGS | METH_KEYWORDS,
      NULL},
    {"load",
     (PyCFunction)PyRelay_io_load,
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


#if defined(IS_PY3K)
    return relay_io_module;
#endif

}

