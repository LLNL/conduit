//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

//---------------------------------------------------------------------------//
// conduit::relay::io::blueprint::save_mesh
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_io_blueprint_save_mesh(PyObject *, //self
                               PyObject *args,
                               PyObject *kwargs)
{
    std::cout << "PyRelay_io_blueprint_save_mesh" << std::endl;
    PyObject   *py_node    = NULL;
    const char *path       = NULL;
    const char *protocol   = NULL;
    PyObject   *py_options = NULL;

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
                                     &py_options))
    {
        return (NULL);
    }
    
    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "relay::io::blueprint::save_mesh "
                        "'node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if( (py_options != NULL) && !PyConduit_Node_Check(py_options) )
    {
        PyErr_SetString(PyExc_TypeError,
                        "relay::io::blueprint::save_mesh "
                        "'options' argument must "
                        "be a conduit.Node");
        return NULL;
    }

    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    Node opts;
    Node *opts_ptr = &opts;

    if(py_options != NULL)
    {
        opts_ptr = PyConduit_Node_Get_Node_Ptr(py_options);
    }

    // default protocol string is empty which auto detects
    std::string protocol_str("");

    if(protocol != NULL)
    {
        protocol_str = std::string(protocol);
    }
    
    try
    {
        relay::io::blueprint::save_mesh(node,
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
// conduit::relay::io::blueprint::load_mesh
//---------------------------------------------------------------------------//

static PyObject * 
PyRelay_io_blueprint_load_mesh(PyObject *, //self
                               PyObject *args,
                               PyObject *kwargs)
{
    PyObject   *py_node    = NULL;
    const char *path       = NULL;
    PyObject   *py_options = NULL;
    
    static const char *kwlist[] = {"node",
                                   "path",
                                   "options",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Os|O",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &path,
                                     &py_options))
    {
        return (NULL);
    }
    
    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "relay::io::blueprint::load_mesh "
                        "'node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if( (py_options != NULL) && !PyConduit_Node_Check(py_options) )
    {
        PyErr_SetString(PyExc_TypeError,
                        "relay::io::blueprint::load_mesh "
                        "'options' argument must "
                        "be a conduit.Node");
        return NULL;
    }

    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    Node opts;
    Node *opts_ptr = &opts;

    if(py_options != NULL)
    {
        opts_ptr = PyConduit_Node_Get_Node_Ptr(py_options);
    }


    try
    {
        relay::io::blueprint::load_mesh(std::string(path),
                                        *opts_ptr,
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
static PyMethodDef relay_io_blueprint_python_funcs[] =
{
    //-----------------------------------------------------------------------//
    //-----------------------------------------------------------------------//
    {"save_mesh",
     (PyCFunction)PyRelay_io_blueprint_save_mesh,
      METH_VARARGS | METH_KEYWORDS,
      NULL},
    {"load_mesh",
     (PyCFunction)PyRelay_io_blueprint_load_mesh,
      METH_VARARGS | METH_KEYWORDS,
      NULL},
    //-----------------------------------------------------------------------//
    // end relay io blueprint methods table
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
relay_io_blueprint_python_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static int 
relay_io_blueprint_python_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static struct PyModuleDef relay_io_blueprint_python_module_def = 
{
        PyModuleDef_HEAD_INIT,
        "relay_io_python_blueprint",
        NULL,
        sizeof(struct module_state),
        relay_io_blueprint_python_funcs,
        NULL,
        relay_io_blueprint_python_traverse,
        relay_io_blueprint_python_clear,
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
CONDUIT_RELAY_PYTHON_API PyObject * PyInit_conduit_relay_io_blueprint_python(void)
#else
CONDUIT_RELAY_PYTHON_API void initconduit_relay_io_blueprint_python(void)
#endif
//---------------------------------------------------------------------------//
{    
    //-----------------------------------------------------------------------//
    // create our main module
    //-----------------------------------------------------------------------//

#if defined(IS_PY3K)
    PyObject *res_mod = PyModule_Create(&relay_io_blueprint_python_module_def);
#else
    PyObject *res_mod = Py_InitModule((char*)"conduit_relay_io_blueprint_python",
                                      relay_io_blueprint_python_funcs);
#endif


    if(res_mod == NULL)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    struct module_state *st = GETSTATE(res_mod);
    
    st->error = PyErr_NewException((char*)"relay_io_blueprint_python.Error",
                                   NULL,
                                   NULL);
    if (st->error == NULL)
    {
        Py_DECREF(res_mod);
        PY_MODULE_INIT_RETURN_ERROR;
    }

    // setup for conduit python c api
    if(import_conduit() < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

#if defined(IS_PY3K)
    return res_mod;
#endif

}

