//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------//
// conduit includes
//---------------------------------------------------------------------------//
#include "conduit.hpp"
#include "relay.hpp"

#include "Relay_Python_Exports.hpp"

// conduit python module capi header
#include "conduit_python.hpp"


using namespace conduit;
using namespace conduit::relay::web;

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
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

//---------------------------------------------------------------------------//
struct PyRelay_Web_WebServer
{
    PyObject_HEAD
    WebServer *webserver;
};


//---------------------------------------------------------------------------//
struct PyRelay_Web_WebSocket
{
    PyObject_HEAD
    WebSocket *websocket;
};


//---------------------------------------------------------------------------//
static int       PyRelay_Web_WebServer_Check(PyObject* obj);


//---------------------------------------------------------------------------//
static PyObject* PyRelay_Web_WebSocket_python_wrap(WebSocket *websocket);

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// End Functions to help with Python 2/3 Compatibility.
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// relay::web::WebServer Object 
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_Web_WebServer_new(PyTypeObject *type,
                          PyObject*, // args -- unused
                          PyObject*) // kwds -- unused
{
    PyRelay_Web_WebServer *self = (PyRelay_Web_WebServer*)type->tp_alloc(type, 0);

    if (self)
    {
        self->webserver = 0;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static void
PyRelay_Web_WebServer_dealloc(PyRelay_Web_WebServer *self)
{
    if(self->webserver != NULL)
    {
        delete self->webserver;
    }
    
    Py_TYPE(self)->tp_free((PyObject*)self);
}


//---------------------------------------------------------------------------//
static int
PyRelay_Web_WebServer_init(PyRelay_Web_WebServer *self)
{
    
    self->webserver = new WebServer();

    return 0;
}



//-----------------------------------------------------------------------------
static PyObject *
PyRelay_Web_WebServer_serve(PyRelay_Web_WebServer *self,
                            PyObject *args,
                            PyObject *kwargs)
{

    static const char *kwlist[] = {"doc_root",
                                   "port",
                                   "ssl_cert_file",
                                   "auth_domain",
                                   "auth_file",
                                    NULL};

    const char *doc_root_c_str;
    Py_ssize_t port = 0;
    const char *ssl_cert_file_c_str;
    const char *auth_domain_c_str;
    const char *auth_file_c_str;
  
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s|nsss",
                                     const_cast<char**>(kwlist),
                                     &doc_root_c_str,
                                     &port,
                                     &ssl_cert_file_c_str,
                                     &auth_domain_c_str,
                                     &auth_file_c_str))
    {
        return NULL;
    }
    

    std::string doc_root(doc_root_c_str);
   
    if(port == 0)
    {
        port = 8080;
    }

    std::string ssl_cert_file;
    if(ssl_cert_file_c_str != NULL)
    {
        ssl_cert_file = std::string(ssl_cert_file_c_str);
    }
        
    std::string auth_domain;

    if(auth_domain_c_str != NULL)
    {
        auth_domain = std::string(auth_domain_c_str);
    }

    std::string auth_file;
    if(auth_file_c_str != NULL)
    {
        auth_file = std::string(auth_file_c_str);
    }

    self->webserver->serve(doc_root,
                           port,
                           ssl_cert_file,
                           auth_domain,
                           auth_file);
    
    
    
    Py_RETURN_NONE; 
}



//-----------------------------------------------------------------------------
static PyObject *
PyRelay_Web_WebServer_shutdown(PyRelay_Web_WebServer *self)
{
    self->webserver->shutdown();
    
    Py_RETURN_NONE; 
}

//-----------------------------------------------------------------------------
static PyObject *
PyRelay_Web_WebServer_is_running(PyRelay_Web_WebServer *self)
{
    if(self->webserver->is_running())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}


//-----------------------------------------------------------------------------
static PyObject *
PyRelay_Web_WebServer_websocket(PyRelay_Web_WebServer *self,
                                PyObject *args,
                                PyObject *kwargs)
{

    static const char *kwlist[] = {"ms_poll",
                                   "ms_timeout",
                                   NULL};
    
    Py_ssize_t ms_poll    = 100;
    Py_ssize_t ms_timeout = 60000;
    
    if(!PyArg_ParseTupleAndKeywords(args,
                                    kwargs,
                                    "|nn",
                                    const_cast<char**>(kwlist),
                                    &ms_poll,
                                    &ms_timeout))
    {
        return NULL;
    }
    
    
    WebSocket *websocket = self->webserver->websocket(ms_poll,ms_timeout);
    
    PyObject *py_websocket = PyRelay_Web_WebSocket_python_wrap(websocket);

    return py_websocket;
}



//----------------------------------------------------------------------------//
// WebServer methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyRelay_Web_WebServer_METHODS[] = {
    //-----------------------------------------------------------------------//
    {"serve",
     (PyCFunction)PyRelay_Web_WebServer_serve,
     METH_NOARGS,
     "Start the web server."},
    {"shutdown",
     (PyCFunction)PyRelay_Web_WebServer_shutdown,
     METH_NOARGS,
     "Shutdown the web server."},
    {"is_running",
     (PyCFunction)PyRelay_Web_WebServer_is_running,
     METH_NOARGS,
     "Returns if the web server is running."},
    {"shutdown",
     (PyCFunction)PyRelay_Web_WebServer_shutdown,
     METH_VARARGS| METH_KEYWORDS,
     "Obtain connected web socket connection."},
    //-----------------------------------------------------------------------//
    // end WebServer methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
static PyTypeObject PyRelay_Web_WebServer_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "WebServer",
   sizeof(PyRelay_Web_WebServer),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyRelay_Web_WebServer_dealloc, /* tp_dealloc */
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
   0, /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "Conduit Relay WebServer objects",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyRelay_Web_WebServer_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyRelay_Web_WebServer_init,
   0, /* alloc */
   PyRelay_Web_WebServer_new,                       /* new */
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
static int
PyRelay_Web_WebServer_Check(PyObject* obj)
{
    return (PyObject_TypeCheck(obj, &PyRelay_Web_WebServer_TYPE));
}


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// relay::web::WebSocket Object 
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_Web_WebSocket_new(PyTypeObject *type,
                          PyObject*, // args -- unused
                          PyObject*) // kwds -- unused
{
    PyRelay_Web_WebSocket *self = (PyRelay_Web_WebSocket*)type->tp_alloc(type, 0);

    if (self)
    {
        self->websocket = NULL;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static void
PyRelay_Web_WebSocket_dealloc(PyRelay_Web_WebSocket *self)
{
    // cpp "socket" pointer is owned by the parent web server, no
    // need to clean them up from python
    Py_TYPE(self)->tp_free((PyObject*)self);
}


//---------------------------------------------------------------------------//
static int
PyRelay_Web_WebSocket_init(PyRelay_Web_WebSocket *self)
{
    self->websocket = NULL;
    return 0;
}

//-----------------------------------------------------------------------------
static PyObject *
PyRelay_Web_WebSocket_send(PyRelay_Web_WebSocket *self,
                           PyObject *args,
                           PyObject *kwargs)
{

    static const char *kwlist[] = {"data",
                                   "protocol",
                                    NULL};

    PyObject *py_node;
    const char *protocol_c_str;
    Py_ssize_t port = 0;
    const char *ssl_cert_file_c_str;
    const char *auth_domain_c_str;
    const char *auth_file_c_str;
  
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "O|s",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &protocol_c_str))
    {
        return NULL;
    }
    
    
    if(py_node != NULL)
    {
        if(!PyConduit_Node_Check(py_node))
        {
            PyErr_SetString(PyExc_TypeError,
                            "relay::WebSocket::send 'data' argument must be a "
                            "Conduit::Node");
            return NULL;
        }
    }
    
    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    std::string protocol = "json";
    if(protocol_c_str != NULL)
    {
        protocol = std::string(protocol_c_str);
    }
        
    self->websocket->send(node,protocol);
        
    Py_RETURN_NONE; 
}


//-----------------------------------------------------------------------------
static PyObject *
PyRelay_Web_WebSocket_is_connected(PyRelay_Web_WebSocket *self)
{
    if(self->websocket->is_connected())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}


//----------------------------------------------------------------------------//
// WebSocket methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyRelay_Web_WebSocket_METHODS[] = {
    //-----------------------------------------------------------------------//
    {"send",
     (PyCFunction)PyRelay_Web_WebSocket_send,
     METH_NOARGS,
     "Send a conduit node over the web socket."},
    {"is_connected",
     (PyCFunction)PyRelay_Web_WebSocket_is_connected,
     METH_NOARGS,
     "Returns if the web socket is connected."},
    //-----------------------------------------------------------------------//
    // end WebSocket methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};



//---------------------------------------------------------------------------//
static PyTypeObject PyRelay_Web_WebSocket_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "WebSocket",
   sizeof(PyRelay_Web_WebSocket),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyRelay_Web_WebSocket_dealloc, /* tp_dealloc */
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
   0, /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "Conduit Relay WebSocket objects",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyRelay_Web_WebSocket_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyRelay_Web_WebSocket_init,
   0, /* alloc */
   PyRelay_Web_WebSocket_new,                       /* new */
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
static int
PyRelay_Web_WebSocket_Check(PyObject* obj)
{
    return (PyObject_TypeCheck(obj, &PyRelay_Web_WebSocket_TYPE));
}


//---------------------------------------------------------------------------//
static PyObject *
PyRelay_Web_WebSocket_python_wrap(WebSocket *websocket)
{
    PyTypeObject *type = (PyTypeObject*)&PyRelay_Web_WebSocket_TYPE;

    PyRelay_Web_WebSocket *retval = (PyRelay_Web_WebSocket*)type->tp_alloc(type, 0);
    retval->websocket = websocket;
    return ((PyObject*)retval);
}


//---------------------------------------------------------------------------//
// Python Module Method Defs
//---------------------------------------------------------------------------//
static PyMethodDef relay_web_python_funcs[] =
{
    //-----------------------------------------------------------------------//
    // end relay web methods table
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
relay_web_python_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static int 
relay_web_python_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static struct PyModuleDef relay_web_python_module_def = 
{
        PyModuleDef_HEAD_INIT,
        "relay_web_python",
        NULL,
        sizeof(struct module_state),
        relay_web_python_funcs,
        NULL,
        relay_web_python_traverse,
        relay_web_python_clear,
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
PyObject *RELAY_PYTHON_API PyInit_relay_web_python(void)
#else
void RELAY_PYTHON_API initrelay_web_python(void)
#endif
//---------------------------------------------------------------------------//
{    
    //-----------------------------------------------------------------------//
    // create our main module
    //-----------------------------------------------------------------------//

#if defined(IS_PY3K)
    PyObject *relay_web_module = PyModule_Create(&relay_web_python_module_def);
#else
    PyObject *relay_web_module = Py_InitModule((char*)"relay_web_python",
                                               relay_web_python_funcs);
#endif


    if(relay_web_module == NULL)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    struct module_state *st = GETSTATE(relay_web_module);
    
    st->error = PyErr_NewException((char*)"relay_web_python.Error",
                                   NULL,
                                   NULL);
    if (st->error == NULL)
    {
        Py_DECREF(relay_web_module);
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

    if (PyType_Ready(&PyRelay_Web_WebServer_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }
    //-----------------------------------------------------------------------//
    // add WebServer
    //-----------------------------------------------------------------------//
    
    Py_INCREF(&PyRelay_Web_WebServer_TYPE);
    PyModule_AddObject(relay_web_module,
                       "WebServer",
                       (PyObject*)&PyRelay_Web_WebServer_TYPE);

#if defined(IS_PY3K)
    return relay_web_module;
#endif

}

