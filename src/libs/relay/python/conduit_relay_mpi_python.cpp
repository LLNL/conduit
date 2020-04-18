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

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------//
// conduit includes
//---------------------------------------------------------------------------//
#include "conduit.hpp"
#include "conduit_relay_mpi.hpp"

#include "conduit_relay_python_exports.h"

// conduit python module capi header
#include "conduit_python.hpp"


using namespace conduit;
using namespace conduit::relay::mpi;

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

// use  proper strdup
#ifdef CONDUIT_PLATFORM_WINDOWS
    #define _conduit_strdup _strdup
#else
    #define _conduit_strdup strdup
#endif

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
static int
PyString_Check(PyObject *o)
{
    return PyUnicode_Check(o);
}

//-----------------------------------------------------------------------------
static char *
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
            res = _conduit_strdup(PyBytes_AS_STRING(temp_bytes));
            Py_DECREF(temp_bytes);
        }
        else
        {
            // TODO: Error
        }
    }
    else if(PyBytes_Check(py_obj))
    {
        res = _conduit_strdup(PyBytes_AS_STRING(py_obj));
    }
    else
    {
        // TODO: ERROR or auto convert?
    }
    
    return res;
}

//-----------------------------------------------------------------------------
static PyObject *
PyString_FromString(const char *s)
{
    return PyUnicode_FromString(s);
}

//-----------------------------------------------------------------------------
static void
PyString_AsString_Cleanup(char *bytes)
{
    free(bytes);
}


//-----------------------------------------------------------------------------
static int
PyInt_Check(PyObject *o)
{
    return PyLong_Check(o);
}

//-----------------------------------------------------------------------------
static long
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
struct PyRelay_MPI_Request
{
    PyObject_HEAD
    Request request;
};

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// relay::mpi::Request Object 
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_Request_new(PyTypeObject *type,
                        PyObject*, // args -- unused
                        PyObject*) // kwds -- unused
{
    PyRelay_MPI_Request *self = (PyRelay_MPI_Request*)type->tp_alloc(type, 0);

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static void
PyRelay_MPI_Request_dealloc(PyRelay_MPI_Request *self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------//
static int
PyRelay_MPI_Request_init(PyRelay_MPI_Request * /*self*/) // self unused
{
    return 0;
}

//----------------------------------------------------------------------------//
// Request methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyRelay_MPI_Request_METHODS[] = {
    //-----------------------------------------------------------------------//
    //-----------------------------------------------------------------------//
    // end Request methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
static PyTypeObject PyRelay_MPI_Request_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "Request",
   sizeof(PyRelay_MPI_Request),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyRelay_MPI_Request_dealloc, /* tp_dealloc */
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
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* flags */
   "Conduit Relay MPI Request objects",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyRelay_MPI_Request_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyRelay_MPI_Request_init,
   0, /* alloc */
   PyRelay_MPI_Request_new, /* new */
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
// uncomment when we need to use
//---------------------------------------------------------------------------//
// static int
// PyRelay_MPI_Request_Check(PyObject* obj)
// {
//     return (PyObject_TypeCheck(obj, &PyRelay_MPI_Request_TYPE));
// }


//-----------------------------------------------------------------------------
/// MPI Module Functions
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
// conduit::relay::mpi::about
//---------------------------------------------------------------------------//
static PyObject *
PyRelay_MPI_about()
{
    //create and return a node with the result of about
    PyObject *py_node_res = PyConduit_Node_Python_Create();
    Node *node = PyConduit_Node_Get_Node_Ptr(py_node_res);
    conduit::relay::mpi::about(*node);
    return (PyObject*)py_node_res;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::rank
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_rank(PyObject *, //self
                 PyObject *args,
                 PyObject *kwargs)
{
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "n",
                                     const_cast<char**>(kwlist),
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    int rank = -1;

    try
    {
        rank = relay::mpi::rank(comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    return PyLong_FromLong((long)rank);
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::size
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_size(PyObject *, //self
                 PyObject *args,
                 PyObject *kwargs)
{
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "n",
                                     const_cast<char**>(kwlist),
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    int size = 0;

    try
    {
        size = relay::mpi::size(comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    return PyLong_FromLong((long)size);
}

//-----------------------------------------------------------------------------
/// Standard MPI Send Recv
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
// conduit::relay::mpi::send
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_send(PyObject *, //self
                 PyObject *args,
                 PyObject *kwargs)
{
    PyObject   *py_node  = NULL;
    Py_ssize_t  dest;
    Py_ssize_t  tag;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"node",
                                   "dest",
                                   "tag",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Onnn",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &dest,
                                     &tag,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::send(node, dest, tag, comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::recv
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_recv(PyObject *, //self
                 PyObject *args,
                 PyObject *kwargs)
{
    PyObject   *py_node  = NULL;
    Py_ssize_t  source;
    Py_ssize_t  tag;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"node",
                                   "source",
                                   "tag",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Onnn",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &source,
                                     &tag,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }


    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::recv(node, source, tag, comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::send_using_schema
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_send_using_schema(PyObject *, //self
                              PyObject *args,
                              PyObject *kwargs)
{
    PyObject   *py_node  = NULL;
    Py_ssize_t  dest;
    Py_ssize_t  tag;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"node",
                                   "dest",
                                   "tag",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Onnn",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &dest,
                                     &tag,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::send_using_schema(node, dest, tag, comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::recv_using_schema
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_recv_using_schema(PyObject *, //self
                              PyObject *args,
                              PyObject *kwargs)
{
    PyObject   *py_node  = NULL;
    Py_ssize_t  source;
    Py_ssize_t  tag;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"node",
                                   "source",
                                   "tag",
                                   "comm",
                                   NULL};

std::cout << "PyRelay_MPI_recv_using_schema" << std::endl;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Onnn",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &source,
                                     &tag,
                                     &mpi_comm_id))
    {
        std::cout << "PyRelay_MPI_recv_using_schema ansdsadiosnisaniods" << std::endl;
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::recv(node, source, tag, comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
// TODO: Generic MPI Reduce
//-----------------------------------------------------------------------------
// //-----------------------------------------------------------------------------
// /// MPI Reduce
// //-----------------------------------------------------------------------------
//
//     /// MPI reduce and all reduce methods.
//
//     /// While the input does not need to be compact,
//     /// reductions require all MPI ranks have identical compact representations.
//
//     /// These methods do not check across ranks for identical compact
//     /// representation.
//
//     /// Conduit empty, object, and list dtypes can not be reduced.
//
//     /// If the send_node is not compact, it will be compacted prior to sending.
//
//     /// for reduce on the root rank and all_reduce for all ranks:
//     ///   if the recv_node is compatible but not compact, data will be placed
//     ///   into a compact buffer, then read back out into the recv_node node.
//     ///
//     ///   if the recv_node is not compatible, it will be reset to
//     ///   a compact compatible type.
//
//     int CONDUIT_RELAY_API reduce(const Node &send_node,
//                                  Node &recv_node,
//                                  MPI_Op mpi_op,
//                                  int root,
//                                  MPI_Comm comm);
//
//     int CONDUIT_RELAY_API all_reduce(const Node &send_node,
//                                      Node &recv_node,
//                                      MPI_Op mpi_op,
//                                      MPI_Comm comm);
//



//-----------------------------------------------------------------------------
/// MPI Reduce Helpers
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
// conduit::relay::mpi::sum_reduce
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_sum_reduce(PyObject *, //self
                       PyObject *args,
                       PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  root;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm
    // TODO: We could use shared arg parsing for reduce helpers
    //       I started down that path, but opted to keep sep parsing
    //       to get over the finish line

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "root",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOnn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &root,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::sum_reduce(send_node,
                               recv_node,
                               root,
                               comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::min_reduce
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_min_reduce(PyObject *, //self
                       PyObject *args,
                       PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  root;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm
    // TODO: We could use shared arg parsing for reduce helpers
    //       I started down that path, but opted to keep sep parsing
    //       to get over the finish line

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "root",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOnn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &root,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::min_reduce(send_node,
                               recv_node,
                               root,
                               comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}


//---------------------------------------------------------------------------//
// conduit::relay::mpi::max_reduce
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_max_reduce(PyObject *, //self
                       PyObject *args,
                       PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  root;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm
    // TODO: We could use shared arg parsing for reduce helpers
    //       I started down that path, but opted to keep sep parsing
    //       to get over the finish line

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "root",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOnn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &root,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::max_reduce(send_node,
                               recv_node,
                               root,
                               comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}


//---------------------------------------------------------------------------//
// conduit::relay::mpi::prod_reduce
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_prod_reduce(PyObject *, //self
                        PyObject *args,
                        PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  root;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm
    // TODO: We could use shared arg parsing for reduce helpers
    //       I started down that path, but opted to keep sep parsing
    //       to get over the finish line

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "root",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOnn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &root,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::prod_reduce(send_node,
                               recv_node,
                               root,
                               comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::sum_all_reduce
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_sum_all_reduce(PyObject *, //self
                           PyObject *args,
                           PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm
    // TODO: We could use shared arg parsing for reduce helpers
    //       I started down that path, but opted to keep sep parsing
    //       to get over the finish line

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::sum_all_reduce(send_node,
                                   recv_node,
                                   comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::min_all_reduce
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_min_all_reduce(PyObject *, //self
                           PyObject *args,
                           PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  mpi_comm_id;


    // TODO: future also accept mpi4py comm
    // TODO: We could use shared arg parsing for reduce helpers
    //       I started down that path, but opted to keep sep parsing
    //       to get over the finish line

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::min_all_reduce(send_node,
                                   recv_node,
                                   comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::max_all_reduce
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_max_all_reduce(PyObject *, //self
                           PyObject *args,
                           PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm
    // TODO: We could use shared arg parsing for reduce helpers
    //       I started down that path, but opted to keep sep parsing
    //       to get over the finish line

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::max_all_reduce(send_node,
                                   recv_node,
                                   comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::prod_all_reduce
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_prod_all_reduce(PyObject *, //self
                            PyObject *args,
                            PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm
    // TODO: We could use shared arg parsing for reduce helpers
    //       I started down that path, but opted to keep sep parsing
    //       to get over the finish line

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::prod_all_reduce(send_node,
                                    recv_node,
                                    comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
// TODO: Async MPI Send Recv
//-----------------------------------------------------------------------------
//
// //-----------------------------------------------------------------------------
// /// Async MPI Send Recv
// //-----------------------------------------------------------------------------
//
//     int CONDUIT_RELAY_API isend(const Node &node,
//                                 int dest,
//                                 int tag,
//                                 MPI_Comm mpi_comm,
//                                 Request *request);
//
//     int CONDUIT_RELAY_API irecv(Node &node,
//                                 int src,
//                                 int tag,
//                                 MPI_Comm comm,
//                                 Request *request);
//
//     int CONDUIT_RELAY_API wait_send(Request *request,
//                                     MPI_Status *status);
//
//     int CONDUIT_RELAY_API wait_recv(Request *request,
//                                     MPI_Status *status);
//
//     int CONDUIT_RELAY_API wait_all_send(int count,
//                                         Request requests[],
//                                         MPI_Status statuses[]);
//
//     int CONDUIT_RELAY_API wait_all_recv(int count,
//                                         Request requests[],
//                                         MPI_Status statuses[]);
//

//-----------------------------------------------------------------------------
/// MPI gather (these expect identical schemas)
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
// conduit::relay::mpi::gather
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_gather(PyObject *, //self
                   PyObject *args,
                   PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  root;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "root",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOnn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &root,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::gather(send_node,
                           recv_node,
                           root,
                           comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::gather_using_schema
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_gather_using_schema(PyObject *, //self
                                PyObject *args,
                                PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  root;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "root",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOnn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &root,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::gather_using_schema(send_node,
                                        recv_node,
                                        root,
                                        comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::all_gather
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_all_gather(PyObject *, //self
                       PyObject *args,
                       PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::all_gather(send_node,
                               recv_node,
                               comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::gather_using_schema
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_all_gather_using_schema(PyObject *, //self
                                    PyObject *args,
                                    PyObject *kwargs)
{
    PyObject   *py_src_node  = NULL;
    PyObject   *py_recv_node = NULL;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"send_node",
                                   "recv_node",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOn",
                                     const_cast<char**>(kwlist),
                                     &py_src_node,
                                     &py_recv_node,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_src_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'send_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    if(!PyConduit_Node_Check(py_recv_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'recv_node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &send_node = *PyConduit_Node_Get_Node_Ptr(py_src_node);
    Node &recv_node = *PyConduit_Node_Get_Node_Ptr(py_recv_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::all_gather_using_schema(send_node,
                                            recv_node,
                                            comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
/// MPI broadcast
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
// conduit::relay::mpi::broadcast
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_broadcast(PyObject *, //self
                      PyObject *args,
                      PyObject *kwargs)
{
    PyObject   *py_node  = NULL;
    Py_ssize_t  root;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"node",
                                   "root",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Onn",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &root,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::broadcast(node, root, comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// conduit::relay::mpi::broadcast_using_schema
//---------------------------------------------------------------------------//
static PyObject * 
PyRelay_MPI_broadcast_using_schema(PyObject *, //self
                                   PyObject *args,
                                   PyObject *kwargs)
{
    PyObject   *py_node  = NULL;
    Py_ssize_t  root;
    Py_ssize_t  mpi_comm_id;

    // TODO: future also accept mpi4py comm

    static const char *kwlist[] = {"node",
                                   "root",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Onn",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &root,
                                     &mpi_comm_id))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'node' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    // get c mpi comm hnd
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_id);

    try
    {
        relay::mpi::broadcast_using_schema(node, root, comm);
    }
    catch(conduit::Error e)
    {
        PyErr_SetString(PyExc_Exception,
                        e.message().c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}


//---------------------------------------------------------------------------//
// Python Module Method Defs
//---------------------------------------------------------------------------//

// TODO: Future Support:
// reduce
// all_reduce
// isend
// irecv
// wait_send
// wait_recv
// wait_all_send
// wait_all_recv
static PyMethodDef relay_mpi_python_funcs[] =
{
    //-----------------------------------------------------------------------//
    //-----------------------------------------------------------------------//
    {"about",
     (PyCFunction)PyRelay_MPI_about,
      METH_NOARGS,
      "About Relay MPI"},
    {"rank",
     (PyCFunction)PyRelay_MPI_rank,
      METH_VARARGS | METH_KEYWORDS,
      "MPI Comm Rank"},
    {"size",
     (PyCFunction)PyRelay_MPI_size,
      METH_VARARGS | METH_KEYWORDS,
      "MPI Comm Size"},
     // -- send + recv ---
    {"send",
     (PyCFunction)PyRelay_MPI_send,
      METH_VARARGS | METH_KEYWORDS,
      "Send Conduit Node via MPI Send"},
    {"recv",
     (PyCFunction)PyRelay_MPI_recv,
      METH_VARARGS | METH_KEYWORDS,
      "Receive Conduit Node via MPI Recv"},
    {"send_using_schema",
     (PyCFunction)PyRelay_MPI_send_using_schema,
      METH_VARARGS | METH_KEYWORDS,
      "Send Conduit Node and Schema via MPI Send"},
    {"recv_using_schema",
     (PyCFunction)PyRelay_MPI_recv_using_schema,
      METH_VARARGS | METH_KEYWORDS,
      "Receive Conduit Node and Schema via MPI Recv"},
     // -- reduce --
    {"sum_reduce",
     (PyCFunction)PyRelay_MPI_sum_reduce,
      METH_VARARGS | METH_KEYWORDS,
      "Sum Reduce a Conduit Node"},
    {"min_reduce",
     (PyCFunction)PyRelay_MPI_min_reduce,
      METH_VARARGS | METH_KEYWORDS,
      "Min Reduce a Conduit Node"},
    {"max_reduce",
     (PyCFunction)PyRelay_MPI_max_reduce,
      METH_VARARGS | METH_KEYWORDS,
      "Max Reduce a Conduit Node"},
    {"prod_reduce",
     (PyCFunction)PyRelay_MPI_prod_reduce,
      METH_VARARGS | METH_KEYWORDS,
      "Prod Reduce a Conduit Node"},
     // -- all reduce --
    {"sum_all_reduce",
     (PyCFunction)PyRelay_MPI_sum_all_reduce,
      METH_VARARGS | METH_KEYWORDS,
      "Sum All Reduce a Conduit Node"},
    {"min_all_reduce",
     (PyCFunction)PyRelay_MPI_min_all_reduce,
      METH_VARARGS | METH_KEYWORDS,
      "Min All Reduce a Conduit Node"},
    {"max_all_reduce",
     (PyCFunction)PyRelay_MPI_max_all_reduce,
      METH_VARARGS | METH_KEYWORDS,
      "Max All Reduce a Conduit Node"},
    {"prod_all_reduce",
     (PyCFunction)PyRelay_MPI_prod_all_reduce,
      METH_VARARGS | METH_KEYWORDS,
      "Prod All Reduce a Conduit Node"},
     // -- gather --
    {"gather",
     (PyCFunction)PyRelay_MPI_gather,
      METH_VARARGS | METH_KEYWORDS,
      "MPI Gather using Conduit Nodes"},
    {"all_gather",
     (PyCFunction)PyRelay_MPI_all_gather,
      METH_VARARGS | METH_KEYWORDS,
      "MPI All Gather using Conduit Nodes"},
    {"gather_using_schema",
     (PyCFunction)PyRelay_MPI_gather_using_schema,
      METH_VARARGS | METH_KEYWORDS,
      "MPI Gather using Conduit Nodes and their Schemas"},
    {"all_gather_using_schema",
     (PyCFunction)PyRelay_MPI_all_gather_using_schema,
      METH_VARARGS | METH_KEYWORDS,
      "MPI All Gather using Conduit Nodes and their Schemas"},
     // -- broadcast --
    {"broadcast",
     (PyCFunction)PyRelay_MPI_broadcast,
      METH_VARARGS | METH_KEYWORDS,
      "MPI Broadcast a Conduit Node"},
    {"broadcast_using_schema",
     (PyCFunction)PyRelay_MPI_broadcast_using_schema,
      METH_VARARGS | METH_KEYWORDS,
      "MPI Broadcast a Conduit Node and its Schema"},
    //-----------------------------------------------------------------------//
    // end relay mpi methods table
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
relay_mpi_python_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static int 
relay_mpi_python_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static struct PyModuleDef relay_mpi_python_module_def = 
{
        PyModuleDef_HEAD_INIT,
        "relay_mpi_python",
        NULL,
        sizeof(struct module_state),
        relay_mpi_python_funcs,
        NULL,
        relay_mpi_python_traverse,
        relay_mpi_python_clear,
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
CONDUIT_RELAY_PYTHON_API PyObject * PyInit_conduit_relay_mpi_python(void)
#else
CONDUIT_RELAY_PYTHON_API void initconduit_relay_mpi_python(void)
#endif
//---------------------------------------------------------------------------//
{    
    //-----------------------------------------------------------------------//
    // create our main module
    //-----------------------------------------------------------------------//

#if defined(IS_PY3K)
    PyObject *relay_mpi_module = PyModule_Create(&relay_mpi_python_module_def);
#else
    PyObject *relay_mpi_module = Py_InitModule((char*)"conduit_relay_mpi_python",
                                               relay_mpi_python_funcs);
#endif


    if(relay_mpi_module == NULL)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    struct module_state *st = GETSTATE(relay_mpi_module);
    
    st->error = PyErr_NewException((char*)"relay_mpi_python.Error",
                                   NULL,
                                   NULL);
    if (st->error == NULL)
    {
        Py_DECREF(relay_mpi_module);
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

    if (PyType_Ready(&PyRelay_MPI_Request_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    //-----------------------------------------------------------------------//
    // add Request
    //-----------------------------------------------------------------------//
    Py_INCREF(&PyRelay_MPI_Request_TYPE);
    PyModule_AddObject(relay_mpi_module,
                       "Request",
                       (PyObject*)&PyRelay_MPI_Request_TYPE);

#if defined(IS_PY3K)
    return relay_mpi_module;
#endif

}

