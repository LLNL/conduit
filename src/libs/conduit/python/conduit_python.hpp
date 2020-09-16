// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_python.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_PYTHON_HPP
#define CONDUIT_PYTHON_HPP

//-----------------------------------------------------------------------------
// -- Python includes (these must be included first) -- 
//-----------------------------------------------------------------------------
#include <Python.h>
#include <structmember.h>
#include "bytesobject.h"

//---------------------------------------------------------------------------//
// conduit includes
//---------------------------------------------------------------------------//
#include "conduit.hpp"

//---------------------------------------------------------------------------//
// These methods are exposed via python capsule at conduit._C_API, 
// which allows them called in other python C modules.
//
// This solution follows the example outlined here:
//  https://docs.python.org/2/extending/extending.html#using-capsules 
//  https://docs.python.org/3/extending/extending.html#using-capsules
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// conduit node methods
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// int       PyConduit_Node_Check(PyObject* obj);
//---------------------------------------------------------------------------//
#define PyConduit_Node_Check_INDEX 0
#define PyConduit_Node_Check_RETURN int
#define PyConduit_Node_Check_PROTO (PyObject* obj)

//---------------------------------------------------------------------------//
// PyNode_Object *PyConduit_Node_Python_Create();
//---------------------------------------------------------------------------//
#define PyConduit_Node_Python_Create_INDEX 1
#define PyConduit_Node_Python_Create_RETURN PyObject*
#define PyConduit_Node_Python_Create_PROTO ()

//---------------------------------------------------------------------------//
// PyNode_Object *PyConduit_Node_Python_Wrap(Node, owns);
//---------------------------------------------------------------------------//
#define PyConduit_Node_Python_Wrap_INDEX 2
#define PyConduit_Node_Python_Wrap_RETURN PyObject*
#define PyConduit_Node_Python_Wrap_PROTO (conduit::Node *n, int owns)


//---------------------------------------------------------------------------//
// Node     *PyConduit_Node_Get_Node_Ptr(PyObject* obj);
//---------------------------------------------------------------------------//
#define PyConduit_Node_Get_Node_Ptr_INDEX 3
#define PyConduit_Node_Get_Node_Ptr_RETURN conduit::Node*
#define PyConduit_Node_Get_Node_Ptr_PROTO (PyObject* obj)


//---------------------------------------------------------------------------//
// Total number of CAPI pointers
//---------------------------------------------------------------------------//
#define PyConduit_API_number_of_entries 4

//---------------------------------------------------------------------------//
#ifdef CONDUIT_MODULE
//---------------------------------------------------------------------------//
// This section is used when compiling the conduit python module.
//---------------------------------------------------------------------------//

static PyConduit_Node_Check_RETURN PyConduit_Node_Check PyConduit_Node_Check_PROTO;

static PyConduit_Node_Python_Create_RETURN PyConduit_Node_Python_Create PyConduit_Node_Python_Create_PROTO;

static PyConduit_Node_Python_Wrap_RETURN PyConduit_Node_Python_Wrap PyConduit_Node_Python_Wrap_PROTO;

static PyConduit_Node_Get_Node_Ptr_RETURN PyConduit_Node_Get_Node_Ptr PyConduit_Node_Get_Node_Ptr_PROTO;


//---------------------------------------------------------------------------//
#else
//---------------------------------------------------------------------------//
// This section is used in modules that use conduit's python c api 
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static void **PyConduit_API;

//---------------------------------------------------------------------------//
#define PyConduit_Node_Check  \
 (*(PyConduit_Node_Check_RETURN (*)PyConduit_Node_Check_PROTO) PyConduit_API[PyConduit_Node_Check_INDEX])

//---------------------------------------------------------------------------//
#define PyConduit_Node_Python_Create  \
 (*(PyConduit_Node_Python_Create_RETURN (*)PyConduit_Node_Python_Create_PROTO) PyConduit_API[PyConduit_Node_Python_Create_INDEX])

//---------------------------------------------------------------------------//
#define PyConduit_Node_Python_Wrap  \
 (*(PyConduit_Node_Python_Wrap_RETURN (*)PyConduit_Node_Python_Wrap_PROTO) PyConduit_API[PyConduit_Node_Python_Wrap_INDEX])

//---------------------------------------------------------------------------//
#define PyConduit_Node_Get_Node_Ptr  \
 (*(PyConduit_Node_Get_Node_Ptr_RETURN (*)PyConduit_Node_Get_Node_Ptr_PROTO) PyConduit_API[PyConduit_Node_Get_Node_Ptr_INDEX])




//---------------------------------------------------------------------------//
// import_conduit()
// called by client python modules in their C-API to use the functions 
// outlined above.
//
// Return -1 on error, 0 on success.
// PyCapsule_Import will set an exception if there's an error.
//---------------------------------------------------------------------------//
static int
import_conduit(void)
{
    PyConduit_API = (void **)PyCapsule_Import("conduit._C_API", 0);
    return (PyConduit_API != NULL) ? 0 : -1;
}

#endif

#endif



