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



