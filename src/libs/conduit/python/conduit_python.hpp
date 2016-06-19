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

//---------------------------------------------------------------------------//
// conduit includes
//---------------------------------------------------------------------------//
#include "conduit.hpp"
#include "Conduit_Python_Exports.hpp"

using namespace conduit;


//---------------------------------------------------------------------------//
// external conduit python C api 
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
struct PyConduit_DataType {
    PyObject_HEAD
    DataType dtype; // DataType is light weight, we can deal with copies
};

//---------------------------------------------------------------------------//
struct PyConduit_Generator {
    PyObject_HEAD
    Generator *generator;
};

//---------------------------------------------------------------------------//
struct PyConduit_Schema {
    PyObject_HEAD
    Schema *schema;
    int python_owns;
};

//---------------------------------------------------------------------------//
struct PyConduit_NodeIterator {
    PyObject_HEAD
    NodeIterator itr; // NoteIterator is light weight, we can deal with copies
};

//---------------------------------------------------------------------------//
struct PyConduit_Node {
   PyObject_HEAD
   Node *node;
   int python_owns;
};

//---------------------------------------------------------------------------//
static PyConduit_DataType *PyConduit_DataType_python_create();
static int       PyConduit_DataType_Check(PyObject* obj);

//---------------------------------------------------------------------------//
static int       PyConduit_Generator_Check(PyObject* obj);

//---------------------------------------------------------------------------//
static PyObject* PyConduit_Schema_python_wrap(Schema *schema,int python_owns);
static int       PyConduit_Schema_Check(PyObject* obj);

//---------------------------------------------------------------------------//
static PyConduit_Node* PyConduit_Node_python_create();
static PyObject* PyConduit_Node_python_wrap(Node *node,int python_owns);
static int       PyConduit_Node_Check(PyObject* obj);
static int       PyConduit_Node_SetFromPython(Node& node, PyObject* value);
static PyObject* PyConduit_createNumpyType(Node& node, int type);
static PyObject* PyConduit_convertNodeToPython(Node& node);

