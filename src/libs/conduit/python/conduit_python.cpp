//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://scalability-llnl.github.io/conduit/.
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
// Functions to help with Python 2/3 Compatibility.
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
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------//
// include numpy
//---------------------------------------------------------------------------//
// TODO: Use 1.7 deprecated API, or not ?
//---------------------------------------------------------------------------//
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

//---------------------------------------------------------------------------//
// conduit includes
//---------------------------------------------------------------------------//
#include "conduit.hpp"
#include "Conduit_Python_Exports.hpp"

using namespace conduit;

//---------------------------------------------------------------------------//
struct PyConduit_DataType {
    PyObject_HEAD
    DataType dtype; // NoteIterator is light weight, we can deal with copies
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
static int       PyConduit_DataType_check(PyObject* obj);

//---------------------------------------------------------------------------//
static PyConduit_Generator *PyConduit_Generator_python_create();
static int       PyConduit_Generator_check(PyObject* obj);


//---------------------------------------------------------------------------//
static PyConduit_Schema* PyConduit_Schema_python_create();
static PyObject* PyConduit_Schema_python_wrap(Schema *schema,int python_owns);
static int       PyConduit_Schema_Check(PyObject* obj);


//---------------------------------------------------------------------------//
static PyConduit_Node* PyConduit_Node_python_create();
static PyObject* PyConduit_Node_python_wrap(Node *node,int python_owns);
static int       PyConduit_Node_Check(PyObject* obj);
static int       PyConduit_Node_SetFromPython(Node& node, PyObject* value);
static PyObject* PyConduit_createNumpyType(Node& node, int type);
static PyObject* PyConduit_convertNodeToPython(Node& node);

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// DataType Object 
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
static PyObject * 
PyConduit_DataType_new(PyTypeObject* type,
                       PyObject* args,
                       PyObject* kwds)
{
    /// TODO: args and kwargs
    
    static const char *kwlist[] = {"value", NULL};
    PyObject* value = NULL;
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "|O",
                                     const_cast<char**>(kwlist),
                                     &value))
    {
        return (NULL);
    }

    PyConduit_DataType *self = (PyConduit_DataType*)type->tp_alloc(type, 0);
    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static bool
PyConduit_DataType_Set_Parse_Args(PyConduit_DataType* self,
                                  PyObject* args,
                                  PyObject* kwargs)
{
    int32 parse_case = -1;
    ///
    /// Three ways to call:
    ///
    /// Copy Constructor style:
    ///   args[0] == PyObject || kwargs[0] == PyObject

    static const char *kwlist_obj[] = {"dtype",
                                        NULL};
    ///
    /// DataType Name First:
    ///    
    static const char *kwlist_name[] = {"dtype_name",
                                        "num_elements",
                                        "offset",
                                        "stride",
                                        "element_bytes",
                                        "endianness",
                                        NULL};
    ///
    /// DataType Id First:
    ///    
    static const char *kwlist_id[] = {"dtype_id",
                                      "num_elements",
                                      "offset",
                                      "stride",
                                      "element_bytes",
                                      "endianness",
                                       NULL};

    PyObject *py_obj = 0;

    if(args != NULL)
    {
        if(PySequence_Size(args)>0)
        {
            py_obj = PySequence_GetItem(args,0);
            if(py_obj)
            {
                // check for object
                if(PyConduit_DataType_check(py_obj))
                {
                    parse_case = 0;
                }
                // else check for string
                else if(PyString_Check(py_obj))
                {
                    parse_case = 1;
                }
                // else check for index_t
                else if(PyIndex_Check(py_obj))
                {
                    parse_case = 2;                
                }
            }
        }
    }

    if(parse_case == -1 && kwargs !=NULL)
    {
        // check for object
        if(PyDict_GetItemString(kwargs,"dtype"))
        {
            parse_case = 0;
        }
        // else check for string
        else if(PyDict_GetItemString(kwargs,"dtype_name"))
        {
            parse_case = 1;            
        }
        // else check for index_t
        else if(PyDict_GetItemString(kwargs,"dtype_id"))
        {
            parse_case = 2;
        }
    }

    std::cout << "parse_case = " << parse_case <<  std::endl;
    /// if we aren't parsing any args, simply return
    if(parse_case == -1)
        return true;

    Py_ssize_t  dtype_id = 0;
    char       *dtype_name = NULL;
    Py_ssize_t  num_elements = 0;
    Py_ssize_t  offset = 0;
    Py_ssize_t  stride = 0;
    Py_ssize_t  element_bytes = 0;
    Py_ssize_t  endianness =0;

    if(parse_case == 0)
    {
        if (!PyArg_ParseTupleAndKeywords(args,
                                         kwargs,
                                         "O",
                                         const_cast<char**>(kwlist_obj),
                                         py_obj))
        {
            // TODO: Set Error?
            return false;
        }
        
        if(!PyConduit_DataType_check(py_obj))
        {
            // TODO: Set Error?
            return false;
        }
        
        PyConduit_DataType *py_dtype = (PyConduit_DataType *)py_obj;
        
        self->dtype.set(py_dtype->dtype);
    
    }
    else if(parse_case == 1)
    {
        if (!PyArg_ParseTupleAndKeywords(args,
                                         kwargs,
                                         "s|nnnnn",
                                         const_cast<char**>(kwlist_name),
                                         &dtype_name,
                                         &num_elements,
                                         &offset,
                                         &stride,
                                         &element_bytes,
                                         &endianness))
        {
            // TODO: Set Error?
            return false;
        }

        dtype_id = DataType::name_to_id(std::string(dtype_name));

        self->dtype.set(dtype_id,
                        num_elements,
                        offset,
                        stride,
                        element_bytes,
                        endianness);
    }
    else if(parse_case ==2)
    {
        if (!PyArg_ParseTupleAndKeywords(args,
                                         kwargs,
                                         "n|nnnnn",
                                         const_cast<char**>(kwlist_id),
                                         &dtype_id,
                                         &num_elements,
                                         &offset,
                                         &stride,
                                         &element_bytes,
                                         &endianness))
        {
            // TODO: Set Error?
            return false;
        }

        self->dtype.set(dtype_id,
                        num_elements,
                        offset,
                        stride,
                        element_bytes,
                        endianness);
    }
    
    return true;
}


//---------------------------------------------------------------------------//
static int
PyConduit_DataType_init(PyConduit_DataType* self,
                        PyObject* args,
                        PyObject* kwargs)
{
    if(!PyConduit_DataType_Set_Parse_Args(self,args,kwargs))
    {
        // todo: error?
        return 0;
    }
    return 0;
}

//---------------------------------------------------------------------------//
static void
PyConduit_DataType_dealloc(PyConduit_DataType *self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_str(PyConduit_DataType *self)
{
   std::string output = self->dtype.to_json();
   return (Py_BuildValue("s", output.c_str()));
}

//-----------------------------------------------------------------------------
static bool
PyConduit_DataType_Parse_Standard_Set_Keyword_Args(PyObject *args,
                                                   PyObject *kwargs,
                                                   Py_ssize_t &num_elements,
                                                   Py_ssize_t &offset,
                                                   Py_ssize_t &stride,
                                                   Py_ssize_t &element_bytes,
                                                   Py_ssize_t &endianness)
                                                   
                                                      
{
    static const char *kwlist[] = {"num_elements",
                                   "offset",
                                   "stride",
                                   "element_bytes",
                                   "endianness",
                                     NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "|nnnnn",
                                     const_cast<char**>(kwlist),
                                     &num_elements,
                                     &offset,
                                     &stride,
                                     &element_bytes,
                                     &endianness))
    {
        return false;
    }
    return true;
}



//-----------------------------------------------------------------------------
// --- DataType constructor helpers for object types --- //
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
static PyObject *
PyConduit_DataType_empty()
{
    PyConduit_DataType *res = PyConduit_DataType_python_create();
    res->dtype.set_id(DataType::EMPTY_T);
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_object(PyObject *cls)
{
    PyConduit_DataType *res = PyConduit_DataType_python_create();
    res->dtype.set_id(DataType::OBJECT_T);
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_list(PyObject *cls)
{
    PyConduit_DataType *res = PyConduit_DataType_python_create();
    res->dtype.set_id(DataType::LIST_T);
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
// --- DataType constructor helpers for signed integers                  --- //
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
static PyObject *
PyConduit_DataType_int8(PyObject *cls,
                        PyObject *args,
                        PyObject *kwargs)
{
    // default args for int8
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(conduit::int8);
    Py_ssize_t element_bytes = sizeof(conduit::int8);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }
    
    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::int8(num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_int16(PyObject *cls,
                         PyObject *args,
                         PyObject *kwargs)
{
    // default args for int16
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(conduit::int16);
    Py_ssize_t element_bytes = sizeof(conduit::int16);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    

    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))

    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
        
    res->dtype.set(DataType::int16(num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_int32(PyObject *cls,
                         PyObject *args,
                         PyObject *kwargs)
{
    // default args for int32
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(conduit::int32);
    Py_ssize_t element_bytes = sizeof(conduit::int32);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
        
    res->dtype.set(DataType::int32(num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness));
    return (PyObject*)res;
}
//-----------------------------------------------------------------------------
static PyObject *
PyConduit_DataType_int64(PyObject *cls,
                         PyObject *args,
                         PyObject *kwargs)
{
    // default args for int64
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(conduit::int64);
    Py_ssize_t element_bytes = sizeof(conduit::int64);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }
    
    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::int64(num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness));
    return (PyObject*)res;
}


//-----------------------------------------------------------------------------
// --- DataType constructor helpers for unsigned integers                --- //
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_uint8(PyObject *cls,
                         PyObject *args,
                         PyObject *kwargs)
{
    // default args for uint8
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(conduit::uint8);
    Py_ssize_t element_bytes = sizeof(conduit::uint8);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    

    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }
    
    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::uint8(num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness));
    return (PyObject*)res;
;
}

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_uint16(PyObject *cls,
                         PyObject *args,
                         PyObject *kwargs)
{
    // default args for uint16
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(conduit::uint16);
    Py_ssize_t element_bytes = sizeof(conduit::uint16);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }
    
    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::uint16(num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_uint32(PyObject *cls,
                         PyObject *args,
                         PyObject *kwargs)
{
    // default args for uint32
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(conduit::uint32);
    Py_ssize_t element_bytes = sizeof(conduit::uint32);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();    

    res->dtype.set(DataType::uint32(num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness));
    return (PyObject*)res;
}
//-----------------------------------------------------------------------------
static PyObject *
PyConduit_DataType_uint64(PyObject *cls,
                          PyObject *args,
                          PyObject *kwargs)
{
    // default args for uint64
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(conduit::uint64);
    Py_ssize_t element_bytes = sizeof(conduit::uint64);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
        
    res->dtype.set(DataType::uint64(num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
// --- DataType constructor helpers for floating point numbers           --- //
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_float32(PyObject *cls,
                           PyObject *args,
                           PyObject *kwargs)
{
    // default args for float32
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(conduit::float32);
    Py_ssize_t element_bytes = sizeof(conduit::float32);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::float32(num_elements,
                                     offset,
                                     stride,
                                     element_bytes,
                                     endianness));
    return (PyObject*)res;
}
//-----------------------------------------------------------------------------
static PyObject *
PyConduit_DataType_float64(PyObject *cls,
                           PyObject *args,
                           PyObject *kwargs)
{
    // default args for float64
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(conduit::float64);
    Py_ssize_t element_bytes = sizeof(conduit::float64);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
        
    res->dtype.set(DataType::float64(num_elements,
                                     offset,
                                     stride,
                                     element_bytes,
                                     endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
// --- DataType constructor helpers for native c signed integers         --- //
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_c_char(PyObject *cls,
                          PyObject *args,
                          PyObject *kwargs)
{
    // default args for c_char
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(CONDUIT_NATIVE_CHAR_DATATYPE_ID);
    Py_ssize_t element_bytes = sizeof(CONDUIT_NATIVE_CHAR_DATATYPE_ID);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))    
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
        
    res->dtype.set(DataType::c_char(num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_c_short(PyObject *cls,
                           PyObject *args,
                           PyObject *kwargs)
{
    // default args for c_short
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(CONDUIT_NATIVE_SHORT_DATATYPE_ID);
    Py_ssize_t element_bytes = sizeof(CONDUIT_NATIVE_SHORT_DATATYPE_ID);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::c_short(num_elements,
                                     offset,
                                     stride,
                                     element_bytes,
                                     endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_c_int(PyObject *cls,
                         PyObject *args,
                         PyObject *kwargs)
{
    // default args for c_int
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(CONDUIT_NATIVE_INT_DATATYPE_ID);
    Py_ssize_t element_bytes = sizeof(CONDUIT_NATIVE_INT_DATATYPE_ID);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
        
    res->dtype.set(DataType::c_int(num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness));
    return (PyObject*)res;
}
//-----------------------------------------------------------------------------
static PyObject *
PyConduit_DataType_c_long(PyObject *cls,
                          PyObject *args,
                          PyObject *kwargs)
{
    // default args for c_long
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(CONDUIT_NATIVE_LONG_DATATYPE_ID);
    Py_ssize_t element_bytes = sizeof(CONDUIT_NATIVE_LONG_DATATYPE_ID);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::c_long(num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness));
    return (PyObject*)res;
}


//-----------------------------------------------------------------------------
// --- DataType constructor helpers for native c unsigned integers       --- //
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_c_unsigned_char(PyConduit_DataType *cls,
                                   PyObject *args,
                                   PyObject *kwargs)
{
    // default args for c_unsigned_char
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(CONDUIT_NATIVE_UNSIGNED_CHAR_DATATYPE_ID);
    Py_ssize_t element_bytes = sizeof(CONDUIT_NATIVE_UNSIGNED_CHAR_DATATYPE_ID);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    

    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::c_unsigned_char(num_elements,
                                             offset,
                                             stride,
                                             element_bytes,
                                             endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_c_unsigned_short(PyConduit_DataType *cls,
                                    PyObject *args,
                                    PyObject *kwargs)
{
    // default args for c_unsigned_short
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(CONDUIT_NATIVE_UNSIGNED_SHORT_DATATYPE_ID);
    Py_ssize_t element_bytes = sizeof(CONDUIT_NATIVE_UNSIGNED_SHORT_DATATYPE_ID);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }
    
    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::c_unsigned_short(num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_c_unsigned_int(PyConduit_DataType *cls,
                                  PyObject *args,
                                  PyObject *kwargs)
{
    // default args for c_unsigned_int
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(CONDUIT_NATIVE_UNSIGNED_INT_DATATYPE_ID);
    Py_ssize_t element_bytes = sizeof(CONDUIT_NATIVE_UNSIGNED_INT_DATATYPE_ID);
    Py_ssize_t endianness = Endianness::DEFAULT_T;


    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }
    
    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::c_unsigned_int(num_elements,
                                            offset,
                                            stride,
                                            element_bytes,
                                            endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
static PyObject *
PyConduit_DataType_c_unsigned_long(PyConduit_DataType *cls,
                                   PyObject *args,
                                   PyObject *kwargs)
{
    // default args for c_unsigned_long
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(CONDUIT_NATIVE_UNSIGNED_LONG_DATATYPE_ID);
    Py_ssize_t element_bytes = sizeof(CONDUIT_NATIVE_UNSIGNED_LONG_DATATYPE_ID);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();    
    res->dtype.set(DataType::c_unsigned_long(num_elements,
                                             offset,
                                             stride,
                                             element_bytes,
                                             endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
// --- DataType constructor helpers for floating point numbers --- //
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
PyObject *
PyConduit_DataType_c_float(PyConduit_DataType *cls,
                           PyObject *args,
                           PyObject *kwargs)
{
    // default args for c_float
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(CONDUIT_NATIVE_FLOAT_DATATYPE_ID);
    Py_ssize_t element_bytes = sizeof(CONDUIT_NATIVE_FLOAT_DATATYPE_ID);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    

    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::c_float(num_elements,
                                     offset,
                                     stride,
                                     element_bytes,
                                     endianness));
    return (PyObject*)res;
}

//-----------------------------------------------------------------------------
static PyObject *
PyConduit_DataType_c_double(PyConduit_DataType *cls,
                            PyObject *args,
                            PyObject *kwargs)
{
    // default args for c_double
    Py_ssize_t num_elements = 1;
    Py_ssize_t offset = 0;
    Py_ssize_t stride = sizeof(CONDUIT_NATIVE_DOUBLE_DATATYPE_ID);
    Py_ssize_t element_bytes = sizeof(CONDUIT_NATIVE_DOUBLE_DATATYPE_ID);
    Py_ssize_t endianness = Endianness::DEFAULT_T;
    
    if(!PyConduit_DataType_Parse_Standard_Set_Keyword_Args(args,
                                                           kwargs,
                                                           num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness))
    {
        // parsing error
        return NULL;
    }

    PyConduit_DataType *res = PyConduit_DataType_python_create();
    
    res->dtype.set(DataType::c_double(num_elements,
                                      offset,
                                      stride,
                                      element_bytes,
                                      endianness));
    return (PyObject*)res;
}


//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_set(PyConduit_DataType *self,
                       PyObject *args,
                       PyObject *kwargs)
{
    if(!PyConduit_DataType_Set_Parse_Args(self,args,kwargs))
    {
        /// TODO: error?
        return NULL;
    }
    Py_RETURN_NONE; 
}


//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_set_id(PyConduit_DataType *self,
                                          PyObject *args)
{
    Py_ssize_t value;

    if (!PyArg_ParseTuple(args, "n", &value))
    {
        PyErr_SetString(PyExc_TypeError,
            "dtype_id must be a signed integer");
        return NULL;
    }

    self->dtype.set_id(value);

    Py_RETURN_NONE; 
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_set_number_of_elements(PyConduit_DataType *self,
                                          PyObject *args)
{
    Py_ssize_t value;

    if (!PyArg_ParseTuple(args, "n", &value))
    {
        PyErr_SetString(PyExc_TypeError,
            "number_of_elements must be a signed integer");
        return NULL;
    }

    self->dtype.set_number_of_elements(value);

    Py_RETURN_NONE; 
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_set_offset(PyConduit_DataType *self,
                              PyObject *args)
{
    Py_ssize_t value;

    if (!PyArg_ParseTuple(args, "n", &value))
    {
        PyErr_SetString(PyExc_TypeError,
            "offset must be be a signed integer");
        return NULL;
    }

    self->dtype.set_offset(value);

    Py_RETURN_NONE; 
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_set_stride(PyConduit_DataType *self,
                              PyObject *args)
{
    Py_ssize_t value;

    if (!PyArg_ParseTuple(args, "n", &value))
    {
        PyErr_SetString(PyExc_TypeError,
            "stride must be a signed integer");
        return NULL;
    }

    self->dtype.set_stride(value);

    Py_RETURN_NONE; 
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_set_element_bytes(PyConduit_DataType *self,
                                     PyObject *args)
{
    Py_ssize_t value;

    if (!PyArg_ParseTuple(args, "n", &value))
    {
        PyErr_SetString(PyExc_TypeError,
            "element bytes must be a signed integer");
        return NULL;
    }

    self->dtype.set_element_bytes(value);

    Py_RETURN_NONE; 
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_set_endianness(PyConduit_DataType *self,
                                  PyObject *args)
{
    Py_ssize_t value;

    if (!PyArg_ParseTuple(args, "n", &value))
    {
        PyErr_SetString(PyExc_TypeError,
            "endianness must be a signed integer");
        return NULL;
    }

    self->dtype.set_endianness(value);

    Py_RETURN_NONE; 
}

//-----------------------------------------------------------------------------
// Getters and info methods.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_id(PyConduit_DataType *self)
{
    return PyLong_FromSsize_t(self->dtype.id());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_total_bytes(PyConduit_DataType *self)
{
    return PyLong_FromSsize_t(self->dtype.total_bytes());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_total_bytes_compact(PyConduit_DataType *self)
{
    return PyLong_FromSsize_t(self->dtype.total_bytes_compact());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_is_compact(PyConduit_DataType *self)
{
    if(self->dtype.is_compact())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_is_compatible(PyConduit_DataType *self,
                                 PyObject *args)
{
    PyObject *py_dtype;
    if ( (!PyArg_ParseTuple(args, "O", &py_dtype)) || 
         (!PyConduit_DataType_check(py_dtype)) )
    {
         PyErr_SetString(PyExc_TypeError, "is_compatible needs a DataType arg");
         return (NULL);
    }
    
    
    if(self->dtype.is_compatible( ((PyConduit_DataType*)py_dtype)->dtype))
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_is_number(PyConduit_DataType *self)
{
    if(self->dtype.is_number())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}


//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_is_floating_point(PyConduit_DataType *self)
{
    if(self->dtype.is_floating_point())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_is_integer(PyConduit_DataType *self)
{
    if(self->dtype.is_integer())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_is_signed_integer(PyConduit_DataType *self)
{
    if(self->dtype.is_signed_integer())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_is_unsigned_integer(PyConduit_DataType *self)
{
    if(self->dtype.is_unsigned_integer())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_number_of_elements(PyConduit_DataType *self)
{
    return PyLong_FromSsize_t(self->dtype.number_of_elements());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_offset(PyConduit_DataType *self)
{
    return PyLong_FromSsize_t(self->dtype.offset());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_stride(PyConduit_DataType *self)
{
    return PyLong_FromSsize_t(self->dtype.stride());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_element_bytes(PyConduit_DataType *self)
{
    return PyLong_FromSsize_t(self->dtype.element_bytes());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_endianness(PyConduit_DataType *self)
{
    return PyLong_FromSsize_t(self->dtype.endianness());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_element_index(PyConduit_DataType *self,
                                 PyObject *args)
{
    Py_ssize_t idx;

    if (!PyArg_ParseTuple(args, "n", &idx))
    {
        PyErr_SetString(PyExc_TypeError,
                "index must be a signed integer");
        return NULL;
    }

    return PyLong_FromSsize_t(self->dtype.element_index(idx));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_name_to_id(PyObject *cls,
                              PyObject *args)
{
    const char *dtype_name;
    if (!PyArg_ParseTuple(args, "s", &dtype_name))
    {
        PyErr_SetString(PyExc_TypeError, "DataType name must be a string");
        return NULL;
    }

    return PyLong_FromSsize_t(DataType::name_to_id(std::string(dtype_name)));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_id_to_name(PyObject *cls,
                              PyObject *args)
{
    Py_ssize_t dtype_id;

    if (!PyArg_ParseTuple(args, "n", &dtype_id))
    {
        PyErr_SetString(PyExc_TypeError,
                "DataType id must be a signed integer");
        return NULL;
    }

    return Py_BuildValue("s", DataType::id_to_name(dtype_id).c_str());
}



//----------------------------------------------------------------------------//
// DataType methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyConduit_DataType_METHODS[] = {
    //-----------------------------------------------------------------------//
    {"set",
     (PyCFunction)PyConduit_DataType_set,
     METH_VARARGS | METH_KEYWORDS,
     "{todo}"},
    {"set_id",
     (PyCFunction)PyConduit_DataType_set_id,
      METH_VARARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
    {"set_number_of_elements",
     (PyCFunction)PyConduit_DataType_set_number_of_elements,
     METH_VARARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"set_offset",
     (PyCFunction)PyConduit_DataType_set_offset,
     METH_VARARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"set_stride",
     (PyCFunction)PyConduit_DataType_set_stride,
     METH_VARARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"set_element_bytes",
     (PyCFunction)PyConduit_DataType_set_element_bytes,
     METH_VARARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"set_endianness",
     (PyCFunction)PyConduit_DataType_set_endianness,
     METH_VARARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"id",
     (PyCFunction)PyConduit_DataType_id,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"total_bytes",
     (PyCFunction)PyConduit_DataType_total_bytes,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"total_bytes_compact",
     (PyCFunction)PyConduit_DataType_total_bytes_compact,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"is_compact",
     (PyCFunction)PyConduit_DataType_is_compact,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"is_compatible",
     (PyCFunction)PyConduit_DataType_is_compatible,
     METH_VARARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"is_number",
     (PyCFunction)PyConduit_DataType_is_number,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"is_floating_point",
     (PyCFunction)PyConduit_DataType_is_floating_point,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"is_integer",
     (PyCFunction)PyConduit_DataType_is_integer,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"is_signed_integer",
     (PyCFunction)PyConduit_DataType_is_signed_integer,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"is_unsigned_integer",
     (PyCFunction)PyConduit_DataType_is_unsigned_integer,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"number_of_elements",
     (PyCFunction)PyConduit_DataType_number_of_elements,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"offset",
     (PyCFunction)PyConduit_DataType_offset,
     METH_VARARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"stride",
     (PyCFunction)PyConduit_DataType_stride,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"element_bytes",
     (PyCFunction)PyConduit_DataType_element_bytes,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"endianness",
     (PyCFunction)PyConduit_DataType_endianness,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"element_index",
     (PyCFunction)PyConduit_DataType_element_index,
     METH_VARARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    // --- class methods --- ///
    //-----------------------------------------------------------------------//
    //-----------------------------------------------------------------------//
    {"name_to_id",
     (PyCFunction)PyConduit_DataType_name_to_id,
     METH_VARARGS | METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"id_to_name",
     (PyCFunction)PyConduit_DataType_id_to_name,
     METH_VARARGS | METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"empty",
     (PyCFunction)PyConduit_DataType_empty,
     METH_NOARGS | METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"object",
     (PyCFunction)PyConduit_DataType_object,
     METH_NOARGS | METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"list",
     (PyCFunction)PyConduit_DataType_list,
     METH_NOARGS | METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"int8",
     (PyCFunction)PyConduit_DataType_int8,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"int16",
     (PyCFunction)PyConduit_DataType_int16,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"int32",
     (PyCFunction)PyConduit_DataType_int32,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"int64",
     (PyCFunction)PyConduit_DataType_int64,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"uint8",
     (PyCFunction)PyConduit_DataType_uint8,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"uint16",
     (PyCFunction)PyConduit_DataType_uint16,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"uint32",
     (PyCFunction)PyConduit_DataType_uint32,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"uint64",
     (PyCFunction)PyConduit_DataType_uint64,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"float32",
     (PyCFunction)PyConduit_DataType_float32,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"float64",
     (PyCFunction)PyConduit_DataType_float64,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"c_char",
     (PyCFunction)PyConduit_DataType_c_char,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"c_short",
     (PyCFunction)PyConduit_DataType_c_short,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"c_int",
     (PyCFunction)PyConduit_DataType_c_int,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"c_long",
     (PyCFunction)PyConduit_DataType_c_long,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"c_unsigned_char",
     (PyCFunction)PyConduit_DataType_c_unsigned_char,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"c_unsigned_short",
     (PyCFunction)PyConduit_DataType_c_unsigned_short,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"c_unsigned_int",
     (PyCFunction)PyConduit_DataType_c_unsigned_int,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"c_unsigned_long",
     (PyCFunction)PyConduit_DataType_c_unsigned_long,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"c_float",
     (PyCFunction)PyConduit_DataType_c_float,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"c_double",
     (PyCFunction)PyConduit_DataType_c_double,
     METH_VARARGS | METH_KEYWORDS| METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    // end DataType methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
static PyTypeObject PyConduit_DataType_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "DataType",
   sizeof(PyConduit_DataType),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyConduit_DataType_dealloc, /* tp_dealloc */
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
   (reprfunc)PyConduit_DataType_str, /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "Conduit DataType objects",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyConduit_DataType_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyConduit_DataType_init,
   0, /* alloc */
   PyConduit_DataType_new,                                   /* new */
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
static PyConduit_DataType *
PyConduit_DataType_python_create()
{
    PyTypeObject* type = (PyTypeObject*)&PyConduit_DataType_TYPE;
    return (PyConduit_DataType*)type->tp_alloc(type,0);
}

//---------------------------------------------------------------------------//
static int
PyConduit_DataType_check(PyObject *obj)
{
    return (PyObject_TypeCheck(obj, &PyConduit_DataType_TYPE));
}


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// Generator Object 
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject * 
PyConduit_Generator_new(PyTypeObject *type,
                        PyObject *args,
                        PyObject *kwargs)
{
    PyConduit_Generator *self = (PyConduit_Generator*)type->tp_alloc(type, 0);

    if (self)
    {
        self->generator = 0;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static void
PyConduit_Generator_dealloc(PyConduit_Generator *self)
{
    if(self->generator != NULL)
    {
        delete self->generator;
    }
    
    Py_TYPE(self)->tp_free((PyObject*)self);
}


//---------------------------------------------------------------------------//
static int
PyConduit_Generator_init(PyConduit_Generator *self,
                         PyObject *args,
                         PyObject *kwargs)
{
    // TODO: Support "data" arg
    static const char *kwlist[] = {"json_schema",
                                   "protocol",
                                   // "data",
                                    NULL};

     char *json_schema = NULL;
     char *protocol = NULL;
 

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s|s",
                                     const_cast<char**>(kwlist),
                                     &json_schema,
                                     &protocol))
    {
        return -1;
    }

    if(protocol == NULL)
    {
        self->generator = new Generator(std::string(json_schema));
    }
    else if(protocol != NULL)
    {
        self->generator = new Generator(std::string(json_schema),
                                        std::string(protocol));
    }
    
    return 0;

}



//-----------------------------------------------------------------------------
static PyObject *
PyConduit_Generator_walk(PyConduit_Generator *self,
                         PyObject *args,
                         PyObject *kwargs)
{

    static const char *kwlist[] = {"node",
                                   "schema",
                                    NULL};

     PyObject *py_node   = NULL;
     PyObject *py_schema = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "|OO",
                                     const_cast<char**>(kwlist),
                                     &py_node,
                                     &py_schema))
    {
        return NULL;
    }
    
    
    if(py_node != NULL)
    {
        if(!PyConduit_Node_Check(py_node))
        {
            PyErr_SetString(PyExc_TypeError,
                            "Generator::walk 'node' argument must be a "
                            "Conduit::Node");
            return NULL;
        }
    }
    
    if(py_schema != NULL)
    {
        if(!PyConduit_Schema_Check(py_schema))
        {
            PyErr_SetString(PyExc_TypeError,
                            "Generator::walk 'schema' argument must be a "
                            "Conduit::Schema");
            return NULL;
        }
    }
    
    if(py_node != NULL)
    {
        Node *node_ptr = ((PyConduit_Node*)py_node)->node;
        self->generator->walk(*node_ptr);
    }
    
    if(py_schema != NULL)
    {
        Schema *schema_ptr = ((PyConduit_Schema*)py_schema)->schema;
        self->generator->walk(*schema_ptr);
    }

    Py_RETURN_NONE; 
}

//-----------------------------------------------------------------------------
static PyObject *
PyConduit_Generator_walk_external(PyConduit_Generator *self,
                                  PyObject *args,
                                  PyObject *kwargs)
{

    static const char *kwlist[] = {"node",
                                    NULL};

     PyObject *py_node   = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "O",
                                     const_cast<char**>(kwlist),
                                     &py_node))
    {
        return NULL;
    }
    
    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "Generator::walk_external 'node' argument must "
                        "be a Conduit::Node");
        return NULL;
    }


    Node *node_ptr = ((PyConduit_Node*)py_node)->node;
    self->generator->walk_external(*node_ptr);

    Py_RETURN_NONE; 
}



//----------------------------------------------------------------------------//
// Generator methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyConduit_Generator_METHODS[] = {
    //-----------------------------------------------------------------------//
    {"walk",
     (PyCFunction)PyConduit_Generator_walk,
     METH_VARARGS | METH_KEYWORDS,
     "Use Generator to parse a JSON schema into passed 'node' or 'schema'"},
    {"walk_external",
     (PyCFunction)PyConduit_Generator_walk_external,
     METH_VARARGS | METH_KEYWORDS,
     "Use Generator to parse a JSON schema into passed 'node' using"
     " external data"},
    //-----------------------------------------------------------------------//
    // end Generator methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
static PyTypeObject PyConduit_Generator_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "Generator",
   sizeof(PyConduit_Generator),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyConduit_Generator_dealloc, /* tp_dealloc */
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
   "Conduit Generator objects",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyConduit_Generator_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyConduit_Generator_init,
   0, /* alloc */
   PyConduit_Generator_new,                       /* new */
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
static PyConduit_Generator *
PyConduit_Generator_python_create()
{
    PyTypeObject* type = (PyTypeObject*)&PyConduit_Generator_TYPE;
    return (PyConduit_Generator*)type->tp_alloc(type,0);
}

//---------------------------------------------------------------------------//
static int
PyConduit_Generator_check(PyObject* obj)
{
    return (PyObject_TypeCheck(obj, &PyConduit_Generator_TYPE));
}



//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// Schema Object 
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
static PyObject * 
PyConduit_Schema_new(PyTypeObject* type,
                     PyObject* args,
                     PyObject* kwds)
{

    PyConduit_Schema* self = (PyConduit_Schema*)type->tp_alloc(type, 0);
    
    if (self)
    {
        self->schema = 0;
        self->python_owns = 0;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static int
PyConduit_Schema_init(PyConduit_Schema* self,
                      PyObject* args,
                      PyObject* kwds)
{
     static const char *kwlist[] = {"value", NULL};
     PyObject* value = NULL;
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "|O",
                                     const_cast<char**>(kwlist),
                                     &value))
    {
         return (0);
    }

     if (value)
     {
         if (PyConduit_Schema_Check(value))
         {
             self->schema = new Schema(*((PyConduit_Schema*)value)->schema);
         }
         else if (PyString_Check(value))
         {
             char *cstr = PyString_AsString(value);
             self->schema = new Schema(cstr);
             PyString_AsString_Cleanup(cstr);
         }
         else
         {
             PyErr_SetString(PyExc_TypeError, "Invalid initializer for schema");
             return (-1);
         }
     }
     else
     {
         self->schema = new Schema();
     }

     //self->python_owns = 1;

     return (0);

}

//---------------------------------------------------------------------------//
static void
PyConduit_Schema_dealloc(PyConduit_Schema* self)
{
    if (self->python_owns && self->schema != NULL)
    {
        delete self->schema;
    }
    
    Py_TYPE(self)->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Schema_str(PyConduit_Schema *self)
{
   std::ostringstream oss;
   self->schema->to_json(oss);
   return (Py_BuildValue("s", oss.str().c_str()));
}


//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Schema_python_owns(PyConduit_Schema *self)
{
    if(self->python_owns)
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Schema_python_detach(PyConduit_Schema *self)
{
    self->python_owns = 0;
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Schema_python_attach(PyConduit_Schema *self)
{
    self->python_owns = 1;
    Py_RETURN_NONE;
}


//-----------------------------------------------------------------------------
//
/// Information Methods
//
//-----------------------------------------------------------------------------
    // const DataType &dtype() const
    // index_t         total_bytes() const;
    // index_t         total_bytes_compact() const;
    // index_t         element_index(index_t idx) const
    // Schema         *parent() const
    // bool            is_root() const

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Schema_dtype(PyConduit_Schema *self)
{
    PyConduit_DataType *retval = PyConduit_DataType_python_create();
    retval->dtype = self->schema->dtype();
    return (PyObject*)retval;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Schema_total_bytes(PyConduit_Schema *self)
{
    return PyLong_FromSsize_t(self->schema->total_bytes());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Schema_total_bytes_compact(PyConduit_Schema *self)
{
    return PyLong_FromSsize_t(self->schema->total_bytes_compact());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Schema_element_index(PyConduit_Schema *self,
                               PyObject *args)
{
    Py_ssize_t idx;

    if (!PyArg_ParseTuple(args, "n", &idx))
    {
        PyErr_SetString(PyExc_TypeError,
                "index must be a signed integer");
        return NULL;
    }

    return PyLong_FromSsize_t(self->schema->element_index(idx));
}

//---------------------------------------------------------------------------//
static PyObject * 
PyConduit_Schema_is_root(PyConduit_Schema *self)
{
    if(self->schema->is_root())
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
PyConduit_Schema_parent(PyConduit_Schema* self)
{
    if(self->schema->is_root())
    {
        Py_RETURN_NONE;
    }
    else
    {
        // python owned == 0 (at least for this instance)
        return PyConduit_Schema_python_wrap(self->schema->parent(),0);
    }
}



//-----------------------------------------------------------------------------
//
/// Transformation Methods
//
//-----------------------------------------------------------------------------
    // void            compact_to(Schema &s_dest) const;
    // std::string     to_json(bool detailed=true,
    //                             index_t indent=2,
    //                             index_t depth=0,
    //                             const std::string &pad=" ",
    //                             const std::string &eoe="\n") const;
    // void            to_json(std::ostringstream &oss,
    //                             bool detailed=true,
    //                             index_t indent=2,
    //                             index_t depth=0,
    //                             const std::string &pad=" ",
    //                             const std::string &eoe="\n") const;

//-----------------------------------------------------------------------------
//
/// Basic I/O methods
//
//-----------------------------------------------------------------------------
    // void            save(const std::string &ofname,
    //                      bool detailed=true,
    //                      index_t indent=2,
    //                      index_t depth=0,
    //                      const std::string &pad=" ",
    //                      const std::string &eoe="\n") const;
    //
    // void            load(const std::string &ifname);


//----------------------------------------------------------------------------//
// Schema methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyConduit_Schema_METHODS[] = {
    //-----------------------------------------------------------------------//
    {"python_owns",
     (PyCFunction)PyConduit_Schema_python_owns,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"python_attach",
     (PyCFunction)PyConduit_Schema_python_attach,
      METH_NOARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
    {"python_detach",
     (PyCFunction)PyConduit_Schema_python_detach,
     METH_NOARGS,
     "{todo}"},
     //-----------------------------------------------------------------------//
     {"dtype",
      (PyCFunction)PyConduit_Schema_dtype,
      METH_NOARGS,
      "{todo}"},
     //-----------------------------------------------------------------------//
     {"total_bytes",
      (PyCFunction)PyConduit_Schema_total_bytes,
       METH_NOARGS,
       "{todo}"},
     //-----------------------------------------------------------------------//
     {"total_bytes_compact",
      (PyCFunction)PyConduit_Schema_total_bytes_compact,
      METH_NOARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
    {"element_index",
     (PyCFunction)PyConduit_Schema_element_index,
     METH_VARARGS,
     "{todo}"},
     //-----------------------------------------------------------------------//
     {"is_root",
      (PyCFunction)PyConduit_Schema_is_root,
      METH_NOARGS,
      "{todo}"},
     //-----------------------------------------------------------------------//
     {"parent",
      (PyCFunction)PyConduit_Schema_parent,
      METH_NOARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
    // end Schema methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
static PyTypeObject PyConduit_Schema_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "Schema",
   sizeof(PyConduit_Schema),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyConduit_Schema_dealloc, /* tp_dealloc */
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
   (reprfunc)PyConduit_Schema_str, /* str */
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
   PyConduit_Schema_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyConduit_Schema_init,
   0, /* alloc */
   PyConduit_Schema_new,                                   /* new */
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
static PyObject *
PyConduit_NodeIterator_new(PyTypeObject *type,
                            PyObject *args,
                            PyObject *kwds)
{
    /// TODO: args and kwargs
    /// TODO: args and kwargs
    
    static const char *kwlist[] = {"value", NULL};
    PyObject* value = NULL;
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "|O",
                                     const_cast<char**>(kwlist),
                                     &value))
    {
        return (NULL);
    }
    
    PyConduit_DataType *self = (PyConduit_DataType*)type->tp_alloc(type, 0);
    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static int
PyConduit_NodeIterator_init(PyConduit_Schema* self,
                            PyObject* args,
                            PyObject* kwds)
{
    /// TODO: args and kwargs
    static const char *kwlist[] = {"value", NULL};
    PyObject* value = NULL;
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "|O",
                                     const_cast<char**>(kwlist),
                                     &value))
        
    {
        return 0;
    }

    return (0);
}

//---------------------------------------------------------------------------//
static void
PyConduit_NodeIterator_dealloc(PyConduit_NodeIterator *self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------//
///TODO: imp
static PyObject *
PyConduit_NodeIterator_str(PyConduit_NodeIterator *self)
{
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_iter(PyObject *self)
{
    Py_INCREF(self);
    return self;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_iternext(PyObject *self)
{
    PyConduit_NodeIterator *py_nitr = (PyConduit_NodeIterator *)self;
    
    if(py_nitr->itr.has_next()) 
    {
        py_nitr->itr.next();
        Py_INCREF(self);
        return self;
    }
    else 
    {
        // Raise StopIteration
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_path(PyConduit_NodeIterator *self)
{
    return (Py_BuildValue("s", self->itr.path().c_str()));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_index(PyConduit_NodeIterator *self)
{
    return PyLong_FromSsize_t(self->itr.index());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_node(PyConduit_NodeIterator *self)
{
    return PyConduit_Node_python_wrap(&(self->itr.node()),0);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_to_front(PyConduit_NodeIterator *self)
{
    self->itr.to_front();
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_has_next(PyConduit_NodeIterator *self)
{
    if(self->itr.has_next())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_next(PyConduit_NodeIterator *self)
{
    if(self->itr.has_next())
    {
        Node &n = self->itr.next();
        return PyConduit_Node_python_wrap(&n,0);
    }
    else
    {
        // if done return None
        Py_RETURN_NONE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_peek_next(PyConduit_NodeIterator *self)
{
    if(self->itr.has_next())
    {
        Node &n = self->itr.peek_next();
        return PyConduit_Node_python_wrap(&n,0);
    }
    else
    {
        // if done return None
        Py_RETURN_NONE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_has_previous(PyConduit_NodeIterator *self)
{
    if(self->itr.has_previous())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_previous(PyConduit_NodeIterator *self)
{
    if(self->itr.has_previous())
    {
        Node &n = self->itr.previous();
        return PyConduit_Node_python_wrap(&n,0);
    }
    else
    {
        // if done return None
        Py_RETURN_NONE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_peek_previous(PyConduit_NodeIterator *self)
{
    if(self->itr.has_previous())
    {
        Node &n = self->itr.peek_previous();
        return PyConduit_Node_python_wrap(&n,0);
    }
    else
    {
        // if done return None
        Py_RETURN_NONE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_to_back(PyConduit_NodeIterator *self)
{
    self->itr.to_back();
    Py_RETURN_NONE;
}


//---------------------------------------------------------------------------//
static PyObject *
PyConduit_NodeIterator_info(PyConduit_NodeIterator *self)
{
    //create and return a node with the result of info
    PyConduit_Node *retval = PyConduit_Node_python_create();
    self->itr.info(*retval->node);
    return (PyObject*)retval;
}

//----------------------------------------------------------------------------//
// NodeIterator methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyConduit_NodeIterator_METHODS[] = {
    //-----------------------------------------------------------------------//
    {"path",
     (PyCFunction)PyConduit_NodeIterator_path, 
     METH_NOARGS,
     "{todo}"}, 
    //-----------------------------------------------------------------------//
    {"index",
     (PyCFunction)PyConduit_NodeIterator_index, 
     METH_NOARGS,
     "{todo}"}, 
     //-----------------------------------------------------------------------//
     {"node",
      (PyCFunction)PyConduit_NodeIterator_node, 
      METH_NOARGS,
      "{todo}"}, 
    //-----------------------------------------------------------------------//
    {"to_front",
     (PyCFunction)PyConduit_NodeIterator_to_front, 
     METH_NOARGS,
     "{todo}"}, 
    //-----------------------------------------------------------------------//
    {"has_next",
     (PyCFunction)PyConduit_NodeIterator_has_next, 
     METH_NOARGS,
     "{todo}"}, 
    //-----------------------------------------------------------------------//
    {"next",
     (PyCFunction)PyConduit_NodeIterator_next, 
     METH_NOARGS,
     "{todo}"}, 
    //-----------------------------------------------------------------------//
    {"peek_next",
     (PyCFunction)PyConduit_NodeIterator_peek_next, 
     METH_NOARGS,
     "{todo}"}, 
    //-----------------------------------------------------------------------//
    {"has_previous",
     (PyCFunction)PyConduit_NodeIterator_has_previous, 
     METH_NOARGS,
     "{todo}"}, 
    //-----------------------------------------------------------------------//
    {"previous",
     (PyCFunction)PyConduit_NodeIterator_previous,
     METH_NOARGS,
     "{todo}"}, 
    //-----------------------------------------------------------------------//
    {"peek_previous",
     (PyCFunction)PyConduit_NodeIterator_peek_previous, 
     METH_NOARGS,
     "{todo}"}, 
    //-----------------------------------------------------------------------//
    {"to_back",
     (PyCFunction)PyConduit_NodeIterator_to_back, 
     METH_NOARGS,
     "{todo}"}, 
    //-----------------------------------------------------------------------//
    {"info",
     (PyCFunction)PyConduit_NodeIterator_info, 
     METH_NOARGS,
     "{todo}"}, 
    //-----------------------------------------------------------------------//
    // end NodeIterator methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
static PyTypeObject PyConduit_NodeIterator_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "Schema",
   sizeof(PyConduit_NodeIterator),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyConduit_NodeIterator_dealloc,   /* tp_dealloc */
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
   (reprfunc)PyConduit_NodeIterator_str,                         /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "Conduit schema objects",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   PyConduit_NodeIterator_iter, /* iter */
   PyConduit_NodeIterator_iternext, /* iternext */
   PyConduit_NodeIterator_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyConduit_NodeIterator_init,
   0, /* alloc */
   PyConduit_NodeIterator_new,   /* new */
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
static PyConduit_NodeIterator *
PyConduit_NodeIterator_python_create()
{
    PyTypeObject* type = (PyTypeObject*)&PyConduit_NodeIterator_TYPE;
    return (PyConduit_NodeIterator*)type->tp_alloc(type,0);
}


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// conduit:::Node class
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <class T>
static void
PyConduit_Fill_DataArray_From_PyArray(DataArray<T> &conduit_array,
                                      PyArrayObject *numpy_array)
{
    npy_int num_ele = conduit_array.number_of_elements();
    
    T *numpy_data = (T*)PyArray_BYTES(numpy_array);
    
    for (npy_intp i = 0; i < num_ele; i++)
    {
        conduit_array[i] = numpy_data[i];
    }
    
}


//---------------------------------------------------------------------------//
// begin Node python special methods
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_new(PyTypeObject* type,
                   PyObject* args,
                   PyObject* kwds)
{
    static const char *kwlist[] = {"value", NULL};
    PyObject* value = NULL;
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "|O",
                                     const_cast<char**>(kwlist),
                                     &value))
    {
        return (NULL);
    }

    PyConduit_Node* self = (PyConduit_Node*)type->tp_alloc(type, 0);

    if (self)
    {
        self->node = 0;
        self->python_owns = 0;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static int
PyConduit_Node_init(PyConduit_Node* self,
                    PyObject* args,
                    PyObject* kwds)
{
    static const char *kwlist[] = {"value", NULL};
    PyObject* value = NULL;
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "|O",
                                     const_cast<char**>(kwlist),
                                     &value))
    {
        return 0;
    }
    
    self->node = new Node();
    self->python_owns = 1;

    if (value)
    {
        return (PyConduit_Node_SetFromPython(*self->node, value));
    }
    else 
    {
        return 0;
    }
}

//---------------------------------------------------------------------------//
static void 
PyConduit_Node_dealloc(PyConduit_Node* self)
{
    if(self->python_owns)
    {
       delete self->node;
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_str(PyConduit_Node* self)
{
   std::ostringstream oss;
   self->node->to_json_stream(oss);
   return (Py_BuildValue("s", oss.str().c_str()));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_GetItem(PyConduit_Node* self,
                      PyObject* key)
{
    if (!PyString_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Key must be a string");
        return (NULL);
    }

    PyObject* retval = NULL;
    char* ckey = PyString_AsString(key);

    if(self->node->has_path(ckey))
    {
        Node& node = (*self->node)[ckey];
        retval = PyConduit_convertNodeToPython(node);
    }
    else
    {
        retval = PyConduit_Node_python_wrap(&(*self->node)[ckey],0);
    }
    
    PyString_AsString_Cleanup(ckey);
    
    return (retval);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_python_owns(PyConduit_Node *self)
{
    if(self->python_owns)
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_python_detach(PyConduit_Node *self)
{
    self->python_owns = 0;
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_python_attach(PyConduit_Node *self)
{
    self->python_owns = 1;
    Py_RETURN_NONE;
}


//---------------------------------------------------------------------------//
// end Node python special methods
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_value(PyConduit_Node* self)
{
    PyObject* retval = NULL;
    retval = PyConduit_convertNodeToPython(*self->node);
    return (retval);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_generate(PyConduit_Node* self,
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
         PyErr_SetString(PyExc_TypeError, "JSON schema must be a string");
         return NULL;
     }
     
     self->node->generate(std::string(json_schema));
     Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_save(PyConduit_Node *self,
                    PyObject *args,
                    PyObject *kwargs)
{   
    const char *path      = NULL;
    const char *protocol  = NULL;
    
    // support:
    // path
    // path, protocol
    
    static const char *kwlist[] = {"path","protocol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s|s",
                                     const_cast<char**>(kwlist),
                                     &path,&protocol))
    {
        return NULL;
    }
    
    std::string path_str(path);
    std::string protocol_str("conduit_pair");
    
    if(protocol != NULL)
    {
        protocol_str = std::string(protocol);
    }
    
    self->node->save(path_str,protocol_str);
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_load(PyConduit_Node *self,
                    PyObject *args,
                    PyObject *kwargs)
{
    PyObject   *py_schema = NULL;
    const char *path      = NULL;
    const char *protocol  = NULL;
    
    // support:
    // path
    // path, schema
    // path, protocol
    
    static const char *kwlist[] = {"path","schema","protocol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s|Os",
                                     const_cast<char**>(kwlist),
                                     &path,&py_schema, &protocol))
    {
        return NULL;
    }
    

    if(py_schema != NULL)
    {
        if(!PyConduit_Schema_Check(py_schema))
        {
            PyErr_SetString(PyExc_TypeError,
                            "Node::load 'schema' argument must be a "
                            "Conduit::Schema");
            return NULL;
        }
    }
    
    std::string path_str(path);
    
    if(py_schema != NULL)
    {
        Schema *schema_ptr = ((PyConduit_Schema*)py_schema)->schema;
        self->node->load(path_str,
                         *schema_ptr);
    }
    else
    {
        std::string protocol_str("conduit_pair");
        
        if( protocol != NULL)
        {
            protocol_str =  std::string(protocol);
        }
        
        self->node->load(path_str,
                         protocol_str);
        
    }

    
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_mmap(PyConduit_Node* self,
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

//-----------------------------------------------------------------------------
//
// -- Node Entry Access --
//
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_fetch(PyConduit_Node* self,
                     PyObject* args)
{
     const char *key;
     PyObject* retval = NULL;
     if (!PyArg_ParseTuple(args, "s", &key))
     {
         PyErr_SetString(PyExc_TypeError, "Key must be a string");
         return NULL;
     }

    retval = PyConduit_Node_python_wrap(&(*self->node).fetch(key),0);
    return (retval);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_child(PyConduit_Node* self,
                    PyObject* args)
{
     Py_ssize_t idx;
     PyObject* retval = NULL;
     if (!PyArg_ParseTuple(args, "n", &idx))
     {
         PyErr_SetString(PyExc_TypeError, "Index must be an integer");
         return NULL;
     }

    retval = PyConduit_Node_python_wrap(&(*self->node).child(idx),0);
    return (retval);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_number_of_children(PyConduit_Node *self)
{
    return PyLong_FromSsize_t(self->node->number_of_children());
}

//---------------------------------------------------------------------------//
static PyObject * 
PyConduit_Node_has_path(PyConduit_Node *self,
                       PyObject* args)
{
    const char *path;

    if (!PyArg_ParseTuple(args, "s", &path))
    {
        PyErr_SetString(PyExc_TypeError, "Path must be a string");
        return NULL;
    }
    
    if(self->node->has_path(std::string(path)))
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject * 
PyConduit_Node_paths(PyConduit_Node *self)
{
    std::vector<std::string> paths;
    self->node->paths(paths);
    
    /// TODO: I think there is a faster way in the Python CAPI
    /// since we know the size of the list.
    PyObject *retval = PyList_New(0);
    
    for (std::vector<std::string>::const_iterator itr = paths.begin();
         itr < paths.end(); ++itr)
    {
        PyList_Append(retval, PyString_FromString( (*itr).c_str()));
    };

    return retval;
}


//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_append(PyConduit_Node* self)
{
    return  PyConduit_Node_python_wrap(&(self->node->append()),0);
}


//---------------------------------------------------------------------------//
static PyObject * 
PyConduit_Node_remove(PyConduit_Node *self,
                      PyObject *args,
                      PyObject *kwargs)
{
    Py_ssize_t idx;
    const char *path = NULL;

    static const char *kwlist[] = {"index","path", NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "|ns",
                                     const_cast<char**>(kwlist),
                                     &idx, &path))
    {
        return (NULL);
    }
    
    if(path != NULL)
    {
        self->node->remove(std::string(path));
    }else
    {
        self->node->remove(idx);
    }

    Py_RETURN_NONE;
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
PyConduit_Node_schema(PyConduit_Node *self)
{
    // python_owns = 0
    return (PyConduit_Schema_python_wrap(self->node->schema_ptr(),0));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_dtype(PyConduit_Node *self)
{
    PyConduit_DataType *retval = PyConduit_DataType_python_create();
    retval->dtype = self->node->dtype();
    return (PyObject*)retval;
}

//---------------------------------------------------------------------------//
// parent access
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject * 
PyConduit_Node_is_root(PyConduit_Node *self)
{
    if(self->node->is_root())
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
PyConduit_Node_parent(PyConduit_Node* self)
{
    if(self->node->is_root())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return PyConduit_Node_python_wrap(self->node->parent(),0);
    }
}

//---------------------------------------------------------------------------//
//memory space info
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_total_bytes(PyConduit_Node *self)
{
    return PyLong_FromSsize_t(self->node->total_bytes());
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_total_bytes_compact(PyConduit_Node *self)
{
    return PyLong_FromSsize_t(self->node->total_bytes_compact());
}

//---------------------------------------------------------------------------//
static PyObject * 
PyConduit_Node_is_compact(PyConduit_Node *self)
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
static PyObject * 
PyConduit_Node_info(PyConduit_Node *self)
{
    PyConduit_Node *retval = PyConduit_Node_python_create();
    self->node->info(*retval->node);
    return (PyObject*)retval;
}


//---------------------------------------------------------------------------//
static int PyConduit_Node_SetItem(PyConduit_Node *self,
                                  PyObject *key,
                                  PyObject *value)
{
    if (!PyString_Check(key))
    {
        PyErr_SetString(PyExc_TypeError, "Key must be a string");
        return (-1);
    }

    char* ckey = PyString_AsString(key);
    Node& node = (*self->node)[ckey];
    PyString_AsString_Cleanup(ckey);
    return (PyConduit_Node_SetFromPython(node, value));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_iter(PyObject *self)
{
    PyConduit_Node *py_n = (PyConduit_Node *)self;

    PyConduit_NodeIterator *retval = NULL;
    retval = PyConduit_NodeIterator_python_create();
    retval->itr =  py_n->node->children();

    return ((PyObject *)retval);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_set(PyConduit_Node* self,
                   PyObject* args)
{
    PyObject* value = NULL;
    
    if (!PyArg_ParseTuple(args, "O", &value))
    {
         return (NULL);
    }

    if (PyConduit_Node_SetFromPython(*self->node, value))
    {
         return (NULL);
    }
    else
    {
        Py_RETURN_NONE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_set_external(PyConduit_Node* self,
                   PyObject* args)
{
    PyObject* value = NULL;

    if (!PyArg_ParseTuple(args, "O", &value))
    {
        return (NULL);
    }

    if (!PyArray_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "set_external requires a numpy array");
        return (NULL);
    }

    PyArray_Descr *desc = PyArray_DESCR((PyArrayObject*)value);
    PyArrayObject *py_arr = (PyArrayObject*)value;
    npy_intp num_ele = PyArray_SIZE(py_arr);
    int elsize = desc->elsize;
    int offset = 0;
    int stride = PyArray_STRIDE(py_arr, 0)*elsize;
    int nd = PyArray_NDIM(py_arr);

    if (nd > 1) {
        PyErr_SetString(PyExc_TypeError, "set_external does not handle multidimensional numpy arrays");
        return (NULL);
    }

    Node& node = *self->node;

    switch (desc->type_num) 
    {
        case NPY_UINT8 :
        {
            uint8 *numpy_data = (uint8*)PyArray_BYTES(py_arr);
            node.set_external(numpy_data, num_ele, offset, stride);
            break;
        }
        case NPY_UINT16 :
        {
            uint16 *numpy_data = (uint16*)PyArray_BYTES(py_arr);
            node.set_external(numpy_data, num_ele, offset, stride);
            break;
        }
        case NPY_UINT32 :
        {
            uint32 *numpy_data = (uint32*)PyArray_BYTES(py_arr);
            node.set_external(numpy_data, num_ele, offset, stride);
            break;
        }
        case NPY_UINT64 :
        {
            uint64 *numpy_data = (uint64*)PyArray_BYTES(py_arr);
            node.set_external(numpy_data, num_ele, offset, stride);
            break;
        }
        case NPY_INT8 :
        {
            int8 *numpy_data = (int8*)PyArray_BYTES(py_arr);
            node.set_external(numpy_data, num_ele, offset, stride);
            break;
        }
        case NPY_INT16 :
        {
            int16 *numpy_data = (int16*)PyArray_BYTES(py_arr);
            node.set_external(numpy_data, num_ele, offset, stride);
            break;
        }
        case NPY_INT32 :
        {
            int32 *numpy_data = (int32*)PyArray_BYTES(py_arr);
            node.set_external(numpy_data, num_ele, offset, stride);
            break;
        }
        case NPY_INT64 :
        {
            int64 *numpy_data = (int64*)PyArray_BYTES(py_arr);
            node.set_external(numpy_data, num_ele, offset, stride);
            break;
        }
        case NPY_FLOAT32 :
        {
            float32 *numpy_data = (float32*)PyArray_BYTES(py_arr);
            node.set_external(numpy_data, num_ele, offset, stride);
            break;
        }
        case NPY_FLOAT64 :
        {
            float64 *numpy_data = (float64*)PyArray_BYTES(py_arr);
            node.set_external(numpy_data, num_ele, offset, stride);
            break;
        }
        default :
        {
            PyErr_SetString(PyExc_TypeError, "Unsupported type");
            return (NULL);
        }
    }

    Py_RETURN_NONE;

}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_set_path(PyConduit_Node* self,
                       PyObject* args)
{
    PyObject* value = NULL;
    PyObject *path  = NULL;
    
    if (!PyArg_ParseTuple(args, "OO", &path, &value))
    {
         return (NULL);
    }

    if (PyConduit_Node_SetItem(self, path, value))
    {
        return (NULL);
    }
    else 
    {
        Py_RETURN_NONE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_to_json(PyConduit_Node* self,
                       PyObject* args,
                       PyObject* kwargs)
{
    
    Py_ssize_t indent = 2;
    Py_ssize_t depth  = 0;

    std::string protocol = "json";
    std::string pad = " ";
    std::string eoe = "\n";

    char *protocol_c_str = NULL;
    char *pad_c_str = NULL;
    char *eoe_c_str = NULL;
    
    static const char *kwlist[] = {"protocol",
                                   "indent",
                                   "depth",
                                   "pad",
                                   "eoe",
                                    NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "|snnss",
                                     const_cast<char**>(kwlist),
                                     &protocol_c_str,
                                     &indent,
                                     &depth,
                                     &pad_c_str,
                                     &eoe_c_str))
    {
        return NULL;
    }
    
    if(protocol_c_str != NULL)
    {
        protocol = std::string(protocol_c_str);
    }
    
    if(pad_c_str != NULL)
    {
        pad = std::string(pad_c_str);
    }

    if(eoe_c_str != NULL)
    {
        eoe = std::string(eoe_c_str);
    }
    
    std::string output = self->node->to_json(protocol,
                                             indent,
                                             depth,
                                             pad,
                                             eoe);

    return (Py_BuildValue("s", output.c_str()));
}


//----------------------------------------------------------------------------//
// Node methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyConduit_Node_METHODS[] = {
    //-----------------------------------------------------------------------//
    {"python_owns",
     (PyCFunction)PyConduit_Node_python_owns,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"python_attach",
     (PyCFunction)PyConduit_Node_python_attach,
      METH_NOARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
    {"python_detach",
     (PyCFunction)PyConduit_Node_python_detach,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"set",
     (PyCFunction)PyConduit_Node_set,
     METH_VARARGS,
     "Sets the node"},
    {"set_external",
     (PyCFunction)PyConduit_Node_set_external,
     METH_VARARGS,
     "Sets the node's data to an external numpy array"},
    //-----------------------------------------------------------------------//
    {"fetch",
     (PyCFunction)PyConduit_Node_fetch,
     METH_VARARGS, 
     "Fetches the node at a given path"},
    //-----------------------------------------------------------------------//
    {"child",
     (PyCFunction)PyConduit_Node_child,
     METH_VARARGS, 
     "Retrieves the child node at a given index"},
    //-----------------------------------------------------------------------//
    {"number_of_children",
      (PyCFunction)PyConduit_Node_number_of_children,
      METH_NOARGS, 
      "Number of child nodes"},
    //-----------------------------------------------------------------------//
    {"has_path",
     (PyCFunction)PyConduit_Node_has_path,
     METH_VARARGS, 
     "Returns if this node has the given path"},
    //-----------------------------------------------------------------------//
    {"paths",
     (PyCFunction)PyConduit_Node_paths,
     METH_NOARGS, 
     "Returns a list with this node's child paths"},
     //-----------------------------------------------------------------------//
     {"info",
      (PyCFunction)PyConduit_Node_info,
      METH_VARARGS, 
      "Returns a node populated with the memory space details for this node"},
    //-----------------------------------------------------------------------//
    {"append",
     (PyCFunction)PyConduit_Node_append,
     METH_NOARGS, 
     "Appends a node (coarse to conduit list)"},
    //-----------------------------------------------------------------------//
    {"remove", 
     (PyCFunction)PyConduit_Node_remove,
     METH_VARARGS | METH_KEYWORDS, 
     "Remove as node at a given index or path."},
    //-----------------------------------------------------------------------//
    {"value",
     (PyCFunction)PyConduit_Node_value,
     METH_NOARGS, 
     "Value access for leaf nodes"},
    //-----------------------------------------------------------------------//
    {"generate",
     (PyCFunction)PyConduit_Node_generate,
     METH_VARARGS,  // will become kwargs
     "Generate a node"},
    //-----------------------------------------------------------------------//
    {"save",
     (PyCFunction)PyConduit_Node_save,
     METH_VARARGS | METH_KEYWORDS, 
     "Saves a node to a file pair"},
    //-----------------------------------------------------------------------//
    {"load",
     (PyCFunction)PyConduit_Node_load,
     METH_VARARGS | METH_KEYWORDS, 
     "Loads a node from a file pair, file with schema, or file with protocol"},
    //-----------------------------------------------------------------------//
    {"mmap",
     (PyCFunction)PyConduit_Node_mmap,
     METH_VARARGS, // will become kwargs
     "Memory Maps a node from a file pair, file with schema, or file with protocol"},
    //-----------------------------------------------------------------------//
    {"set_path",
     (PyCFunction)PyConduit_Node_set_path,
     METH_VARARGS,
     "Sets the node at the given path"},
    //-----------------------------------------------------------------------//
    {"schema",
     (PyCFunction)PyConduit_Node_schema, 
     METH_NOARGS,
     "Returns the schema for the node"}, 
     //-----------------------------------------------------------------------//
     {"dtype",
      (PyCFunction)PyConduit_Node_dtype, 
      METH_NOARGS,
      "Returns the conduit DataType for the node"}, 
    //-----------------------------------------------------------------------//
    {"is_root",
     (PyCFunction)PyConduit_Node_is_root, 
     METH_NOARGS,
     "Returns if this node is the root of the hierarchy."}, 
    //-----------------------------------------------------------------------//
    {"parent",
     (PyCFunction)PyConduit_Node_parent, 
     METH_NOARGS,
     "Returns this nodes parent, or None if no parent"}, 
    //-----------------------------------------------------------------------//
    {"total_bytes",
     (PyCFunction)PyConduit_Node_total_bytes, 
     METH_NOARGS,
     "Returns the total bytes of this node's data"}, 
    //-----------------------------------------------------------------------//
    {"total_bytes_compact",
     (PyCFunction)PyConduit_Node_total_bytes_compact, 
     METH_NOARGS,
     "Returns the total bytes of compact rep of node's data"}, 
    //-----------------------------------------------------------------------//
    {"is_compact",
     (PyCFunction)PyConduit_Node_is_compact, 
     METH_NOARGS,
     "Returns if this node's data is in compact form"}, 
     //-----------------------------------------------------------------------//
     {"to_json",
      (PyCFunction)PyConduit_Node_to_json, 
      METH_VARARGS| METH_KEYWORDS,
      "Returns a JSON string representation of the node."},
     //-----------------------------------------------------------------------//
     {"children",
      (PyCFunction)PyConduit_Node_iter, 
      METH_NOARGS,
      "Returns a NodeIterator for this node"}, 
    //-----------------------------------------------------------------------//
    // end node methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
static PyMappingMethods node_as_mapping = {
   (lenfunc)0,    // len operator is not supported
   (binaryfunc)PyConduit_Node_GetItem,
   (objobjargproc)PyConduit_Node_SetItem,
};

//---------------------------------------------------------------------------//
static PyTypeObject PyConduit_Node_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "Node",
   sizeof(PyConduit_Node),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyConduit_Node_dealloc,                   /* tp_dealloc */
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
   (reprfunc)PyConduit_Node_str,                         /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "Conduit node objects",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   (getiterfunc)PyConduit_Node_iter, /* iter */
   0, /* iternext */
   PyConduit_Node_METHODS,
   0,
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyConduit_Node_init,
   0, /* alloc */
   PyConduit_Node_new,                                   /* new */
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
PyConduit_about(PyObject *self)
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
     (PyCFunction)PyConduit_about,
      METH_NOARGS,
      NULL},
    //-----------------------------------------------------------------------//
    // end conduit methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, METH_VARARGS, NULL}
};


//---------------------------------------------------------------------------//
static int
PyConduit_Schema_Check(PyObject* obj)
{
    return (PyObject_TypeCheck(obj, &PyConduit_Schema_TYPE));
}


//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Schema_python_wrap(Schema *schema, int python_owns)
{
    PyTypeObject *type = (PyTypeObject*)&PyConduit_Schema_TYPE;

    PyConduit_Schema *retval = (PyConduit_Schema*)type->tp_alloc(type, 0);
    retval->schema = schema;
    retval->python_owns = python_owns;
    return ((PyObject*)retval);
}

//---------------------------------------------------------------------------//
static PyConduit_Schema *
PyConduit_Schema_python_create()
{
    Schema *schema = new Schema();
    // python_owns = 1
    return (PyConduit_Schema *)PyConduit_Schema_python_wrap(schema,1); 
}


//---------------------------------------------------------------------------//
static int
PyConduit_Node_Check(PyObject *obj)
{
    return (PyObject_TypeCheck(obj, &PyConduit_Node_TYPE));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_python_wrap(Node *node, int python_owns)
{
    PyTypeObject* type = (PyTypeObject*)&PyConduit_Node_TYPE;
    PyConduit_Node* retval = (PyConduit_Node*)type->tp_alloc(type, 0);
    retval->node = node;
    retval->python_owns = python_owns;
    return ((PyObject*)retval);
}

//---------------------------------------------------------------------------//
static PyConduit_Node *
PyConduit_Node_python_create()
{
    Node *node = new Node();
    // python_owns = 1
    return (PyConduit_Node *)PyConduit_Node_python_wrap(node,1); 
}


//---------------------------------------------------------------------------//
static int
PyConduit_Node_SetFromPython(Node &node,
                             PyObject *value)
{
    if (PyConduit_Node_Check(value))
    {
        node = *((PyConduit_Node*)value)->node;
    }
    else if (PyConduit_Schema_Check(value))
    {
        node = *((PyConduit_Schema*)value)->schema;
    }
    else if (PyString_Check(value))
    {
        char *cstr = PyString_AsString(value);
        node = cstr;
        PyString_AsString_Cleanup(cstr);
    }
    else if (PyInt_Check(value))
    {
        node = PyInt_AsLong(value);
    }
    else if (PyLong_Check(value))
    {
        node = PyLong_AsLong(value);
    }
    else if (PyFloat_Check(value))
    {
        node = PyFloat_AS_DOUBLE(value);
    }
    else if (PyArray_Check(value))
    {
        PyArray_Descr *desc = PyArray_DESCR((PyArrayObject*)value);
        PyArrayObject *py_arr = (PyArrayObject*)value;
        npy_intp num_ele = PyArray_SIZE(py_arr);
        switch (desc->type_num) 
        {
            case NPY_UINT8 :
            {
                node.set(DataType::uint8(num_ele));
                uint8_array c_arr = node.value();
                PyConduit_Fill_DataArray_From_PyArray(c_arr, py_arr);
                break;
            }
            case NPY_UINT16 :
            {
                node.set(DataType::uint16(num_ele));
                uint16_array c_arr = node.value();
                PyConduit_Fill_DataArray_From_PyArray(c_arr, py_arr);
                break;
            }
            case NPY_UINT32 :
            {
                node.set(DataType::uint32(num_ele));
                uint32_array c_arr = node.value();
                PyConduit_Fill_DataArray_From_PyArray(c_arr, py_arr);
                break;
            }
            case NPY_UINT64 :
            {
                node.set(DataType::uint64(num_ele));
                uint64_array c_arr = node.value();
                PyConduit_Fill_DataArray_From_PyArray(c_arr, py_arr);
                break;
            }
            case NPY_INT8 :
            {
                node.set(DataType::int8(num_ele));
                uint8_array c_arr = node.value();
                PyConduit_Fill_DataArray_From_PyArray(c_arr, py_arr);
                break;
            }
            case NPY_INT16 :
            {
                node.set(DataType::int16(num_ele));
                uint16_array c_arr = node.value();
                PyConduit_Fill_DataArray_From_PyArray(c_arr, py_arr);
                break;
            }
            case NPY_INT32 :
            {
                node.set(DataType::int32(num_ele));
                uint32_array c_arr = node.value();
                PyConduit_Fill_DataArray_From_PyArray(c_arr, py_arr);
                break;
            }
            case NPY_INT64 :
            {
                node.set(DataType::int64(num_ele));
                uint64_array c_arr = node.value();
                PyConduit_Fill_DataArray_From_PyArray(c_arr, py_arr);
                break;
            }
            case NPY_FLOAT32 :
            {
                node.set(DataType::float32(num_ele));
                float32_array c_arr = node.value();
                PyConduit_Fill_DataArray_From_PyArray(c_arr, py_arr);
                break;
            }
            case NPY_FLOAT64 :
            {
                node.set(DataType::float64(num_ele));
                float64_array c_arr = node.value();
                PyConduit_Fill_DataArray_From_PyArray(c_arr, py_arr);
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
    void* data = node.element_ptr(0);
    npy_intp len = dtype.number_of_elements();
    // TODO: This only deals with contiguous data?
    if (len == 1) 
    {
        retval = PyArray_Scalar(data, descr, NULL);
    }
    else 
    {
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
            retval = PyConduit_Node_python_wrap(&node,0);
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
conduit_python_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static int 
conduit_python_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static struct PyModuleDef conduit_python_module_def = 
{
        PyModuleDef_HEAD_INIT,
        "conduit_python",
        NULL,
        sizeof(struct module_state),
        conduit_python_funcs,
        NULL,
        conduit_python_traverse,
        conduit_python_clear,
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
PyObject *CONDUIT_PYTHON_API PyInit_conduit_python(void)
#else
void CONDUIT_PYTHON_API initconduit_python(void)
#endif
//---------------------------------------------------------------------------//
{    
    //-----------------------------------------------------------------------//
    // create our main module
    //-----------------------------------------------------------------------//

#if defined(IS_PY3K)
    PyObject *conduit_module = PyModule_Create(&conduit_python_module_def);
#else
    PyObject *conduit_module = Py_InitModule("conduit_python",
                                             conduit_python_funcs);
#endif

    if(conduit_module == NULL)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    struct module_state *st = GETSTATE(conduit_module);
    
    st->error = PyErr_NewException("conduit_python.Error", NULL, NULL);
    if (st->error == NULL)
    {
        Py_DECREF(conduit_module);
        PY_MODULE_INIT_RETURN_ERROR;
    }

    //-----------------------------------------------------------------------//
    // init our custom types
    //-----------------------------------------------------------------------//

    if (PyType_Ready(&PyConduit_DataType_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    if (PyType_Ready(&PyConduit_Schema_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    if (PyType_Ready(&PyConduit_Generator_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    if (PyType_Ready(&PyConduit_NodeIterator_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    if (PyType_Ready(&PyConduit_Node_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    //-----------------------------------------------------------------------//
    // add DataType
    //-----------------------------------------------------------------------//
    
    Py_INCREF(&PyConduit_DataType_TYPE);
    PyModule_AddObject(conduit_module,
                       "DataType",
                       (PyObject*)&PyConduit_DataType_TYPE);
    //-----------------------------------------------------------------------//
    // add Schema
    //-----------------------------------------------------------------------//

    Py_INCREF(&PyConduit_Schema_TYPE);
    PyModule_AddObject(conduit_module,
                       "Schema",
                       (PyObject*)&PyConduit_Schema_TYPE);

    //-----------------------------------------------------------------------//
    // add Generator
    //-----------------------------------------------------------------------//

    Py_INCREF(&PyConduit_Generator_TYPE);
    PyModule_AddObject(conduit_module,
                       "Generator",
                       (PyObject*)&PyConduit_Generator_TYPE);

    //-----------------------------------------------------------------------//
    // add NodeIterator
    //-----------------------------------------------------------------------//

    Py_INCREF(&PyConduit_NodeIterator_TYPE);
    PyModule_AddObject(conduit_module,
                       "NodeIterator",
                       (PyObject*)&PyConduit_NodeIterator_TYPE);

    //-----------------------------------------------------------------------//
    // add Node
    //-----------------------------------------------------------------------//

    Py_INCREF(&PyConduit_Node_TYPE);
    PyModule_AddObject(conduit_module,
                       "Node",
                       (PyObject*)&PyConduit_Node_TYPE);

    // req setup for numpy
    import_array();
    
#if defined(IS_PY3K)
    return conduit_module;
#endif

}

