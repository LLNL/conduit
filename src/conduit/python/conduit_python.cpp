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

//---------------------------------------------------------------------------//
struct PyConduit_DataType {
    PyObject_HEAD
    DataType dtype; // NoteIterator is light weight, we can deal with copies
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
static int
PyConduit_DataType_init(PyConduit_Schema* self,
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
        return (0);
    }
    return (0);
}

//---------------------------------------------------------------------------//
static void
PyConduit_DataType_dealloc(PyConduit_DataType *self)
{
    self->ob_type->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_DataType_str(PyConduit_DataType *self)
{
   std::string output = self->dtype.to_json();
   return (Py_BuildValue("s", output.c_str()));
}


//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
//     void       set(const DataType& dtype);
//     void       set(index_t dtype_id);
//     void       set(index_t dtype_id,
//                    index_t num_elements,
//                    index_t offset,
//                    index_t stride,
//                    index_t element_bytes,
//                    index_t endianness);


//     void       set_number_of_elements(index_t v)
//     void       set_offset(index_t v)
//     void       set_stride(index_t v)
//     void       set_element_bytes(index_t v)
//     void       set_endianness(index_t v)
//                     { m_endianness = v;}
//


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
//     index_t     id()    const { return m_id;}
//     index_t     total_bytes()   const;
//     index_t     total_bytes_compact() const;
//     bool        is_compact() const;
//     bool        is_compatible(const DataType& type) const;
//
//     bool        is_number()           const;
//     bool        is_float()            const;
//     bool        is_integer()          const;
//     bool        is_signed_integer()   const;
//     bool        is_unsigned_integer() const;

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
PyConduit_DataType_is_float(PyConduit_DataType *self)
{
    if(self->dtype.is_float())
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
    if(self->dtype.is_float())
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


//
//
//     index_t    number_of_elements()  const { return m_num_ele;}
//     index_t    offset()              const { return m_offset;}
//     index_t    stride()              const { return m_stride;}
//     index_t    element_bytes()       const { return m_ele_bytes;}
//     index_t    endianness()          const { return m_endianness;}
//     index_t    element_index(index_t idx) const;

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
// Schema methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyConduit_DataType_METHODS[] = {
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
    {"is_float",
     (PyCFunction)PyConduit_DataType_is_float,
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
    {"name_to_id",
     PyConduit_DataType_name_to_id,
     METH_VARARGS | METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"id_to_name",
     PyConduit_DataType_id_to_name,
     METH_VARARGS | METH_CLASS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    // end DataType methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
static PyTypeObject PyConduit_DataType_TYPE = {
   PyObject_HEAD_INIT(NULL)
   0,
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
PyConduit_DataType_check(PyObject* obj)
{
    return (PyObject_TypeCheck(obj, &PyConduit_DataType_TYPE));
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
PyConduit_Schema_New(PyTypeObject* type,
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
             self->schema = new Schema(PyString_AsString(value));
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
    if (self->python_owns)
    {
        delete self->schema;
    }
    self->ob_type->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Schema_str(PyConduit_Schema *self)
{
   std::string output = self->schema->to_json();
   return (Py_BuildValue("s", output.c_str()));
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
    // end Schema methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
static PyTypeObject PyConduit_Schema_TYPE = {
   PyObject_HEAD_INIT(NULL)
   0,
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
   PyConduit_Schema_New,                                   /* new */
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
    self->ob_type->tp_free((PyObject*)self);
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
   PyObject_HEAD_INIT(NULL)
   0,
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
/// TODO: We may be able to use a conduit data array here to avoid copies
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
// begin Node python special methods
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_New(PyTypeObject* type,
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

    self->ob_type->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_str(PyConduit_Node* self)
{
   std::string output = self->node->to_json();
   return (Py_BuildValue("s", output.c_str()));
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
PyConduit_Node_data(PyConduit_Node* self)
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
         PyErr_SetString(PyExc_TypeError, "Save file path must be a string");
         return NULL;
     }
     
     self->node->generate(std::string(json_schema));
     Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_save(PyConduit_Node* self,
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
PyConduit_Node_load(PyConduit_Node* self,
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
    return (PyConduit_Schema_python_wrap(self->node->schema_pointer(),0));
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
PyConduit_Node_has_parent(PyConduit_Node *self)
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
PyConduit_Node_parent(PyConduit_Node* self)
{
    if(~self->node->has_parent())
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

    return (PyConduit_Node_SetFromPython(node, value));
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_iter(PyObject *self)
{
    PyConduit_Node *py_n = (PyConduit_Node *)self;

    PyConduit_NodeIterator *retval = NULL;
    retval = PyConduit_NodeIterator_python_create();
    retval->itr =  py_n->node->iterator();

    return ((PyObject *)retval);
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_set(PyConduit_Node* self,
                   PyObject* args)
{
    PyObject* value;
    if (!PyArg_ParseTuple(args, "O", &value)) {
         return (NULL);
    }

    if (PyConduit_Node_SetFromPython(*self->node, value)) {
         return (NULL);
    } else {
        Py_RETURN_NONE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyConduit_Node_set_path(PyConduit_Node* self,
                       PyObject* args)
{
    PyObject* value, *path;
    if (!PyArg_ParseTuple(args, "OO", &path, &value)) {
         return (NULL);
    }

    if (PyConduit_Node_SetItem(self, path, value)) {
        return (NULL);
    } else {
        Py_RETURN_NONE;
    }
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
    {"remove", // PyCFunctionWithKeywords ?
     (PyCFunction)PyConduit_Node_remove,
     METH_KEYWORDS, 
     "Remove as node at a given index or path."},
    //-----------------------------------------------------------------------//
    {"data",
     (PyCFunction)PyConduit_Node_data,
     METH_NOARGS, 
     "{data val of node}"},
    //-----------------------------------------------------------------------//
    {"generate",
     (PyCFunction)PyConduit_Node_generate,
     METH_VARARGS,  // will become kwargs
     "Generate a node"},
    //-----------------------------------------------------------------------//
    {"save",
     (PyCFunction)PyConduit_Node_save,
     METH_VARARGS, 
     "Saves a node to a file pair"},
    //-----------------------------------------------------------------------//
    {"load",
     (PyCFunction)PyConduit_Node_load,
     METH_VARARGS,  // will become kwargs
     "Loads a node from a file pair"},
    //-----------------------------------------------------------------------//
    {"mmap",
     (PyCFunction)PyConduit_Node_mmap,
     METH_VARARGS, // will become kwargs
     "Memory Maps a node from a file pair"},
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
    {"has_parent",
     (PyCFunction)PyConduit_Node_has_parent, 
     METH_NOARGS,
     "Returns of the node has a parent node"}, 
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
     {"iterator",
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
   PyObject_HEAD_INIT(NULL)
   0,
   //PyObject_VAR_HEAD
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
   PyConduit_Node_New,                                   /* new */
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
// Main entry point
//---------------------------------------------------------------------------//
extern "C" void
CONDUIT_PYTHON_API initconduit_python(void)
{    
    //-----------------------------------------------------------------------//
    // create our main module
    //-----------------------------------------------------------------------//

    PyObject *conduit_module =  Py_InitModule("conduit_python",
                                              conduit_python_funcs);
    
    //-----------------------------------------------------------------------//
    // init our custom types
    //-----------------------------------------------------------------------//

    if (PyType_Ready(&PyConduit_DataType_TYPE) < 0)
        return;

    if (PyType_Ready(&PyConduit_Schema_TYPE) < 0)
        return;

    if (PyType_Ready(&PyConduit_NodeIterator_TYPE) < 0)
        return;

    if (PyType_Ready(&PyConduit_Node_TYPE) < 0)
        return;

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
}

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
    if (PyConduit_Node_Check(value)) {
        node = *((PyConduit_Node*)value)->node;
    } else if (PyConduit_Schema_Check(value)) {
        node = *((PyConduit_Schema*)value)->schema;
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
