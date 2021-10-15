// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.


//-----------------------------------------------------------------------------
// -- Python includes (these must be included first) --
//-----------------------------------------------------------------------------
#include <Python.h>
#include <structmember.h>
#include "bytesobject.h"

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
// -- standard lib includes --
//-----------------------------------------------------------------------------
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------//
// conduit includes
//---------------------------------------------------------------------------//
#include "conduit.hpp"
#include "conduit_blueprint.hpp"

#include "conduit_blueprint_python_exports.h"

// conduit python module capi header
#include "conduit_python.hpp"

using namespace conduit;

//---------------------------------------------------------------------------//
// conduit::blueprint::table::examples::basic
//---------------------------------------------------------------------------//

// doc string
const char *PyBlueprint_table_examples_basic_doc_str =
"basic(nx, ny, nz, res)\n"
"\n"
"Creates a basic table blueprint example.\n"
"\n"
"https://llnl-conduit.readthedocs.io/en/latest/blueprint_table.html#basic\n"
"\n"
"Arguments:\n"
" nx: Number of points in x.\n"
" ny: Number of points in y.\n"
" nz: Number of points in z.\n"
" dest: Table output (conduit.Node instance)\n";

// python func
static PyObject *
PyBlueprint_table_examples_basic(PyObject *, //self
                                PyObject *args,
                                PyObject *kwargs)
{
    Py_ssize_t nx = 0;
    Py_ssize_t ny = 0;
    Py_ssize_t nz = 0;

    PyObject   *py_node  = NULL;

    static const char *kwlist[] = {"nx",
                                   "ny",
                                   "nz",
                                   "res",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "nnnO",
                                     const_cast<char**>(kwlist),
                                     &nx,
                                     &ny,
                                     &nz,
                                     &py_node))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'res' argument must be a "
                        "conduit.Node instance");
        return NULL;
    }

    Node &node = *PyConduit_Node_Get_Node_Ptr(py_node);

    blueprint::table::examples::basic(nx,
                                      ny,
                                      nz,
                                      node);

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// Python Module Method Defs
//---------------------------------------------------------------------------//
static PyMethodDef blueprint_table_examples_python_funcs[] =
{
    //-----------------------------------------------------------------------//
    {"basic",
     (PyCFunction)PyBlueprint_table_examples_basic,
      METH_VARARGS | METH_KEYWORDS,
      PyBlueprint_table_examples_basic_doc_str},
    //-----------------------------------------------------------------------//
    // end methods table
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
blueprint_table_examples_python_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static int 
blueprint_table_examples_python_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static struct PyModuleDef blueprint_table_examples_python_module_def = 
{
        PyModuleDef_HEAD_INIT,
        "blueprint_table_examples_python",
        NULL,
        sizeof(struct module_state),
        blueprint_table_examples_python_funcs,
        NULL,
        blueprint_table_examples_python_traverse,
        blueprint_table_examples_python_clear,
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
CONDUIT_BLUEPRINT_PYTHON_API PyObject *PyInit_conduit_blueprint_table_examples_python(void)
#else
CONDUIT_BLUEPRINT_PYTHON_API void initconduit_blueprint_table_examples_python(void)
#endif
//---------------------------------------------------------------------------//
{
    //-----------------------------------------------------------------------//
    // create our main module
    //-----------------------------------------------------------------------//

#if defined(IS_PY3K)
    PyObject *py_module = PyModule_Create(&blueprint_table_examples_python_module_def);
#else
    PyObject *py_module = Py_InitModule((char*)"conduit_blueprint_table_examples_python",
                                               blueprint_table_examples_python_funcs);
#endif

    if(py_module == NULL)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    struct module_state *st = GETSTATE(py_module);

    st->error = PyErr_NewException((char*)"blueprint_table_examples_python.Error",
                                   NULL,
                                   NULL);
    if (st->error == NULL)
    {
        Py_DECREF(py_module);
        PY_MODULE_INIT_RETURN_ERROR;
    }

    // setup for conduit python c api
    if(import_conduit() < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }


#if defined(IS_PY3K)
    return py_module;
#endif

}
