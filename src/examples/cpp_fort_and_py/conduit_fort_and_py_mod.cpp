// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_fort_and_py_mod.cpp
///

//-----------------------------------------------------------------------------
//
// c functions that we bind to create the fortran conduit_fort_and_py_mod
// 
//-----------------------------------------------------------------------------

#include <conduit.hpp>
#include <conduit_cpp_to_c.hpp>

// conduit python module capi header
#include "conduit_python.hpp"

// embedded interp
#include "python_interpreter.hpp"

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------


// single python interp instance for our example module.
PythonInterpreter *interp = NULL;

extern "C" {

// returns our static instance of our python interpreter
// if not already inited initializes it
PythonInterpreter *init_python_interpreter()
{
    if( interp == NULL)
    {
        interp = new PythonInterpreter();
        if( !interp->initialize() )
        {
            std::cout << "ERROR: interp->initialize() failed " << std::endl;
           return NULL;
        }
        // setup for conduit python c api
        if(!interp->run_script("import conduit"))
        {
            std::cout << "ERROR: `import conduit` failed" << std::endl;
           return NULL;
        }

        if(import_conduit() < 0)
        {
           std::cout << "failed to import Conduit Python C-API";
           return NULL;
        }

        // // Turn this on if you want to see every line
        // // the python interpreter executes
        //interp->set_echo(true);
    }
    return interp;
}

//-----------------------------------------------------------------------------
// access node passed from fortran to python
void
conduit_fort_to_py(conduit_node *data)
{
    PythonInterpreter *pyintp = init_python_interpreter();

    // get cpp ref to passed node
    conduit::Node &n = conduit::cpp_node_ref(data);

    // create py object to wrap the conduit node
    PyObject *py_node = PyConduit_Node_Python_Wrap(&n,
                                                   0); // python owns => false

    // get global dict and insert wrapped conduit node
    PyObject *py_mod_dict =  pyintp->global_dict();

    pyintp->set_dict_object(py_mod_dict,
                            py_node,
                            "my_node");

    //
    // NOTE: we aren't checking pyintp->run_script return to simplify
    //       this example -- but you should check in real cases!
    //

    pyintp->run_script("print('Hello from Python, here is what you passed:')");
    pyintp->run_script("print(my_node)");
    pyintp->run_script("vals_view = my_node['values'].reshape(my_node['shape'])");
    pyintp->run_script("print(vals_view)");
}

//-----------------------------------------------------------------------------
// create a node in python and return it for access in fortran
conduit_node *
conduit_fort_from_py(const char *py_name)
{
    PythonInterpreter *pyintp = init_python_interpreter();

    std::ostringstream oss;
    oss << py_name << " = conduit.Node()" << std::endl
        << py_name << "['values'] = [10.0,20.0,30.0,40.0]" << std::endl
        << py_name << "['shape']  = [2,2]" << std::endl
        << "print('Hello from python, I created:')" << std::endl
        << "print(" << py_name << ")" << std::endl;

    //
    // NOTE: we aren't checking pyintp->run_script return to simplify
    //       this example -- but you should check in real cases!
    //

    pyintp->run_script(oss.str());

    // get global dict and fetch wrapped conduit node
    PyObject *py_mod_dict =  pyintp->global_dict();

    PyObject *py_obj = pyintp->get_dict_object(py_mod_dict,
                                               py_name);

    if(!PyConduit_Node_Check(py_obj))
    {
        // error!
    }

    conduit::Node *cpp_res = PyConduit_Node_Get_Node_Ptr(py_obj);
    // return the c pointer
    return conduit::c_node(cpp_res);
}

}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------
