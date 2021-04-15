// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_cpp_and_fort_ex.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Demos creating and passing Conduit Nodes between C++ and Python
//-----------------------------------------------------------------------------

#include <iostream>

#include "conduit.hpp"
#include "conduit_relay.hpp"
#include "conduit_blueprint.hpp"

#include "python_interpreter.hpp"
// conduit python module capi header
#include "conduit_python.hpp"


// single python interp instance for our example.
PythonInterpreter *interp = NULL;

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


int main(int argc, char **argv)
{

    PythonInterpreter *pyintp = init_python_interpreter();

    // create a python interp, wrap conduit node into there

    conduit::Node n;
    n["values"] = { 1.0,2.0,3.0,4.0};
    n["shape"]  = {2,2};

    std::cout << "Hello from C++, here is the Node we created:" << std::endl;
    n.print();

    // create py object to wrap the conduit node
    PyObject *py_node = PyConduit_Node_Python_Wrap(&n,
                                                   0); // python owns => false

    // get global dict and insert wrapped conduit node
    PyObject *py_mod_dict = pyintp->global_dict();

    pyintp->set_dict_object(py_mod_dict,
                            py_node,
                            "my_node");

    //
    // NOTE: we aren't checking pyintp->run_script return to simplify
    //       this example -- but you should check in real cases!
    //

    // access python in cpp
    pyintp->run_script("print(my_node)");
    pyintp->run_script("import numpy");
    pyintp->run_script("vals_view = my_node['values'].reshape(my_node['shape'])");
    pyintp->run_script("print('Hello from Python, here is the vals_view')");
    pyintp->run_script("print(vals_view)");


    std::string py_name = "my_py_node";

    // create a node in python, and access it in c++
    std::ostringstream oss;
    oss << py_name << " = conduit.Node()" << std::endl
        << py_name << "['values'] = [10.0,20.0,30.0,40.0]" << std::endl
        << py_name << "['shape']  = [2,2]" << std::endl
        << "print('Hello from python, I created:')" << std::endl
        << "print(" << py_name << ")" << std::endl;

    pyintp->run_script(oss.str());
    
    // fetch wrapped conduit node
    PyObject *py_obj = pyintp->get_dict_object(py_mod_dict,
                                               py_name);

    if(!PyConduit_Node_Check(py_obj))
    {
        // error!
    }

    conduit::Node *cpp_res = PyConduit_Node_Get_Node_Ptr(py_obj);

    std::cout << "Hello from C++, here is the Node we got from python:" << std::endl;
    cpp_res->print();

}


