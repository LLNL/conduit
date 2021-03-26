// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2021, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
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
/// file: python_interpreter.hpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
///
/// Simple C++ Embeddable Python Interpreter.
///
/// ADAPTED FROM https://github.com/Alpine-DAV/ascent/tree/develop/src/flow
//-----------------------------------------------------------------------------

#ifndef PYTHON_INTERPRETER_HPP
#define PYTHON_INTERPRETER_HPP

#include <Python.h>
#include <string>

class PythonInterpreter
{
public:
                 PythonInterpreter();
    virtual     ~PythonInterpreter();

    /// instance lifetime control
    bool         initialize(int argc=0, char **argv=NULL);

    bool         is_running() { return m_running; }

    /// Note: blows away everything in the main dict
    /// use with caution!
    void         reset();
    void         shutdown();

    /// echo (default = false)
    ///  when enabled, controls if contents of execd python 
    //   scripts are echoed to conduit info
    bool         echo_enabled() const { return m_echo; }
    /// change echo setting
    void         set_echo(bool value) { m_echo = value; }

    void         set_program_name(const char *name);
    void         set_argv(int argc, char **argv);

    /// helper to add a system path to access new modules
    bool         add_system_path(const std::string &path);

    /// script exec
    bool         run_script(const std::string &script);
    bool         run_script_file(const std::string &fname);
    
    /// script exec in specific dict
    bool         run_script(const std::string &script,
                            PyObject *py_dict);
    bool         run_script_file(const std::string &fname,
                                 PyObject *py_dict);

    /// set into global dict
    bool         set_global_object(PyObject *py_obj,
                                   const std::string &name);
    /// fetch from global dict, returns borrowed reference
    PyObject    *get_global_object(const std::string &name);
    /// access global dict object
    PyObject    *global_dict() { return m_py_global_dict; }

    /// set into given dict
    bool         set_dict_object(PyObject *py_dict,
                                 PyObject *py_obj,
                                 const std::string &name);
    /// fetch from given dict, returns borrowed reference
    PyObject    *get_dict_object(PyObject *py_dict, 
                                 const std::string &name);

    /// error checking
    bool         check_error();
    void         clear_error();
    std::string  error_message() const { return m_error_msg; }

    /// helpers to obtain values from basic objects
    static bool  PyObject_to_double(PyObject *py_obj,
                                    double &res);

    static bool  PyObject_to_string(PyObject *py_obj,
                                    std::string &res);

    static bool  PyObject_to_int(PyObject *py_obj,
                                 int &res);

private:
    bool         PyTraceback_to_string(PyObject *py_etype,
                                       PyObject *py_eval,
                                       PyObject *py_etrace,
                                       std::string &res);

    bool         m_handled_init;
    bool         m_running;
    bool         m_echo;
    bool         m_error;
    std::string  m_error_msg;

    PyObject    *m_py_main_module;
    PyObject    *m_py_global_dict;

    PyObject    *m_py_trace_module;
    PyObject    *m_py_sio_module;
    PyObject    *m_py_trace_print_exception_func;
    PyObject    *m_py_sio_class;

};



#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


