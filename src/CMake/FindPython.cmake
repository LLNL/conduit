# Find the interpreter first
find_package(PythonInterp REQUIRED)
if(PYTHONINTERP_FOUND)
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                                "import sys;from distutils.sysconfig import get_python_inc;sys.stdout.write(get_python_inc())"
                        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
                        ERROR_VARIABLE ERROR_FINDING_INCLUDES)
        
        get_filename_component(PYTHON_BIN_DIR ${PYTHON_EXECUTABLE} PATH)
        set(PYTHON_GLOB_TEST "${PYTHON_BIN_DIR}/../lib/libpython*")
        FILE(GLOB PYTHON_GLOB_RESULT ${PYTHON_GLOB_TEST})
        get_filename_component(PYTHON_LIBRARY "${PYTHON_GLOB_RESULT}" ABSOLUTE)
        MESSAGE(STATUS "{PythonLibs from PythonInterp} using: PYTHON_LIBRARY=${PYTHON_LIBRARY}")
        find_package(PythonLibs)
endif()
