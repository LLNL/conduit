#!/bin/bash
###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

set -ev

##########################################################
# test our installed python example
##########################################################

if [ "${ENABLE_PYTHON}" == "ON" ]; then
    if [ "${BUILD_SHARED_LIBS}" == "ON" ]; then
        cd ${TRAVIS_BUILD_DIR}/travis-debug-install/
        env LD_LIBRARY_PATH=${RUN_LIB_PATH} ./bin/run_python_with_conduit.sh <  examples/conduit/python/conduit_python_example.py 
    fi
fi


