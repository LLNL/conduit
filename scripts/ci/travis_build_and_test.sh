#!/bin/bash
###############################################################################
# Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-666778
#
# All rights reserved.
#
# This file is part of Conduit.
#
# For details, see: http://software.llnl.gov/conduit/.
#
# Please also read conduit/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

# stop on first error and echo commands
set -ev

if [ "${DOCKER}" = "ON" ]; then
    # test docker example, which uses uberenv + spack
    cd src/examples/docker/ubuntu/
    ./example_build.sh
else
    # add installed tpls to our path
    export PATH=${TRAVIS_HOME}/cmake/bin:$PATH
    export PATH=${TRAVIS_HOME}/miniconda/bin:$PATH
    # echo cmake version
    cmake --version
    cd $TRAVIS_BUILD_DIR
    # create out-of-source build dir, and an install dir
    mkdir travis-debug-build
    mkdir travis-debug-install
    cd    travis-debug-build
    # build type and install loc
    CMAKE_OPTS="-DCMAKE_BUILD_TYPE=Debug"
    CMAKE_OPTS="${CMAKE_OPTS} -DCMAKE_INSTALL_PREFIX=../travis-debug-install"
    # c & c++ compilers
    CMAKE_OPTS="${CMAKE_OPTS} -DCMAKE_C_COMPILER=${CONDUIT_CC}"
    CMAKE_OPTS="${CMAKE_OPTS} -DCMAKE_CXX_COMPILER=${CONDUIT_CXX}"
    # shared or static libs
    CMAKE_OPTS="${CMAKE_OPTS} -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
    # enable fortran support
    CMAKE_OPTS="${CMAKE_OPTS} -DCMAKE_Fortran_COMPILER=${CONDUIT_FC}"
    CMAKE_OPTS="${CMAKE_OPTS} -DENABLE_FORTRAN=ON"
    # enable python support
    CMAKE_OPTS="${CMAKE_OPTS} -DENABLE_PYTHON=ON"
    # enable hdf5 support
    CMAKE_OPTS="${CMAKE_OPTS} -DHDF5_DIR=${TRAVIS_HOME}/miniconda"
    # enable coverage (only when using shared libs case)
    CMAKE_OPTS="${CMAKE_OPTS} -DENABLE_COVERAGE=${ENABLE_COVERAGE}"
    # configure with cmake
    cmake  ${CMAKE_OPTS} ../src
    # build, test, and install
    make
    env CTEST_OUTPUT_ON_FAILURE=1 make test
    make install
    # test our examples that demo using an installed conduit
    # using with cmake example
    cd ${TRAVIS_BUILD_DIR}/src/examples/using-with-cmake
    mkdir build
    cd build
    cmake -DCONDUIT_DIR=${TRAVIS_BUILD_DIR}/travis-debug-install ../
    make
    ./example
    # using with make example
    cd ${TRAVIS_BUILD_DIR}/src/examples/using-with-make
    env CXX=${CONDUIT_CXX} CONDUIT_DIR=${TRAVIS_BUILD_DIR}/travis-debug-install HDF5_DIR=${TRAVIS_HOME}/miniconda/lib make
    env LD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/travis-debug-install/lib/:${TRAVIS_HOME}/miniconda/lib ./example
fi

