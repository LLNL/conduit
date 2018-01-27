###############################################################################
# Copyright (c) Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
#!/bin/bash
set -ev

if [ "${ENABLE_COVERAGE}" == "ON" ]; then
    echo "skipping examples vs install tests (ENABLE_COVERAGE=ON)"
    exit 0
fi

##########################################################
# test our examples that demo using an installed conduit
##########################################################

##########################################
# using with cmake example
##########################################
cd ${TRAVIS_BUILD_DIR}/travis-debug-install/examples/using-with-cmake
mkdir build
cd build
cmake -DCONDUIT_DIR=${TRAVIS_BUILD_DIR}/travis-debug-install ../
make
./example
##########################################
# using with make example
##########################################
# find spack installed HDF5_DIR
export HDF5_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/hdf5*`
cd ${TRAVIS_BUILD_DIR}/travis-debug-install/examples/using-with-make
env CXX=${COMPILER_CXX} CONDUIT_DIR=${TRAVIS_BUILD_DIR}/travis-debug-install HDF5_DIR=${HDF5_DIR} make
env LD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/travis-debug-install/lib/:${HDF5_DIR}/lib ./example

