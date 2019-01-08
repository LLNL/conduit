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
cd ${TRAVIS_BUILD_DIR}/travis-debug-install/examples/conduit/using-with-cmake
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
cd ${TRAVIS_BUILD_DIR}/travis-debug-install/examples/conduit/using-with-make
if [ "${ENABLE_ADIOS}" == "ON" ]; then
    # find spack installed ADIOS_DIR
    export ADIOS_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/adios*`
    export ZFP_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/zfp*`
    export LZ4_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/lz4*`
    export BLOSC_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/c-blosc*`
    export SZ_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/sz*`
    env CXX=${COMPILER_CXX} CONDUIT_DIR=${TRAVIS_BUILD_DIR}/travis-debug-install ADIOS_DIR=${ADIOS_DIR} ZFP_DIR=${ZFP_DIR} LZ4_DIR=${LZ4_DIR} BLOSC_DIR=${BLOSC_DIR} SZ_DIR=${SZ_DIR} make with_adios
    env LD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/travis-debug-install/lib/:${HDF5_DIR}/lib:${ADIOS_DIR}/lib:${ZFP_DIR}/lib:${LZ4_DIR}/lib:${BLOSC_DIR}/lib:${SZ_DIR}/lib ./example
else
    env CXX=${COMPILER_CXX} CONDUIT_DIR=${TRAVIS_BUILD_DIR}/travis-debug-install make
    env LD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/travis-debug-install/lib/:${HDF5_DIR}/lib ./example
fi

